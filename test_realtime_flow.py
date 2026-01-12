#!/usr/bin/env python3
"""
Simulates a realistic jam session flow against the middleware.

This mimics what the iOS app does:
1. Login and check credits
2. Warmup the Modal container
3. Start a jam with a loop
4. Continuously fetch and consume chunks (simulating real-time playback)
5. Stop the jam
6. Close the session
7. Verify billing was applied correctly
"""

import asyncio
import httpx
import sys
import time

BASE_URL = "http://localhost:8000"

# A minimal valid WAV file (silence, 1 second, 48kHz stereo)
# In real usage, this would be actual audio from the iOS app
def make_silent_wav() -> bytes:
    import struct
    import wave
    import io

    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(48000)
        # 1 second of silence
        wav.writeframes(b'\x00\x00\x00\x00' * 48000)
    return buffer.getvalue()


async def main():
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=120.0) as client:
        print("=" * 60)
        print("DARIUS MIDDLEWARE - REAL-TIME FLOW TEST")
        print("=" * 60)

        # 1. Login
        print("\n[1] Logging in...")
        resp = await client.post(
            "/auth/mock-login",
            json={"test_user_id": "realtime_test_user"}
        )
        resp.raise_for_status()
        data = resp.json()
        token = data["token"]
        print(f"    Token: {token[:50]}...")

        headers = {"Authorization": f"Bearer {token}"}

        # 2. Check initial balance
        print("\n[2] Checking initial balance...")
        resp = await client.get("/credits/balance", headers=headers)
        initial_balance = float(resp.json()["balance_seconds"])
        print(f"    Balance: {initial_balance} seconds")

        # 3. Warmup
        print("\n[3] Warming up Modal container...")
        print("    (This may take ~60 seconds on cold start)")
        start = time.time()
        resp = await client.post("/jam/warmup", headers=headers)
        resp.raise_for_status()
        warmup_data = resp.json()
        session_id = warmup_data["session_id"]
        elapsed = time.time() - start
        print(f"    Session ID: {session_id}")
        print(f"    Warmed: {warmup_data.get('warmed')}")
        print(f"    Warmup took: {elapsed:.1f}s")

        # 4. Check status
        print("\n[4] Checking session status...")
        resp = await client.get(f"/jam/status?session_id={session_id}", headers=headers)
        status = resp.json()
        print(f"    Billing state: {status['billing_state']}")
        print(f"    Billable elapsed: {status['billable_elapsed_seconds']}")

        # 5. Start jam
        print("\n[5] Starting jam...")
        loop_audio = make_silent_wav()
        resp = await client.post(
            "/jam/start",
            headers=headers,
            data={
                "session_id": session_id,
                "bpm": "120",
                "bars_per_chunk": "4",
                "beats_per_bar": "4",
                "styles": "",
                "style_weights": "",
                "loop_weight": "1.0",
                "loudness_mode": "auto",
                "loudness_headroom_db": "1.0",
                "guidance_weight": "1.1",
                "temperature": "1.1",
                "topk": "40",
            },
            files={"loop_audio": ("loop.wav", loop_audio, "audio/wav")},
        )
        resp.raise_for_status()
        print(f"    Status: {resp.json().get('status')}")

        # 6. Simulate real-time playback: fetch and consume chunks
        # Each chunk is 4 bars at 120 BPM = 8 seconds of audio
        # We'll simulate fetching 4 chunks (32 seconds of music)
        print("\n[6] Simulating real-time playback (4 chunks)...")

        for i in range(4):
            # Fetch next chunk
            print(f"\n    Chunk {i}:")
            fetch_start = time.time()
            resp = await client.get(
                f"/jam/next?session_id={session_id}&timeout_seconds=10",
                headers=headers,
            )
            resp.raise_for_status()
            chunk_data = resp.json()
            fetch_time = time.time() - fetch_start

            if chunk_data.get("chunk"):
                chunk = chunk_data["chunk"]
                audio_len = len(chunk.get("audio_base64", ""))
                print(f"      Fetched in {fetch_time:.2f}s")
                print(f"      Index: {chunk['index']}, Audio size: {audio_len} bytes (base64)")

                # Consume the chunk
                resp = await client.post(
                    "/jam/consume",
                    headers=headers,
                    data={"session_id": session_id, "chunk_index": str(chunk["index"])},
                )
                resp.raise_for_status()
                print(f"      Consumed chunk {chunk['index']}")

                # Simulate playback time (in real app, this would be the audio duration)
                # We'll wait a short time to let billing accumulate
                await asyncio.sleep(2.0)
            else:
                print(f"      No chunk available")

        # 7. Check status mid-session
        print("\n[7] Mid-session status check...")
        resp = await client.get(f"/jam/status?session_id={session_id}", headers=headers)
        status = resp.json()
        print(f"    Billing state: {status['billing_state']}")
        print(f"    Billable elapsed: {float(status['billable_elapsed_seconds']):.1f}s")
        print(f"    Credits remaining: {float(status['credits_remaining']):.1f}s")
        print(f"    Jam running: {status['jam_running']}")

        # 8. Test parameter update
        print("\n[8] Testing parameter hot-swap...")
        resp = await client.post(
            "/jam/update",
            headers=headers,
            data={
                "session_id": session_id,
                "temperature": "1.3",
                "guidance_weight": "1.5",
            },
        )
        resp.raise_for_status()
        print(f"    Update result: {resp.json()}")

        # 9. Stop jam
        print("\n[9] Stopping jam...")
        resp = await client.post(
            "/jam/stop",
            headers=headers,
            data={"session_id": session_id},
        )
        resp.raise_for_status()
        print(f"    Status: {resp.json().get('status')}")

        # 10. Close session
        print("\n[10] Closing session...")
        resp = await client.post(
            "/jam/close",
            headers=headers,
            data={"session_id": session_id},
        )
        resp.raise_for_status()
        close_data = resp.json()
        print(f"    Status: {close_data.get('status')}")
        print(f"    Total charged: {float(close_data.get('total_charged', 0)):.2f}s")

        # 11. Wait for billing loop to finalize
        print("\n[11] Waiting for billing loop to finalize...")
        await asyncio.sleep(6)

        # 12. Check final balance
        print("\n[12] Final balance check...")
        resp = await client.get("/credits/balance", headers=headers)
        final_balance = float(resp.json()["balance_seconds"])
        credits_used = initial_balance - final_balance

        print(f"    Initial balance: {initial_balance:.2f}s")
        print(f"    Final balance: {final_balance:.2f}s")
        print(f"    Credits used: {credits_used:.2f}s")

        print("\n" + "=" * 60)
        print("TEST COMPLETE")
        print("=" * 60)

        # Sanity check
        if credits_used > 0:
            print("\n✅ Billing is working - credits were deducted")
        else:
            print("\n❌ WARNING: No credits deducted - check billing loop")

        return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
