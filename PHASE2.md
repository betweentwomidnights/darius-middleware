# Phase 2: Real Modal Client

## Overview

Replace `MockModalClient` with a real HTTP client that talks to the deployed Modal backend. The Modal API is now protected with an API key, which is already configured in `.env`.

## Prerequisites

The `.env` file should have these values (already configured):
```
MODAL_API_URL=https://the-collabage-patch--modal-darius-web.modal.run
DARIUS_API_KEY=<the-secret-key>
```

## Task 1: Update Config

In `app/config.py`, add the new settings if not already present:

```python
modal_api_url: str = ""
darius_api_key: str = ""
```

## Task 2: Implement Real Modal Client

Replace `app/modal_client.py` with a real implementation using `httpx`. Key requirements:

### Authentication
Every request (except /health) must include the header:
```
X-API-Key: <DARIUS_API_KEY>
```

### Endpoints to Implement

| Method | Endpoint | Request Type | Notes |
|--------|----------|--------------|-------|
| POST | /warmup | Form (optional session_id) | Returns session_id if not provided |
| POST | /status | Form (session_id, keepalive) | keepalive=0 for peek, 1 to extend |
| GET | /status/peek | Query (session_id) | Alias for status keepalive=0 |
| POST | /jam/start | Multipart Form | Includes loop_audio file |
| GET | /jam/next | Query (session_id, timeout_seconds) | Returns audio chunk |
| POST | /jam/consume | Form (session_id, chunk_index) | |
| POST | /jam/stop | Form (session_id) | |
| POST | /jam/update | Form (session_id, styles, etc.) | Hot-swap params |
| POST | /jam/reseed | Multipart Form (optional loop_audio) | |
| POST | /jam/reseed_splice | Multipart Form (optional combined_audio) | |
| POST | /session/close | Form (session_id, reason) | |

### Client Structure

```python
import httpx
from app.config import settings

class ModalClient:
    """Real Modal API client."""
    
    def __init__(self):
        self.base_url = settings.modal_api_url.rstrip("/")
        self.api_key = settings.darius_api_key
        self._client: httpx.AsyncClient | None = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"X-API-Key": self.api_key},
                timeout=httpx.Timeout(120.0, connect=10.0),  # Long timeout for warmup
            )
        return self._client
    
    async def warmup(self, session_id: str | None = None) -> dict:
        client = await self._get_client()
        data = {}
        if session_id:
            data["session_id"] = session_id
        response = await client.post("/warmup", data=data)
        response.raise_for_status()
        return response.json()
    
    async def status_peek(self, session_id: str) -> dict:
        client = await self._get_client()
        response = await client.get("/status/peek", params={"session_id": session_id})
        response.raise_for_status()
        return response.json()
    
    async def jam_start(
        self,
        session_id: str,
        loop_audio: bytes,
        bpm: int = 120,
        bars_per_chunk: int = 4,
        beats_per_bar: int = 4,
        styles: str = "",
        style_weights: str = "",
        loop_weight: float = 1.0,
        loudness_mode: str = "auto",
        loudness_headroom_db: float = 1.0,
        guidance_weight: float = 1.1,
        temperature: float = 1.1,
        topk: int = 40,
        target_sample_rate: int | None = None,
    ) -> dict:
        client = await self._get_client()
        
        data = {
            "session_id": session_id,
            "bpm": str(bpm),
            "bars_per_chunk": str(bars_per_chunk),
            "beats_per_bar": str(beats_per_bar),
            "styles": styles,
            "style_weights": style_weights,
            "loop_weight": str(loop_weight),
            "loudness_mode": loudness_mode,
            "loudness_headroom_db": str(loudness_headroom_db),
            "guidance_weight": str(guidance_weight),
            "temperature": str(temperature),
            "topk": str(topk),
            "target_sample_rate": str(target_sample_rate) if target_sample_rate else "",
        }
        
        files = {"loop_audio": ("loop.wav", loop_audio, "audio/wav")}
        
        response = await client.post("/jam/start", data=data, files=files)
        response.raise_for_status()
        return response.json()
    
    async def jam_next(self, session_id: str, timeout_seconds: int = 10) -> dict:
        client = await self._get_client()
        response = await client.get(
            "/jam/next",
            params={"session_id": session_id, "timeout_seconds": timeout_seconds},
        )
        response.raise_for_status()
        return response.json()
    
    async def jam_consume(self, session_id: str, chunk_index: int) -> dict:
        client = await self._get_client()
        response = await client.post(
            "/jam/consume",
            data={"session_id": session_id, "chunk_index": str(chunk_index)},
        )
        response.raise_for_status()
        return response.json()
    
    async def jam_stop(self, session_id: str) -> dict:
        client = await self._get_client()
        response = await client.post("/jam/stop", data={"session_id": session_id})
        response.raise_for_status()
        return response.json()
    
    async def jam_update(
        self,
        session_id: str,
        styles: str = "",
        style_weights: str = "",
        loop_weight: float | None = None,
        guidance_weight: float | None = None,
        temperature: float | None = None,
        topk: int | None = None,
    ) -> dict:
        client = await self._get_client()
        data = {"session_id": session_id, "styles": styles, "style_weights": style_weights}
        if loop_weight is not None:
            data["loop_weight"] = str(loop_weight)
        if guidance_weight is not None:
            data["guidance_weight"] = str(guidance_weight)
        if temperature is not None:
            data["temperature"] = str(temperature)
        if topk is not None:
            data["topk"] = str(topk)
        
        response = await client.post("/jam/update", data=data)
        response.raise_for_status()
        return response.json()
    
    async def jam_reseed(self, session_id: str, loop_audio_bytes: bytes | None = None) -> dict:
        client = await self._get_client()
        data = {"session_id": session_id}
        files = None
        if loop_audio_bytes:
            files = {"loop_audio": ("loop.wav", loop_audio_bytes, "audio/wav")}
        
        response = await client.post("/jam/reseed", data=data, files=files)
        response.raise_for_status()
        return response.json()
    
    async def jam_reseed_splice(
        self,
        session_id: str,
        anchor_bars: float = 2.0,
        combined_audio_bytes: bytes | None = None,
    ) -> dict:
        client = await self._get_client()
        data = {"session_id": session_id, "anchor_bars": str(anchor_bars)}
        files = None
        if combined_audio_bytes:
            files = {"combined_audio": ("combined.wav", combined_audio_bytes, "audio/wav")}
        
        response = await client.post("/jam/reseed_splice", data=data, files=files)
        response.raise_for_status()
        return response.json()
    
    async def session_close(self, session_id: str, reason: str = "manual_close") -> dict:
        client = await self._get_client()
        response = await client.post(
            "/session/close",
            data={"session_id": session_id, "reason": reason},
        )
        response.raise_for_status()
        return response.json()
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Global client instance
_modal_client: ModalClient | None = None


def get_modal_client() -> ModalClient:
    """Get or create the global Modal client."""
    global _modal_client
    if _modal_client is None:
        _modal_client = ModalClient()
    return _modal_client
```

### Error Handling

Wrap httpx exceptions and return consistent error responses:

```python
from httpx import HTTPStatusError, RequestError

try:
    response = await client.post(...)
    response.raise_for_status()
    return response.json()
except HTTPStatusError as e:
    return {"ok": False, "error": f"HTTP {e.response.status_code}: {e.response.text}"}
except RequestError as e:
    return {"ok": False, "error": f"Request failed: {str(e)}"}
```

## Task 3: Update Lifespan to Close Client

In `app/main.py`, update the lifespan to properly close the httpx client:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start billing loop
    task = asyncio.create_task(billing_loop())
    yield
    # Shutdown
    task.cancel()
    # Close Modal client
    client = get_modal_client()
    await client.close()
```

## Task 4: Create Real-Time Test Script

Create `test_realtime_flow.py` that simulates a realistic jam session:

```python
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
```

## Task 5: Run Verification

After implementing the real client:

1. Make sure `.env` has the correct `MODAL_API_URL` and `DARIUS_API_KEY`
2. Start the middleware: `uvicorn app.main:app --reload --port 8000`
3. Run the test: `python test_realtime_flow.py`

Expected output:
- Warmup should take ~60s on cold start, ~2s if Modal container is still warm
- Each `/jam/next` should return real base64 audio (~500KB-1MB)
- Credits should be deducted based on actual time elapsed
- Final balance should be lower than initial balance

## Notes

### Timeout Considerations
- Warmup can take 60+ seconds on cold start
- Set httpx timeout to at least 120 seconds for warmup
- Individual jam operations are fast (<5s typically)

### The Real Audio Loop
- The test script uses a silent WAV file
- In production, the iOS app sends real audio (the combined loop)
- The audio quality doesn't affect billing, just generation output

### Billing Accuracy
- Middleware billing may lag slightly behind Modal's real elapsed time
- This is expected - the billing loop polls every 5 seconds
- Final reconciliation happens on session close