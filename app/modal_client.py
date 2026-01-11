"""Mock Modal client for Phase 1 development."""

import asyncio
import base64
import time
from typing import Any


class MockModalClient:
    """Simulates Modal API responses for local development."""

    def __init__(self):
        self._sessions: dict[str, dict[str, Any]] = {}
        self._chunks: dict[str, list[dict]] = {}  # session_id -> list of chunks

    async def warmup(self, session_id: str) -> dict:
        """Simulate /warmup endpoint with 2-second delay."""
        await asyncio.sleep(2.0)
        now = time.time()
        self._sessions[session_id] = {
            "warmed": True,
            "billing_state": "active",
            "billable_start_ts": now,
            "last_activity_ts": now,
            "jam_running": False,
            "created_ts": now - 2.0,
        }
        return {
            "ok": True,
            "session_id": session_id,
            "warmed": True,
            "container_warmup_seconds": 2.0,
        }

    async def status_peek(self, session_id: str) -> dict:
        sess = self._sessions.get(session_id)
        if not sess:
            return {"ok": False, "error": "Session not found"}

        now = time.time()
        billable_start = sess.get("billable_start_ts", now)
        
        # If closed, freeze elapsed at the close time, don't return 0
        if sess.get("billing_state") == "closed":
            closed_activity_ts = sess.get("closed_last_activity_ts", billable_start)
            elapsed = closed_activity_ts - billable_start
        else:
            elapsed = now - billable_start

        return {
            "ok": True,
            "session_id": session_id,
            "billing_state": sess.get("billing_state", "warming"),
            "billable_elapsed_seconds": elapsed,
            "jam_running": sess.get("jam_running", False),
        }

    async def jam_start(
        self,
        session_id: str,
        loop_audio: bytes,
        bpm: int = 120,
        bars_per_chunk: int = 4,
        **kwargs,
    ) -> dict:
        """Simulate /jam/start endpoint."""
        sess = self._sessions.get(session_id)
        if not sess:
            return {"ok": False, "error": "Session not found"}

        now = time.time()
        sess["jam_running"] = True
        sess["last_activity_ts"] = now
        sess["bpm"] = bpm
        sess["bars_per_chunk"] = bars_per_chunk

        # Initialize chunk queue with some fake chunks
        self._chunks[session_id] = []
        for i in range(3):  # Pre-generate 3 chunks
            self._chunks[session_id].append({
                "index": i,
                "audio_base64": base64.b64encode(b"fake_audio_data").decode(),
                "metadata": {
                    "bpm": bpm,
                    "bars": bars_per_chunk,
                    "beats_per_bar": 4,
                    "sample_rate": 48000,
                    "channels": 2,
                    "total_samples": 384000,
                    "seconds_per_bar": 2.0,
                    "loop_duration_seconds": 8.0,
                    "guidance_weight": 1.1,
                    "temperature": 1.1,
                    "topk": 40,
                },
            })

        return {
            "ok": True,
            "session_id": session_id,
            "status": "started",
        }

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
        """Simulate /jam/update endpoint."""

        sess = self._sessions.get(session_id)
        if not sess:
            return {"ok": False, "error": "Session not found"}

        sess["last_activity_ts"] = time.time()

        if styles:
            sess["styles"] = styles
        if style_weights:
            sess["style_weights"] = style_weights
        if loop_weight is not None:
            sess["loop_weight"] = loop_weight
        if guidance_weight is not None:
            sess["guidance_weight"] = guidance_weight
        if temperature is not None:
            sess["temperature"] = temperature
        if topk is not None:
            sess["topk"] = topk

        return {"ok": True, "session_id": session_id}

    async def jam_next(self, session_id: str, timeout_seconds: int = 10) -> dict:
        """Simulate /jam/next endpoint."""
        sess = self._sessions.get(session_id)
        if not sess or not sess.get("jam_running"):
            return {"ok": False, "error": "Jam not running"}

        chunks = self._chunks.get(session_id, [])
        if not chunks:
            return {"ok": False, "error": "No chunks available"}

        # Return first available chunk
        chunk = chunks[0]
        return {
            "ok": True,
            "chunk": chunk,
        }

    async def jam_consume(self, session_id: str, chunk_index: int) -> dict:
        """Simulate /jam/consume endpoint."""
        sess = self._sessions.get(session_id)
        if not sess:
            return {"ok": False, "error": "Session not found"}

        now = time.time()
        sess["last_activity_ts"] = now

        # Remove consumed chunk and generate a new one
        chunks = self._chunks.get(session_id, [])
        chunks = [c for c in chunks if c["index"] != chunk_index]

        # Generate next chunk
        next_index = chunk_index + 3
        chunks.append({
            "index": next_index,
            "audio_base64": base64.b64encode(b"fake_audio_data").decode(),
            "metadata": {
                "bpm": sess.get("bpm", 120),
                "bars": sess.get("bars_per_chunk", 4),
                "beats_per_bar": 4,
                "sample_rate": 48000,
                "channels": 2,
                "total_samples": 384000,
                "seconds_per_bar": 2.0,
                "loop_duration_seconds": 8.0,
                "guidance_weight": 1.1,
                "temperature": 1.1,
                "topk": 40,
            },
        })
        self._chunks[session_id] = chunks

        return {"ok": True}

    async def jam_stop(self, session_id: str) -> dict:
        """Simulate /jam/stop endpoint."""
        sess = self._sessions.get(session_id)
        if not sess:
            return {"ok": False, "error": "Session not found"}

        now = time.time()
        sess["jam_running"] = False
        sess["last_activity_ts"] = now

        return {
            "ok": True,
            "session_id": session_id,
            "status": "stopped",
        }

    async def jam_reseed(
        self,
        session_id: str,
        loop_audio_bytes: bytes | None = None,
    ) -> dict:
        """Simulate /jam/reseed endpoint."""

        sess = self._sessions.get(session_id)
        if not sess:
            return {"ok": False, "error": "Session not found"}

        sess["last_activity_ts"] = time.time()
        if loop_audio_bytes is not None:
            sess["last_loop_audio_bytes"] = len(loop_audio_bytes)

        return {"ok": True, "session_id": session_id}

    async def jam_reseed_splice(
        self,
        session_id: str,
        anchor_bars: float = 2.0,
        combined_audio_bytes: bytes | None = None,
    ) -> dict:
        """Simulate /jam/reseed_splice endpoint."""

        sess = self._sessions.get(session_id)
        if not sess:
            return {"ok": False, "error": "Session not found"}

        sess["last_activity_ts"] = time.time()
        sess["last_anchor_bars"] = anchor_bars
        if combined_audio_bytes is not None:
            sess["last_combined_audio_bytes"] = len(combined_audio_bytes)

        return {
            "ok": True,
            "session_id": session_id,
            "anchor_bars": anchor_bars,
        }

    async def session_close(self, session_id: str) -> dict:
        """Simulate /session/close endpoint."""
        sess = self._sessions.get(session_id)
        if not sess:
            return {"ok": False, "error": "Session not found"}

        now = time.time()
        sess["billing_state"] = "closed"
        sess["closed_ts"] = now
        sess["closed_last_activity_ts"] = sess.get("last_activity_ts", now)

        return {
            "ok": True,
            "session_id": session_id,
            "status": "closed",
            "closed_ts": now,
            "billing_state": "closed",
        }


# Global mock client instance
_mock_client: MockModalClient | None = None


def get_modal_client() -> MockModalClient:
    """Get or create the global mock modal client."""
    global _mock_client
    if _mock_client is None:
        _mock_client = MockModalClient()
    return _mock_client
