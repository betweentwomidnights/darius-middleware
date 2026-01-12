"""Real Modal API client using httpx."""

import httpx
from typing import Any

from app.config import settings


class ModalClient:
    """Real Modal API client."""

    def __init__(self):
        self.base_url = settings.modal_api_url.rstrip("/")
        self.api_key = settings.darius_api_key
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"X-API-Key": self.api_key},
                timeout=httpx.Timeout(120.0, connect=10.0),  # Long timeout for warmup
            )
        return self._client

    async def warmup(self, session_id: str | None = None) -> dict[str, Any]:
        """Call /warmup endpoint."""
        try:
            client = await self._get_client()
            data = {}
            if session_id:
                data["session_id"] = session_id
            response = await client.post("/warmup", data=data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"ok": False, "error": f"HTTP {e.response.status_code}: {e.response.text}"}
        except httpx.RequestError as e:
            return {"ok": False, "error": f"Request failed: {str(e)}"}

    async def status_peek(self, session_id: str) -> dict[str, Any]:
        """Call /status/peek endpoint (does not extend container lifetime)."""
        try:
            client = await self._get_client()
            response = await client.get("/status/peek", params={"session_id": session_id})
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"ok": False, "error": f"HTTP {e.response.status_code}: {e.response.text}"}
        except httpx.RequestError as e:
            return {"ok": False, "error": f"Request failed: {str(e)}"}

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
    ) -> dict[str, Any]:
        """Call /jam/start endpoint with multipart form data."""
        try:
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
        except httpx.HTTPStatusError as e:
            return {"ok": False, "error": f"HTTP {e.response.status_code}: {e.response.text}"}
        except httpx.RequestError as e:
            return {"ok": False, "error": f"Request failed: {str(e)}"}

    async def jam_next(self, session_id: str, timeout_seconds: int = 10) -> dict[str, Any]:
        """Call /jam/next endpoint."""
        try:
            client = await self._get_client()
            response = await client.get(
                "/jam/next",
                params={"session_id": session_id, "timeout_seconds": timeout_seconds},
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"ok": False, "error": f"HTTP {e.response.status_code}: {e.response.text}"}
        except httpx.RequestError as e:
            return {"ok": False, "error": f"Request failed: {str(e)}"}

    async def jam_consume(self, session_id: str, chunk_index: int) -> dict[str, Any]:
        """Call /jam/consume endpoint."""
        try:
            client = await self._get_client()
            response = await client.post(
                "/jam/consume",
                data={"session_id": session_id, "chunk_index": str(chunk_index)},
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"ok": False, "error": f"HTTP {e.response.status_code}: {e.response.text}"}
        except httpx.RequestError as e:
            return {"ok": False, "error": f"Request failed: {str(e)}"}

    async def jam_stop(self, session_id: str) -> dict[str, Any]:
        """Call /jam/stop endpoint."""
        try:
            client = await self._get_client()
            response = await client.post("/jam/stop", data={"session_id": session_id})
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"ok": False, "error": f"HTTP {e.response.status_code}: {e.response.text}"}
        except httpx.RequestError as e:
            return {"ok": False, "error": f"Request failed: {str(e)}"}

    async def jam_update(
        self,
        session_id: str,
        styles: str = "",
        style_weights: str = "",
        loop_weight: float | None = None,
        guidance_weight: float | None = None,
        temperature: float | None = None,
        topk: int | None = None,
    ) -> dict[str, Any]:
        """Call /jam/update endpoint."""
        try:
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
        except httpx.HTTPStatusError as e:
            return {"ok": False, "error": f"HTTP {e.response.status_code}: {e.response.text}"}
        except httpx.RequestError as e:
            return {"ok": False, "error": f"Request failed: {str(e)}"}

    async def jam_reseed(self, session_id: str, loop_audio_bytes: bytes | None = None) -> dict[str, Any]:
        """Call /jam/reseed endpoint."""
        try:
            client = await self._get_client()
            data = {"session_id": session_id}
            files = None
            if loop_audio_bytes:
                files = {"loop_audio": ("loop.wav", loop_audio_bytes, "audio/wav")}

            response = await client.post("/jam/reseed", data=data, files=files)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"ok": False, "error": f"HTTP {e.response.status_code}: {e.response.text}"}
        except httpx.RequestError as e:
            return {"ok": False, "error": f"Request failed: {str(e)}"}

    async def jam_reseed_splice(
        self,
        session_id: str,
        anchor_bars: float = 2.0,
        combined_audio_bytes: bytes | None = None,
    ) -> dict[str, Any]:
        """Call /jam/reseed_splice endpoint."""
        try:
            client = await self._get_client()
            data = {"session_id": session_id, "anchor_bars": str(anchor_bars)}
            files = None
            if combined_audio_bytes:
                files = {"combined_audio": ("combined.wav", combined_audio_bytes, "audio/wav")}

            response = await client.post("/jam/reseed_splice", data=data, files=files)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"ok": False, "error": f"HTTP {e.response.status_code}: {e.response.text}"}
        except httpx.RequestError as e:
            return {"ok": False, "error": f"Request failed: {str(e)}"}

    async def session_close(self, session_id: str, reason: str = "manual_close") -> dict[str, Any]:
        """Call /session/close endpoint."""
        try:
            client = await self._get_client()
            response = await client.post(
                "/session/close",
                data={"session_id": session_id, "reason": reason},
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"ok": False, "error": f"HTTP {e.response.status_code}: {e.response.text}"}
        except httpx.RequestError as e:
            return {"ok": False, "error": f"Request failed: {str(e)}"}

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
