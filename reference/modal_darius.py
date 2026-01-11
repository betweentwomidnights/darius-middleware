
import os
import time
import uuid
from typing import Optional, List, Dict, Any

import modal

APP_NAME = "modal-darius"

# ---------------------------------------------------------------------
# Persistent cache root (single Volume mount point!)
# ---------------------------------------------------------------------
CACHE_ROOT = "/cache"
HF_CACHE_DIR = f"{CACHE_ROOT}/huggingface"
MRT_CACHE_DIR = f"{CACHE_ROOT}/magenta_rt"
JAX_CACHE_DIR = f"{CACHE_ROOT}/jax"

cache_vol = modal.Volume.from_name("darius-mrt-cache", create_if_missing=True)

# ---------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------
GPU_IMAGE = modal.Image.from_dockerfile("Dockerfile.x86")

WEB_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.27.0",
        "python-multipart>=0.0.9",
    )
)

app = modal.App(APP_NAME)

# ---------------------------------------------------------------------
# Session metadata (for status/credit plumbing)
# ---------------------------------------------------------------------
# modal.Dict is a persistent, distributed key-value store accessible from both
# the GPU session containers and the CPU web router. Use primitive types.
SCALEDOWN_WINDOW_SECONDS = 60 * 10  # keep warm 10m after last request
SESSION_META_DICT_NAME = f"{APP_NAME}-session-meta"
session_meta = modal.Dict.from_name(SESSION_META_DICT_NAME, create_if_missing=True)

# API key secret for authenticating middleware requests
api_secret = modal.Secret.from_name("darius-api-key")

def _parse_csv_strings(s: str) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_csv_floats(s: str) -> List[float]:
    if not s:
        return []
    out: List[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            pass
    return out


def _patch_t5x_for_gpu_coords() -> None:
    """
    Patch t5x partitioning helpers so they work on GPU-only devices where TPU-only
    attrs like `core_on_chip` may be missing.
    """
    try:
        import logging
        import jax
        from t5x import partitioning as _t5x_part

        def _bounds_from_last_device_gpu_safe(last_device):
            core = getattr(last_device, "core_on_chip", None)
            coords = getattr(last_device, "coords", None)

            # TPU path
            if coords is not None and core is not None:
                x, y, z = coords
                return x + 1, y + 1, z + 1, core + 1

            # GPU/CPU fallback
            proc_count = getattr(jax, "process_count", None)
            if callable(proc_count):
                return jax.process_count(), jax.local_device_count()
            return jax.host_count(), jax.local_device_count()

        def _get_coords_gpu_safe(device):
            core = getattr(device, "core_on_chip", None)
            coords = getattr(device, "coords", None)

            # TPU path
            if coords is not None and core is not None:
                return (*coords, core)

            # GPU/CPU fallback
            return (device.process_index, device.id % jax.local_device_count())

        _t5x_part.bounds_from_last_device = _bounds_from_last_device_gpu_safe
        _t5x_part.get_coords = _get_coords_gpu_safe

        logging.info("Patched t5x.partitioning for GPU coords/core_on_chip incompat.")
    except Exception as e:
        import logging

        logging.exception("t5x GPU-coords patch failed: %s", e)


# ---------------------------------------------------------------------
# GPU session class (one container per session_id)
# ---------------------------------------------------------------------
@app.cls(
    image=GPU_IMAGE,
    gpu="l40s",

    # Per-request max runtime (NOT idle lifetime)
    timeout=60 * 10,

    # Keep the container warm after the last request
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,

    min_containers=0,
    max_containers=3,

    # Snapshots OFF (avoid SIGSEGV with XLA/JAX/T5X)
    enable_memory_snapshot=False,

    # Single mount point only!
    volumes={CACHE_ROOT: cache_vol},
)
class DariusSession:
    session_id: str = modal.parameter()

    # ------------------------------
    # Meta helpers
    # ------------------------------
    def _meta_update(self, **kv: Any) -> None:
        """Best-effort write of session metadata to a shared modal.Dict."""
        try:
            cur = session_meta.get(self.session_id, {}) or {}
            if not isinstance(cur, dict):
                cur = {}
            cur.update(kv)
            session_meta[self.session_id] = cur
        except Exception:
            # never let meta bookkeeping break core functionality
            pass

    def _meta_snapshot(self) -> Dict[str, Any]:
        try:
            cur = session_meta.get(self.session_id, {}) or {}
            return cur if isinstance(cur, dict) else {}
        except Exception:
            return {}


    @modal.enter()
    def _load_and_warm(self) -> None:
        import threading
        import logging

        if getattr(self, "_warmed", False):
            return

        logging.basicConfig(level=logging.INFO)
        self._log = logging.getLogger(f"darius[{self.session_id}]")

        # Make sure persistent dirs exist
        os.makedirs(HF_CACHE_DIR, exist_ok=True)
        os.makedirs(MRT_CACHE_DIR, exist_ok=True)
        os.makedirs(JAX_CACHE_DIR, exist_ok=True)

        # Force caches into the mounted volume
        os.environ["XDG_CACHE_HOME"] = CACHE_ROOT

        # HuggingFace Hub cache
        os.environ["HF_HOME"] = HF_CACHE_DIR
        os.environ["HUGGINGFACE_HUB_CACHE"] = HF_CACHE_DIR
        os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR

        # MagentaRT cache (your code + any internal usage)
        os.environ["MAGENTA_RT_CACHE_DIR"] = MRT_CACHE_DIR

        # JAX/XLA compilation cache (persist across cold starts)
        os.environ["JAX_COMPILATION_CACHE_DIR"] = JAX_CACHE_DIR

        # Avoid probing TPU backends (prevents libtpu.so warnings)
        # Must be set before importing jax/t5x.
        os.environ.setdefault("JAX_PLATFORMS", "cuda")

        # Optional GPU-friendly XLA flags (kept as defaults; override if you want)
        os.environ.setdefault(
            "XLA_FLAGS",
            " ".join(
                [
                    "--xla_gpu_enable_triton_gemm=true",
                    "--xla_gpu_enable_latency_hiding_scheduler=true",
                    "--xla_gpu_autotune_level=2",
                ]
            ),
        )

        # Patch T5X partitioning before model touches it
        _patch_t5x_for_gpu_coords()

        self._worker_lock = threading.Lock()
        self._worker = None

        self._warmed = False
        self._warmup_seconds = None

        # "Touch" vs "activity":
        # - last_touched_ts: updated on every request (observational; mirrors Modal's real idle timer behavior)
        # - last_activity_ts: updated only while the session is billable (warmed and not closed)
        self._last_touched_ts = None
        self._last_activity_ts = None

        # Billing markers (we do not bill for warmup time)
        self._billable_start_ts = None
        self._closed_last_activity_ts = None
        self._billable_end_estimated_ts = None

        self._session_created_ts = getattr(self, "_session_created_ts", time.time())
        self._warm_started_ts = time.time()
        self._warm_completed_ts = None
        self._closed_ts = None
        self._meta_update(
            session_id=self.session_id,
            session_created_ts=self._session_created_ts,
            warm_started_ts=self._warm_started_ts,
            warm_completed_ts=None,

            # "warm" + runtime state
            warmed=False,
            jam_running=False,

            # close/billing state
            closed=False,
            closed_ts=None,
            close_reason="",

            # idle tracking (observational)
            last_touched_ts=None,
            estimated_scaledown_ts=None,

            # billable tracking (do not bill warmup)
            last_activity_ts=None,
            billable_start_ts=None,
            closed_last_activity_ts=None,
            billable_end_estimated_ts=None,
            billable_elapsed_seconds=None,
            billing_state="warming",

            scaledown_window_seconds=SCALEDOWN_WINDOW_SECONDS,
            tag=os.getenv("MRT_TAG", "large"),
        )


        from magenta_rt import system
        from model_management import CheckpointManager

        ckpt_dir = CheckpointManager.resolve_checkpoint_dir()
        tag = os.getenv("MRT_TAG", "large")

        self._log.info("Loading MagentaRT tag=%s ckpt_dir=%r ...", tag, ckpt_dir)
        t0 = time.time()

        self._mrt = system.MagentaRT(
            tag=tag,
            device="gpu",
            checkpoint_dir=ckpt_dir,
        )

        self._mrt_warmup()

        self._warmup_seconds = time.time() - t0
        self._warm_completed_ts = time.time()
        self._warmed = True

        # Billing starts when warm completes (we do not bill warmup time).
        self._billable_start_ts = self._warm_completed_ts
        self._last_activity_ts = self._warm_completed_ts
        self._closed_last_activity_ts = None
        self._billable_end_estimated_ts = self._last_activity_ts + SCALEDOWN_WINDOW_SECONDS

        # Publish warm completion into shared meta so /status keepalive=0 is accurate.
        self._meta_update(
            warmed=True,
            warm_completed_ts=self._warm_completed_ts,
            container_warmup_seconds=self._warmup_seconds,
            billable_start_ts=self._billable_start_ts,
            last_activity_ts=self._last_activity_ts,
            billable_end_estimated_ts=self._billable_end_estimated_ts,
            billing_state="active",
        )

        # This request itself should count as activity (the container is now warm & billable).
        self._touch(count_activity=True)

        # Persist whatever was downloaded/compiled into the volume
        cache_vol.commit()

        self._log.info("Warmup complete in %.2fs", self._warmup_seconds)

    def _touch(self, *, count_activity: bool = True) -> None:
        """Record a request 'touch'.

        - Always updates last_touched_ts (observational).
        - Updates last_activity_ts only when:
            * the session is warmed, AND
            * the session is not closed, AND
            * count_activity=True

        This lets /status keepalive=0 and your AWS middleware reason about billable time
        without being confused by 'peek' calls or post-close pings.
        """
        now = time.time()
        self._last_touched_ts = now

        # Billable activity tracking
        if count_activity and getattr(self, "_warmed", False) and not getattr(self, "_closed_ts", None):
            self._last_activity_ts = now

        # Estimate billable end time from the *activity* timestamp (frozen after close)
        billable_ref_ts = self._closed_last_activity_ts if getattr(self, "_closed_ts", None) else self._last_activity_ts
        self._billable_end_estimated_ts = (billable_ref_ts + SCALEDOWN_WINDOW_SECONDS) if billable_ref_ts else None

        self._meta_update(
            last_touched_ts=now,
            estimated_scaledown_ts=now + SCALEDOWN_WINDOW_SECONDS,
            last_activity_ts=self._last_activity_ts,
            billable_start_ts=self._billable_start_ts,
            closed_last_activity_ts=self._closed_last_activity_ts,
            billable_end_estimated_ts=self._billable_end_estimated_ts,
        )

    def _mrt_warmup(self) -> None:
        import tempfile
        import numpy as np
        import soundfile as sf

        from magenta_rt import audio as au
        from utils import take_bar_aligned_tail, make_bar_aligned_context

        codec_fps = float(self._mrt.codec.frame_rate)
        ctx_seconds = float(self._mrt.config.context_length_frames) / codec_fps
        sr = int(self._mrt.sample_rate)

        bpm = 120.0
        beats_per_bar = 4

        samples = int(max(1, round(ctx_seconds * sr)))
        silent = np.zeros((samples, 2), dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, silent, sr, subtype="PCM_16")
            tmp_path = tmp.name

        try:
            loop = au.Waveform.from_file(tmp_path).resample(sr).as_stereo()
            ctx_tail = take_bar_aligned_tail(loop, bpm, beats_per_bar, ctx_seconds)

            tokens_full = self._mrt.codec.encode(ctx_tail).astype(np.int32)
            tokens = tokens_full[:, : self._mrt.config.decoder_codec_rvq_depth]

            context_tokens = make_bar_aligned_context(
                tokens,
                bpm=bpm,
                fps=float(self._mrt.codec.frame_rate),
                ctx_frames=self._mrt.config.context_length_frames,
                beats_per_bar=beats_per_bar,
            )

            state = self._mrt.init_state()
            state.context_tokens = context_tokens
            style_vec = self._mrt.embed_style("warmup")

            _wav, _state = self._mrt.generate_chunk(state=state, style=style_vec)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    @modal.method()
    def status(self) -> Dict[str, Any]:
        """
        Cheap status call that ALSO keeps the container warm (counts as a request).
        Use this only when you *intend* to keep the GPU container alive.
        """
        self._touch()
        jam_running = False
        with self._worker_lock:
            if self._worker is not None and self._worker.is_alive():
                jam_running = True

        now = time.time()
        warm_completed_ts = getattr(self, "_warm_completed_ts", None)
        warm_started_ts = getattr(self, "_warm_started_ts", None)
        session_created_ts = getattr(self, "_session_created_ts", None)
        closed_ts = getattr(self, "_closed_ts", None)
        # Billing fields
        last_activity_ts = getattr(self, "_last_activity_ts", None)
        billable_start_ts = getattr(self, "_billable_start_ts", None)
        closed_last_activity_ts = getattr(self, "_closed_last_activity_ts", None)
        billable_ref_ts = closed_last_activity_ts if closed_ts else last_activity_ts
        billable_end_estimated_ts = (billable_ref_ts + SCALEDOWN_WINDOW_SECONDS) if billable_ref_ts else None

        billable_elapsed_seconds = None
        if billable_start_ts and billable_end_estimated_ts:
            billable_elapsed_seconds = max(0.0, min(now, billable_end_estimated_ts) - billable_start_ts)
        elif billable_start_ts:
            billable_elapsed_seconds = max(0.0, now - billable_start_ts)

        if not billable_start_ts:
            billing_state = "warming"
        elif closed_ts:
            billing_state = "closed"
        else:
            billing_state = "active"

        idle_seconds = (now - self._last_touched_ts) if self._last_touched_ts else None
        warm_age_seconds = (now - warm_completed_ts) if warm_completed_ts else None
        estimated_scaledown_ts = (self._last_touched_ts + SCALEDOWN_WINDOW_SECONDS) if self._last_touched_ts else None

        self._meta_update(
            warmed=bool(self._warmed),
            jam_running=bool(jam_running),
            container_warmup_seconds=self._warmup_seconds,
            warm_started_ts=warm_started_ts,
            warm_completed_ts=warm_completed_ts,
            session_created_ts=session_created_ts,
            closed_ts=closed_ts,
            closed=bool(closed_ts),
            estimated_scaledown_ts=estimated_scaledown_ts,
            last_activity_ts=last_activity_ts,
            billable_start_ts=billable_start_ts,
            closed_last_activity_ts=closed_last_activity_ts,
            billable_end_estimated_ts=billable_end_estimated_ts,
            billable_elapsed_seconds=billable_elapsed_seconds,
            billing_state=billing_state,
        )

        return {
            "ok": True,
            "session_id": self.session_id,
            "warmed": bool(self._warmed),
            "container_warmup_seconds": self._warmup_seconds,
            "jam_running": jam_running,
            "tag": os.getenv("MRT_TAG", "large"),
            "scaledown_window_seconds": SCALEDOWN_WINDOW_SECONDS,

            "session_created_ts": session_created_ts,
            "warm_started_ts": warm_started_ts,
            "warm_completed_ts": warm_completed_ts,
            "last_touched_ts": self._last_touched_ts,
            "closed_ts": closed_ts,

            "last_activity_ts": last_activity_ts,
            "billable_start_ts": billable_start_ts,
            "closed_last_activity_ts": closed_last_activity_ts,
            "billable_end_estimated_ts": billable_end_estimated_ts,
            "billable_elapsed_seconds": billable_elapsed_seconds,
            "billing_state": billing_state,

            "idle_seconds": idle_seconds,
            "warm_age_seconds": warm_age_seconds,
            "estimated_scaledown_ts": estimated_scaledown_ts,
        }

    @modal.method()
    def warmup(self) -> Dict[str, Any]:
        """
        Returns warmup metadata. Note: the heavy work happens in @enter() for a new session_id.
        """
        t0 = time.time()
        self._touch()
        return {
            "ok": True,
            "session_id": self.session_id,
            "warmed": bool(self._warmed),
            "container_warmup_seconds": self._warmup_seconds,
            # local handler time only (Modal scheduling/transport is measured by the web router)
            "handler_seconds": time.time() - t0,
            "tag": os.getenv("MRT_TAG", "large"),
            "cache_root": CACHE_ROOT,
            "hf_cache_dir": HF_CACHE_DIR,
            "mrt_cache_dir": MRT_CACHE_DIR,
            "jax_cache_dir": JAX_CACHE_DIR,
        }

    @staticmethod
    def _build_style_vector(
        mrt,
        *,
        styles_csv: str,
        weights_csv: str,
        loop_embed,
        loop_weight: float,
    ):
        """
        Minimal style mixing (loop tail + optional text styles).

        NOTE: We intentionally skip mean/centroids in v1 to avoid extra asset loading.
        """
        import numpy as np

        text_styles = _parse_csv_strings(styles_csv)
        text_weights = _parse_csv_floats(weights_csv)

        comps: List[np.ndarray] = []
        weights: List[float] = []

        if loop_embed is not None and float(loop_weight) > 0:
            comps.append(loop_embed.astype(np.float32, copy=False))
            weights.append(float(loop_weight))

        for i, s in enumerate(text_styles):
            w = 1.0
            if i < len(text_weights):
                w = float(text_weights[i])
            if w <= 0:
                continue
            e = mrt.embed_style(s)
            comps.append(e.astype(np.float32, copy=False))
            weights.append(float(w))

        if not comps:
            return mrt.embed_style("").astype(np.float32, copy=False)

        wsum = float(sum(weights))
        if wsum <= 0:
            return mrt.embed_style("").astype(np.float32, copy=False)

        weights = [w / wsum for w in weights]
        out = np.zeros_like(comps[0], dtype=np.float32)
        for w, e in zip(weights, comps):
            out += w * e
        return out

    @modal.method()
    def peek_status(self) -> Dict[str, Any]:
        """
        Status snapshot that does NOT call _touch().

        This is meant for metering / UI "is it warm?" checks when you do NOT want
        to extend the container's idle timer.
        """
        now = time.time()
        meta = self._meta_snapshot()

        last_touched_ts = meta.get("last_touched_ts", getattr(self, "_last_touched_ts", None))
        warm_completed_ts = meta.get("warm_completed_ts", getattr(self, "_warm_completed_ts", None))
        session_created_ts = meta.get("session_created_ts", getattr(self, "_session_created_ts", None))
        warm_started_ts = meta.get("warm_started_ts", getattr(self, "_warm_started_ts", None))
        closed_ts = meta.get("closed_ts", getattr(self, "_closed_ts", None))

        idle_seconds = (now - last_touched_ts) if last_touched_ts else None
        warm_age_seconds = (now - warm_completed_ts) if warm_completed_ts else None
        estimated_scaledown_ts = (last_touched_ts + meta.get("scaledown_window_seconds", SCALEDOWN_WINDOW_SECONDS)) if last_touched_ts else None

        return {
            "ok": True,
            "session_id": self.session_id,
            "warmed": bool(meta.get("warmed", getattr(self, "_warmed", False))),
            "jam_running": bool(meta.get("jam_running", False)),
            "tag": meta.get("tag", os.getenv("MRT_TAG", "large")),
            "scaledown_window_seconds": meta.get("scaledown_window_seconds", SCALEDOWN_WINDOW_SECONDS),

            "session_created_ts": session_created_ts,
            "warm_started_ts": warm_started_ts,
            "warm_completed_ts": warm_completed_ts,
            "last_touched_ts": last_touched_ts,
            "closed_ts": closed_ts,

            "idle_seconds": idle_seconds,
            "warm_age_seconds": warm_age_seconds,
            "estimated_scaledown_ts": estimated_scaledown_ts,
            "peek": True,
        }

    @modal.method()
    def jam_start(
        self,
        *,
        loop_wav_bytes: bytes,
        bpm: float,
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
        target_sample_rate: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Starts a JamWorker in this session container.
        """
        import tempfile
        import numpy as np
        import soundfile as sf

        from magenta_rt import audio as au
        from utils import take_bar_aligned_tail
        from jam_worker import JamParams, JamWorker

        self._touch()

        if not loop_wav_bytes:
            return {"ok": False, "error": "Empty file", "session_id": self.session_id}

        # prevent multiple workers per session
        with self._worker_lock:
            if self._worker is not None and self._worker.is_alive():
                return {"ok": False, "error": "Jam already running", "session_id": self.session_id}

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(loop_wav_bytes)
            tmp_path = tmp.name

        try:
            # load loop at model SR
            loop = au.Waveform.from_file(tmp_path).resample(self._mrt.sample_rate).as_stereo()

            # derive ctx tail
            codec_fps = float(self._mrt.codec.frame_rate)
            ctx_seconds = float(self._mrt.config.context_length_frames) / codec_fps
            loop_tail = take_bar_aligned_tail(loop, bpm, beats_per_bar, ctx_seconds)

            # style vec (tail-biased)
            loop_tail_embed = self._mrt.embed_style(loop_tail)
            style_vec = self._build_style_vector(
                self._mrt,
                styles_csv=styles,
                weights_csv=style_weights,
                loop_embed=loop_tail_embed,
                loop_weight=float(loop_weight),
            ).astype(np.float32, copy=False)

            # target SR defaults to input SR (same as HF app.py)
            inp_info = sf.info(tmp_path)
            input_sr = int(inp_info.samplerate)
            target_sr = int(target_sample_rate or input_sr)

            params = JamParams(
                bpm=float(bpm),
                beats_per_bar=int(beats_per_bar),
                bars_per_chunk=int(bars_per_chunk),
                target_sr=int(target_sr),
                loudness_mode=str(loudness_mode),
                headroom_db=float(loudness_headroom_db),
                style_vec=style_vec,
                ref_loop=loop_tail,
                combined_loop=loop,
                guidance_weight=float(guidance_weight),
                temperature=float(temperature),
                topk=int(topk),
            )

            # remember initial mix weights for future updates
            params._loop_weight = float(loop_weight)
            params._styles_csv = str(styles)
            params._style_weights_csv = str(style_weights)

            worker = JamWorker(self._mrt, params)
            worker.daemon = True

            with self._worker_lock:
                self._worker = worker
                self._worker.start()


            self._meta_update(jam_running=True, warmed=bool(self._warmed), warm_completed_ts=self._warm_completed_ts, container_warmup_seconds=self._warmup_seconds)
            return {"ok": True, "session_id": self.session_id, "status": "started"}
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    @modal.method()
    def jam_next(self, *, timeout_seconds: float = 25.0) -> Dict[str, Any]:
        """
        Returns the next available chunk (or null if none ready before timeout).
        """
        self._touch()

        with self._worker_lock:
            worker = self._worker

        if worker is None or (not worker.is_alive()):
            return {"ok": False, "error": "No active jam", "session_id": self.session_id}

        chunk = worker.get_next_chunk(timeout=float(timeout_seconds))
        if chunk is None:
            return {"ok": True, "session_id": self.session_id, "chunk": None}

        return {
            "ok": True,
            "session_id": self.session_id,
            "chunk": {
                "index": chunk.index,
                "audio_base64": chunk.audio_base64,
                "metadata": chunk.metadata,
            },
        }

    @modal.method()
    def jam_consume(self, *, chunk_index: int) -> Dict[str, Any]:
        """
        Acknowledge chunk consumption so the worker can drop it from its buffer.
        """
        self._touch()

        with self._worker_lock:
            worker = self._worker

        if worker is None or (not worker.is_alive()):
            return {"ok": False, "error": "No active jam", "session_id": self.session_id}

        worker.mark_chunk_consumed(int(chunk_index))
        return {"ok": True, "session_id": self.session_id}

    @modal.method()
    def jam_stop(self) -> Dict[str, Any]:
        """
        Stop worker and free per-session state.
        """
        self._touch()

        with self._worker_lock:
            worker = self._worker
            self._worker = None

        if worker is not None:
            try:
                worker.stop()
                worker.join(timeout=2.0)
            except Exception:
                pass

        self._meta_update(jam_running=False, warmed=bool(self._warmed), warm_completed_ts=self._warm_completed_ts, container_warmup_seconds=self._warmup_seconds)
        return {"ok": True, "session_id": self.session_id, "status": "stopped"}

    @modal.method()
    def session_close(self, *, reason: str = "") -> Dict[str, Any]:
        """
        Explicitly close a session (stop jam + mark closed).

        Note: any request still counts as activity (it will reset Modal's idle timer),
        but this gives your billing/middleware a clear "session ended here" signal.
        """
        # Do NOT call _touch() here; closing should be a semantic end-marker.
        with self._worker_lock:
            worker = self._worker
            self._worker = None

        if worker is not None:
            try:
                worker.stop()
                worker.join(timeout=2.0)
            except Exception:
                pass

        now = time.time()
        self._closed_ts = now

        # Freeze billable activity at close time so post-close pings don't extend billing.
        self._closed_last_activity_ts = getattr(self, "_last_activity_ts", None)
        if self._closed_last_activity_ts is None and getattr(self, "_billable_start_ts", None):
            self._closed_last_activity_ts = getattr(self, "_billable_start_ts", None)

        self._billable_end_estimated_ts = (self._closed_last_activity_ts + SCALEDOWN_WINDOW_SECONDS) if self._closed_last_activity_ts else None

        estimated_scaledown_ts = (self._last_touched_ts + SCALEDOWN_WINDOW_SECONDS) if self._last_touched_ts else None

        self._meta_update(
            closed=True,
            closed_ts=now,
            close_reason=str(reason or ""),
            jam_running=False,
            estimated_scaledown_ts=estimated_scaledown_ts,
            closed_last_activity_ts=self._closed_last_activity_ts,
            billable_end_estimated_ts=self._billable_end_estimated_ts,
            billing_state="closed",
        )

        return {
            "ok": True,
            "session_id": self.session_id,
            "status": "closed",
            "closed_ts": now,
            "close_reason": str(reason or ""),
            "last_touched_ts": self._last_touched_ts,
            "estimated_scaledown_ts": estimated_scaledown_ts,
            "closed_last_activity_ts": self._closed_last_activity_ts,
            "billable_end_estimated_ts": self._billable_end_estimated_ts,
            "billing_state": "closed",
        }

    @modal.method()
    def jam_update(
        self,
        *,
        styles: str = "",
        style_weights: str = "",
        loop_weight: Optional[float] = None,
        guidance_weight: Optional[float] = None,
        temperature: Optional[float] = None,
        topk: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Update jam knobs + style mix while running.
        Mirrors the HuggingFace implementation:
        - knobs are pushed into MRT immediately
        - style_vec is updated atomically (JamWorker ramps it if style_ramp_seconds > 0)
        """
        import numpy as np

        self._touch()

        with self._worker_lock:
            worker = self._worker

        if worker is None or (not worker.is_alive()):
            return {"ok": False, "error": "No active jam", "session_id": self.session_id}

        # 1) Update generation knobs (safe no-ops if None)
        try:
            worker.update_knobs(
                guidance_weight=guidance_weight,
                temperature=temperature,
                topk=topk,
            )
        except Exception as e:
            return {"ok": False, "error": f"update_knobs failed: {e}", "session_id": self.session_id}

        # 2) Rebuild style vector
        try:
            with worker._lock:
                # keep prior loop_weight if not provided
                prior_lw = getattr(worker.params, "_loop_weight", 1.0)
                lw = float(prior_lw if loop_weight is None else loop_weight)

                # Use the session's reference loop tail for loop_embed, if present
                ref_loop = getattr(worker.params, "ref_loop", None)

            loop_embed = None
            if ref_loop is not None:
                try:
                    loop_embed = self._mrt.embed_style(ref_loop)
                except Exception:
                    loop_embed = None

            style_vec = self._build_style_vector(
                self._mrt,
                styles_csv=str(styles),
                weights_csv=str(style_weights),
                loop_embed=loop_embed,
                loop_weight=float(lw),
            ).astype(np.float32, copy=False)

            # atomic swap for the generator thread
            with worker._lock:
                worker.params.style_vec = style_vec
                worker.params._loop_weight = float(lw)

            return {"ok": True, "session_id": self.session_id}
        except Exception as e:
            return {"ok": False, "error": f"style update failed: {e}", "session_id": self.session_id}

    @modal.method()
    def jam_reseed(
        self,
        *,
        loop_wav_bytes: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """
        Immediate reseed: replace the worker's context tokens from a waveform.
        Matches HuggingFace behavior:
        - If loop_wav_bytes provided, use that as the new combined bounce
        - Else, fall back to reseeding from the worker's internal target-SR spool tail
        """
        import numpy as np
        import tempfile
        from magenta_rt import audio as au
        from utils import take_bar_aligned_tail

        self._touch()

        with self._worker_lock:
            worker = self._worker

        if worker is None or (not worker.is_alive()):
            return {"ok": False, "error": "No active jam", "session_id": self.session_id}

        wav_model = None

        # Option 1: use uploaded new “combined” bounce from the app
        if loop_wav_bytes is not None:
            if not loop_wav_bytes:
                return {"ok": False, "error": "Empty file", "session_id": self.session_id}
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(loop_wav_bytes)
                path = tmp.name
            try:
                wav_model = au.Waveform.from_file(path).resample(self._mrt.sample_rate).as_stereo()
            finally:
                try:
                    os.unlink(path)
                except Exception:
                    pass
        else:
            # Option 2: reseed from what we've been streaming (target-SR spool tail)
            with worker._lock:
                spool = getattr(worker, "_spool", None)
                target_sr = int(getattr(worker.params, "target_sr", self._mrt.sample_rate))
                ctx_seconds = float(getattr(worker, "_ctx_seconds", 0.0))

            if spool is None or getattr(spool, "shape", (0,))[0] == 0:
                return {"ok": False, "error": "No internal audio available to reseed from", "session_id": self.session_id}

            tail_len = int(round(max(0.5, ctx_seconds) * target_sr))
            tail = spool[-tail_len:, :].astype(np.float32, copy=False) if spool.shape[0] >= tail_len else spool.astype(np.float32, copy=False)

            wav_target = au.Waveform(tail, int(target_sr)).as_stereo()
            wav_model = wav_target.resample(self._mrt.sample_rate).as_stereo()

        # Update reference loops for loudness + future style mixing, then reseed tokens
        with worker._lock:
            worker.params.combined_loop = wav_model
            try:
                loop_tail = take_bar_aligned_tail(
                    wav_model,
                    worker.params.bpm,
                    worker.params.beats_per_bar,
                    float(getattr(worker, "_ctx_seconds", 0.0)),
                )
                worker.params.ref_loop = loop_tail
            except Exception:
                pass

        try:
            worker.reseed_from_waveform(wav_model)
            return {"ok": True, "session_id": self.session_id}
        except Exception as e:
            return {"ok": False, "error": f"reseed_from_waveform failed: {e}", "session_id": self.session_id}

    @modal.method()
    def jam_reseed_splice(
        self,
        *,
        combined_wav_bytes: Optional[bytes] = None,
        anchor_bars: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Seamless reseed: token-splice reseed (preferred during live jams).
        Mirrors HuggingFace behavior:
        - If combined_wav_bytes provided, splice from that
        - Else, fall back to splicing from the worker's internal target-SR spool tail
        """
        import numpy as np
        import tempfile
        from magenta_rt import audio as au

        self._touch()

        with self._worker_lock:
            worker = self._worker

        if worker is None or (not worker.is_alive()):
            return {"ok": False, "error": "No active jam", "session_id": self.session_id}

        wav_model = None

        if combined_wav_bytes is not None:
            if not combined_wav_bytes:
                return {"ok": False, "error": "Empty file", "session_id": self.session_id}
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(combined_wav_bytes)
                path = tmp.name
            try:
                wav_model = au.Waveform.from_file(path).resample(self._mrt.sample_rate).as_stereo()
            finally:
                try:
                    os.unlink(path)
                except Exception:
                    pass
        else:
            with worker._lock:
                spool = getattr(worker, "_spool", None)
                target_sr = int(getattr(worker.params, "target_sr", self._mrt.sample_rate))
                ctx_seconds = float(getattr(worker, "_ctx_seconds", 0.0))

            if spool is None or getattr(spool, "shape", (0,))[0] == 0:
                return {"ok": False, "error": "No internal audio available to reseed from", "session_id": self.session_id}

            tail_len = int(round(max(0.5, ctx_seconds) * target_sr))
            tail = spool[-tail_len:, :].astype(np.float32, copy=False) if spool.shape[0] >= tail_len else spool.astype(np.float32, copy=False)
            wav_target = au.Waveform(tail, int(target_sr)).as_stereo()
            wav_model = wav_target.resample(self._mrt.sample_rate).as_stereo()

        try:
            worker.reseed_splice(wav_model, anchor_bars=float(anchor_bars))
            return {"ok": True, "session_id": self.session_id, "anchor_bars": float(anchor_bars)}
        except Exception as e:
            return {"ok": False, "error": f"reseed_splice failed: {e}", "session_id": self.session_id}

# ---------------------------------------------------------------------
# CPU web router
# ---------------------------------------------------------------------
@app.function(
    image=WEB_IMAGE,
    volumes={CACHE_ROOT: cache_vol},
    secrets=[api_secret],
)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, Form, File, UploadFile, HTTPException, Query, Request
    from fastapi.responses import JSONResponse
    from pathlib import Path
    from starlette.concurrency import run_in_threadpool

    api = FastAPI()

    # --- API Key Authentication Middleware ---
    API_KEY = os.environ.get("DARIUS_API_KEY")

    @api.middleware("http")
    async def verify_api_key(request: Request, call_next):
        # Allow health checks without auth
        if request.url.path == "/health":
            return await call_next(request)
        
        # Check for API key header
        provided_key = request.headers.get("X-API-Key")
        if not API_KEY or provided_key != API_KEY:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid or missing API key"}
            )
        return await call_next(request)

    @api.get("/health")
    async def health():
        return {"ok": True, "service": "modal-darius"}

    @api.post("/warmup")
    async def warmup(session_id: Optional[str] = Form(None)):
        sid = session_id or str(uuid.uuid4())

        t0 = time.time()
        res = await run_in_threadpool(lambda: DariusSession(session_id=sid).warmup.remote())
        total = time.time() - t0

        # Make request_seconds meaningful (end-to-end server time, includes Modal hop)
        res.pop("request_seconds", None)
        res["request_seconds"] = total
        return res

    @api.post("/status")
    async def status(session_id: str = Form(...), keepalive: int = Form(1)):
        """
        keepalive=1 (default): calls GPU status() and keeps container warm.
        keepalive=0: returns shared meta snapshot without touching the GPU container.
        """
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")

        t0 = time.time()
        if int(keepalive) == 0:
            meta = await run_in_threadpool(lambda: session_meta.get(session_id, {}) or {})
            if not isinstance(meta, dict):
                meta = {}
            now = time.time()
            last_touched_ts = meta.get("last_touched_ts")
            warm_completed_ts = meta.get("warm_completed_ts")
            idle_seconds = (now - last_touched_ts) if last_touched_ts else None
            warm_age_seconds = (now - warm_completed_ts) if warm_completed_ts else None
            estimated_scaledown_ts = (last_touched_ts + meta.get("scaledown_window_seconds", SCALEDOWN_WINDOW_SECONDS)) if last_touched_ts else None

            last_activity_ts = meta.get("closed_last_activity_ts") if meta.get("closed_ts") else meta.get("last_activity_ts")
            billable_start_ts = meta.get("billable_start_ts")
            closed_last_activity_ts = meta.get("closed_last_activity_ts")
            billable_end_estimated_ts = (last_activity_ts + meta.get("scaledown_window_seconds", SCALEDOWN_WINDOW_SECONDS)) if last_activity_ts else meta.get("billable_end_estimated_ts")

            billable_elapsed_seconds = None
            if billable_start_ts and billable_end_estimated_ts:
                billable_elapsed_seconds = max(0.0, min(now, billable_end_estimated_ts) - billable_start_ts)
            elif billable_start_ts:
                billable_elapsed_seconds = max(0.0, now - billable_start_ts)

            if not billable_start_ts:
                billing_state = "warming"
            elif meta.get("closed_ts"):
                billing_state = "closed"
            else:
                billing_state = "active"
            res = {
                "ok": True,
                "session_id": session_id,
                "warmed": bool(meta.get("warmed", False)),
                "jam_running": bool(meta.get("jam_running", False)),
                "tag": meta.get("tag", os.getenv("MRT_TAG", "large")),
                "scaledown_window_seconds": meta.get("scaledown_window_seconds", SCALEDOWN_WINDOW_SECONDS),
                "session_created_ts": meta.get("session_created_ts"),
                "warm_started_ts": meta.get("warm_started_ts"),
                "warm_completed_ts": meta.get("warm_completed_ts"),
                "last_touched_ts": last_touched_ts,
                "closed_ts": meta.get("closed_ts"),


                "last_activity_ts": meta.get("last_activity_ts"),
                "billable_start_ts": meta.get("billable_start_ts"),
                "closed_last_activity_ts": meta.get("closed_last_activity_ts"),
                "billable_end_estimated_ts": billable_end_estimated_ts,
                "billable_elapsed_seconds": billable_elapsed_seconds,
                "billing_state": billing_state,

                "idle_seconds": idle_seconds,
                "warm_age_seconds": warm_age_seconds,
                "estimated_scaledown_ts": estimated_scaledown_ts,
                "peek": True,
            }
        else:
            res = await run_in_threadpool(lambda: DariusSession(session_id=session_id).status.remote())

        res["request_seconds"] = time.time() - t0
        return res


    @api.post("/status/peek")
    async def status_peek(session_id: str = Form(...)):
        # Alias for /status keepalive=0 (no GPU touch)
        return await status(session_id=session_id, keepalive=0)

    @api.get("/status/peek")
    async def status_peek_get(session_id: str = Query(...)):
        # Convenience GET for curl/browser usage
        t0 = time.time()
        meta = await run_in_threadpool(lambda: session_meta.get(session_id, {}) or {})
        if not isinstance(meta, dict):
            meta = {}
        now = time.time()
        last_touched_ts = meta.get("last_touched_ts")
        warm_completed_ts = meta.get("warm_completed_ts")
        idle_seconds = (now - last_touched_ts) if last_touched_ts else None
        warm_age_seconds = (now - warm_completed_ts) if warm_completed_ts else None
        estimated_scaledown_ts = meta.get("estimated_scaledown_ts")
        if estimated_scaledown_ts is None and last_touched_ts:
            estimated_scaledown_ts = last_touched_ts + meta.get("scaledown_window_seconds", SCALEDOWN_WINDOW_SECONDS)


        last_activity_ts = meta.get("closed_last_activity_ts") if meta.get("closed_ts") else meta.get("last_activity_ts")
        billable_start_ts = meta.get("billable_start_ts")
        closed_last_activity_ts = meta.get("closed_last_activity_ts")
        billable_end_estimated_ts = (last_activity_ts + meta.get("scaledown_window_seconds", SCALEDOWN_WINDOW_SECONDS)) if last_activity_ts else meta.get("billable_end_estimated_ts")

        billable_elapsed_seconds = None
        if billable_start_ts and billable_end_estimated_ts:
            billable_elapsed_seconds = max(0.0, min(now, billable_end_estimated_ts) - billable_start_ts)
        elif billable_start_ts:
            billable_elapsed_seconds = max(0.0, now - billable_start_ts)

        if not billable_start_ts:
            billing_state = "warming"
        elif meta.get("closed_ts"):
            billing_state = "closed"
        else:
            billing_state = "active"

        res = {
            "ok": True,
            "session_id": session_id,
            "warmed": bool(meta.get("warmed", False)),
            "jam_running": bool(meta.get("jam_running", False)),
            "tag": meta.get("tag"),
            "scaledown_window_seconds": meta.get("scaledown_window_seconds", SCALEDOWN_WINDOW_SECONDS),
            "session_created_ts": meta.get("session_created_ts"),
            "warm_started_ts": meta.get("warm_started_ts"),
            "warm_completed_ts": meta.get("warm_completed_ts"),
            "last_touched_ts": last_touched_ts,
            "closed_ts": meta.get("closed_ts"),


            "last_activity_ts": meta.get("last_activity_ts"),
            "billable_start_ts": meta.get("billable_start_ts"),
            "closed_last_activity_ts": meta.get("closed_last_activity_ts"),
            "billable_end_estimated_ts": billable_end_estimated_ts,
            "billable_elapsed_seconds": billable_elapsed_seconds,
            "billing_state": billing_state,

            "idle_seconds": idle_seconds,
            "warm_age_seconds": warm_age_seconds,
            "estimated_scaledown_ts": estimated_scaledown_ts,
            "peek": True,
            "request_seconds": time.time() - t0,
        }
        return res

    @api.post("/session/close")
    async def session_close(session_id: str = Form(...), reason: str = Form("manual_close")):
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        t0 = time.time()
        res = await run_in_threadpool(
            lambda: DariusSession(session_id=session_id).session_close.remote(reason=reason)
        )
        res["request_seconds"] = time.time() - t0
        return res

    @api.post("/jam/start")
    async def jam_start(
        session_id: str = Form(...),
        loop_audio: UploadFile = File(...),
        bpm: float = Form(...),
        bars_per_chunk: int = Form(4),
        beats_per_bar: int = Form(4),
        styles: str = Form(""),
        style_weights: str = Form(""),
        loop_weight: float = Form(1.0),
        loudness_mode: str = Form("auto"),
        loudness_headroom_db: float = Form(1.0),
        guidance_weight: float = Form(1.1),
        temperature: float = Form(1.1),
        topk: int = Form(40),
        target_sample_rate: Optional[int] = Form(None),
    ):
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        loop_bytes = await loop_audio.read()
        t0 = time.time()
        res = await run_in_threadpool(
            lambda: DariusSession(session_id=session_id).jam_start.remote(
                loop_wav_bytes=loop_bytes,
                bpm=float(bpm),
                bars_per_chunk=int(bars_per_chunk),
                beats_per_bar=int(beats_per_bar),
                styles=str(styles),
                style_weights=str(style_weights),
                loop_weight=float(loop_weight),
                loudness_mode=str(loudness_mode),
                loudness_headroom_db=float(loudness_headroom_db),
                guidance_weight=float(guidance_weight),
                temperature=float(temperature),
                topk=int(topk),
                target_sample_rate=(int(target_sample_rate) if target_sample_rate is not None else None),
            )
        )
        res["request_seconds"] = time.time() - t0
        return res

    @api.get("/jam/next")
    async def jam_next(
        session_id: str = Query(...),
        timeout_seconds: float = Query(25.0),
    ):
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")

        # Avoid long-polling past typical HTTP ingress limits.
        ts = min(float(timeout_seconds), 8.0)

        t0 = time.time()
        res = await run_in_threadpool(
            lambda: DariusSession(session_id=session_id).jam_next.remote(timeout_seconds=ts)
        )
        res["request_seconds"] = time.time() - t0
        return res

    @api.post("/jam/consume")
    async def jam_consume(
        session_id: str = Form(...),
        chunk_index: int = Form(...),
    ):
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        t0 = time.time()
        res = await run_in_threadpool(
            lambda: DariusSession(session_id=session_id).jam_consume.remote(chunk_index=int(chunk_index))
        )
        res["request_seconds"] = time.time() - t0
        return res

    @api.post("/jam/stop")
    async def jam_stop(session_id: str = Form(...)):
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        t0 = time.time()
        res = await run_in_threadpool(lambda: DariusSession(session_id=session_id).jam_stop.remote())
        res["request_seconds"] = time.time() - t0
        return res

    @api.post("/jam/update")
    async def jam_update(
        session_id: str = Form(...),
        styles: str = Form(""),
        style_weights: str = Form(""),
        loop_weight: Optional[float] = Form(None),
        guidance_weight: Optional[float] = Form(None),
        temperature: Optional[float] = Form(None),
        topk: Optional[int] = Form(None),
    ):
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        t0 = time.time()
        res = await run_in_threadpool(
            lambda: DariusSession(session_id=session_id).jam_update.remote(
                styles=str(styles),
                style_weights=str(style_weights),
                loop_weight=(float(loop_weight) if loop_weight is not None else None),
                guidance_weight=(float(guidance_weight) if guidance_weight is not None else None),
                temperature=(float(temperature) if temperature is not None else None),
                topk=(int(topk) if topk is not None else None),
            )
        )
        res["request_seconds"] = time.time() - t0
        return res

    @api.post("/jam/reseed")
    async def jam_reseed(
        session_id: str = Form(...),
        loop_audio: Optional[UploadFile] = File(None),
    ):
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        loop_bytes = await loop_audio.read() if loop_audio is not None else None
        t0 = time.time()
        res = await run_in_threadpool(
            lambda: DariusSession(session_id=session_id).jam_reseed.remote(loop_wav_bytes=loop_bytes)
        )
        res["request_seconds"] = time.time() - t0
        return res

    @api.post("/jam/reseed_splice")
    async def jam_reseed_splice(
        session_id: str = Form(...),
        anchor_bars: float = Form(2.0),
        combined_audio: Optional[UploadFile] = File(None),
    ):
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        combined_bytes = await combined_audio.read() if combined_audio is not None else None
        t0 = time.time()
        res = await run_in_threadpool(
            lambda: DariusSession(session_id=session_id).jam_reseed_splice.remote(
                combined_wav_bytes=combined_bytes,
                anchor_bars=float(anchor_bars),
            )
        )
        res["request_seconds"] = time.time() - t0
        return res

    @api.get("/debug/cache")

    async def debug_cache():
        def summarize_dir(p: str):
            path = Path(p)
            if not path.exists():
                return {"path": p, "exists": False}
            files = 0
            total = 0
            for fp in path.rglob("*"):
                if fp.is_file():
                    files += 1
                    try:
                        total += fp.stat().st_size
                    except Exception:
                        pass
            return {"path": p, "exists": True, "files": files, "bytes": total}

        return {
            "cache_root": summarize_dir(CACHE_ROOT),
            "hf": summarize_dir(HF_CACHE_DIR),
            "mrt": summarize_dir(MRT_CACHE_DIR),
            "jax": summarize_dir(JAX_CACHE_DIR),
        }

    return api
