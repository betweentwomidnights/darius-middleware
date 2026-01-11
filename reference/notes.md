(.venv) kev@MSI:~/modal-darius$ curl -sS -X POST "$BASE/warmup" | jq .
{
  "ok": true,
  "session_id": "0f80f01b-6360-43c5-a01e-d9b4bfc8dd2e",
  "warmed": true,
  "container_warmup_seconds": 49.80913233757019,
  "handler_seconds": 0.20324063301086426,
  "tag": "large",
  "cache_root": "/cache",
  "hf_cache_dir": "/cache/huggingface",
  "mrt_cache_dir": "/cache/magenta_rt",
  "jax_cache_dir": "/cache/jax",
  "request_seconds": 66.17970824241638
}
(.venv) kev@MSI:~/modal-darius$

(.venv) kev@MSI:~/modal-darius$ SID="0f80f01b-6360-43c5-a01e-d9b4bfc8dd2e"

(.venv) kev@MSI:~/modal-darius$ curl -sS -X POST "$BASE/status"   -F "session_id=$SID"   -F "keepalive=0" | jq .
{
  "ok": true,
  "session_id": "0f80f01b-6360-43c5-a01e-d9b4bfc8dd2e",
  "warmed": true,
  "jam_running": false,
  "tag": "large",
  "scaledown_window_seconds": 600,
  "session_created_ts": 1768030296.1161666,
  "warm_started_ts": 1768030296.1161673,
  "warm_completed_ts": 1768030347.273232,
  "last_touched_ts": 1768030347.8358073,
  "closed_ts": null,
  "last_activity_ts": 1768030347.8358073,
  "billable_start_ts": 1768030347.273232,
  "closed_last_activity_ts": null,
  "billable_end_estimated_ts": 1768030947.8358073,
  "billable_elapsed_seconds": 101.51608729362488,
  "billing_state": "active",
  "idle_seconds": 100.95351195335388,
  "warm_age_seconds": 101.51608729362488,
  "estimated_scaledown_ts": 1768030947.8358073,
  "peek": true,
  "request_seconds": 0.14223384857177734
}
(.venv) kev@MSI:~/modal-darius$

(.venv) kev@MSI:~/modal-darius$ curl -sS -X POST "$BASE/jam/start"   -F "session_id=$SID"   -F "bpm=120"   -F "bars_per_chunk=4"   -F "beats_per_bar=4"   -F "styles="   -F "style_weights="   -F "loop_weight=1.0"   -F "loudness_mode=auto"   -F "loudness_headroom_db=1.0"   -F "guidance_weight=1.1"   -F "temperature=1.1"   -F "topk=40"   -F "target_sample_rate="   -F "loop_audio=@loop.wav" | jq .
{
  "ok": true,
  "session_id": "0f80f01b-6360-43c5-a01e-d9b4bfc8dd2e",
  "status": "started",
  "request_seconds": 4.122900485992432
}
(.venv) kev@MSI:~/modal-darius$

(.venv) kev@MSI:~/modal-darius$ curl -sS "$BASE/jam/next?session_id=$SID&timeout_seconds=5" | jq .

...base64...

    "metadata": {
      "bpm": 120,
      "bars": 4,
      "beats_per_bar": 4,
      "sample_rate": 48000,
      "channels": 2,
      "total_samples": 384000,
      "seconds_per_bar": 2,
      "loop_duration_seconds": 8,
      "guidance_weight": 1.1,
      "temperature": 1.1,
      "topk": 40
    }
  },
  "request_seconds": 2.1516172885894775
}
(.venv) kev@MSI:~/modal-darius$

(.venv) kev@MSI:~/modal-darius$ curl -sS -X POST "$BASE/jam/consume"   -F "session_id=$SID"   -F "chunk_index=0" | jq .
{
  "ok": true,
  "session_id": "0f80f01b-6360-43c5-a01e-d9b4bfc8dd2e",
  "request_seconds": 0.5411746501922607
}
(.venv) kev@MSI:~/modal-darius$

(.venv) kev@MSI:~/modal-darius$ curl -sS -X POST "$BASE/status"   -F "session_id=$SID"   -F "keepalive=0" | jq .
{
  "ok": true,
  "session_id": "0f80f01b-6360-43c5-a01e-d9b4bfc8dd2e",
  "warmed": true,
  "jam_running": true,
  "tag": "large",
  "scaledown_window_seconds": 600,
  "session_created_ts": 1768030296.1161666,
  "warm_started_ts": 1768030296.1161673,
  "warm_completed_ts": 1768030347.273232,
  "last_touched_ts": 1768030551.2413917,
  "closed_ts": null,
  "last_activity_ts": 1768030551.2413917,
  "billable_start_ts": 1768030347.273232,
  "closed_last_activity_ts": null,
  "billable_end_estimated_ts": 1768031151.2413917,
  "billable_elapsed_seconds": 230.9049677848816,
  "billing_state": "active",
  "idle_seconds": 26.936808109283447,
  "warm_age_seconds": 230.9049677848816,
  "estimated_scaledown_ts": 1768031151.2413917,
  "peek": true,
  "request_seconds": 0.06903195381164551
}
(.venv) kev@MSI:~/modal-darius$

(.venv) kev@MSI:~/modal-darius$ curl -sS -X POST "$BASE/jam/stop"   -F "session_id=$SID" | jq .
{
  "ok": true,
  "session_id": "0f80f01b-6360-43c5-a01e-d9b4bfc8dd2e",
  "status": "stopped",
  "request_seconds": 0.7829201221466064
}
(.venv) kev@MSI:~/modal-darius$

(.venv) kev@MSI:~/modal-darius$ curl -sS -X POST "$BASE/session/close"   -F "session_id=$SID"   -F "reason=manual_close" | jq .
{
  "ok": true,
  "session_id": "0f80f01b-6360-43c5-a01e-d9b4bfc8dd2e",
  "status": "closed",
  "closed_ts": 1768030639.2451477,
  "close_reason": "manual_close",
  "last_touched_ts": 1768030613.6963997,
  "estimated_scaledown_ts": 1768031213.6963997,
  "closed_last_activity_ts": 1768030613.6963997,
  "billable_end_estimated_ts": 1768031213.6963997,
  "billing_state": "closed",
  "request_seconds": 0.5989813804626465
}
(.venv) kev@MSI:~/modal-darius$