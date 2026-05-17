"""
voice-app · FastRTC Stream-Server.

Mountet einen WebRTC-Audio-Endpunkt auf /
und einen JSON-Metrics-Endpunkt auf /metrics.

Pipeline-Modus (sequential vs streaming) wird per Env-Var PIPELINE_MODE
gesteuert (siehe pipeline.py).
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Iterator

import gradio as gr
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastrtc.webrtc import WebRTC
from fastrtc import (
    AlgoOptions,
    ReplyOnPause,
    Stream,
)

load_dotenv()

from metrics import TurnTrace, recent_turns  # noqa: E402
from pipeline import (  # noqa: E402
    reset_conversation_history,
    run_sequential,
    run_streaming,
    transcribe_audio,
)


MODE = os.environ.get("PIPELINE_MODE", "streaming").strip().lower()
assert MODE in {"sequential", "streaming"}, f"Bad PIPELINE_MODE: {MODE!r}"
print(f"[app] PIPELINE_MODE = {MODE}", flush=True)

VAD_PAUSE = float(os.environ.get("VAD_PAUSE_DURATION", "0.6"))
VAD_STARTED = float(os.environ.get("VAD_STARTED_TALKING", "0.2"))
VAD_SPEECH = float(os.environ.get("VAD_SPEECH_THRESHOLD", "0.1"))
RTC_STUN_URLS = [
    url.strip()
    for url in os.environ.get("RTC_STUN_URLS", "stun:stun.l.google.com:19302").split(",")
    if url.strip()
]
RTC_CONFIGURATION = (
    {"iceServers": [{"urls": RTC_STUN_URLS}]}
    if RTC_STUN_URLS
    else {}
)


_original_webrtc_handle_offer = WebRTC.handle_offer


async def _handle_offer_with_pending_ice(self, body, set_outputs):
    """Wait briefly for the SDP offer if trickle ICE arrives first."""
    webrtc_id = body.get("webrtc_id")
    if body.get("type") == "ice-candidate" and "candidate" in body:
        if webrtc_id not in self.pcs:
            for _ in range(40):
                await asyncio.sleep(0.05)
                if webrtc_id in self.pcs:
                    break
            else:
                print(
                    f"[webrtc] ignoring early ICE candidate for {webrtc_id}",
                    flush=True,
                )
                return JSONResponse({"status": "success"})
    elif webrtc_id and webrtc_id not in self.pcs:
        reset_conversation_history()
    return await _original_webrtc_handle_offer(self, body, set_outputs)


WebRTC.handle_offer = _handle_offer_with_pending_ice


# -------------------------------------------------------------- handler


def handle_turn(audio: tuple[int, np.ndarray]) -> Iterator:
    """Wird von FastRTC nach jedem End-of-turn aufgerufen.

    Parameter
    ---------
    audio : (sample_rate, samples) – samples ist int16 numpy
    """
    trace = TurnTrace(mode=MODE)

    transcript = transcribe_audio(audio).strip()
    trace.stamp("t_stt_done")
    trace.transcript = transcript

    if not transcript:
        print("[handle_turn] leeres Transcript, skip", flush=True)
        return

    print(f"[handle_turn] USER: {transcript!r}", flush=True)

    try:
        if MODE == "sequential":
            yield from run_sequential(transcript, trace)
        else:
            yield from run_streaming(transcript, trace)
    finally:
        trace.persist()


# -------------------------------------------------------------- FastRTC stream

stream = Stream(
    handler=ReplyOnPause(
        handle_turn,
        can_interrupt=True,
        algo_options=AlgoOptions(
            audio_chunk_duration=VAD_PAUSE,
            started_talking_threshold=VAD_STARTED,
            speech_threshold=VAD_SPEECH,
        ),
    ),
    modality="audio",
    mode="send-receive",
    rtc_configuration=RTC_CONFIGURATION,
)


# -------------------------------------------------------------- FastAPI app

app = FastAPI(title="Speech Processing Lab · voice-app")

# Mount the FastRTC routes (WebRTC offer/answer)
stream.mount(app)


@app.get("/metrics")
def metrics(limit: int = 50) -> JSONResponse:
    return JSONResponse(recent_turns(limit=min(max(limit, 1), 500)))


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "mode": MODE, "ts": time.time()}


@app.get("/manifest.json")
def manifest() -> JSONResponse:
    return JSONResponse(
        {
            "name": "Speech Processing Lab",
            "short_name": "Voice Lab",
            "start_url": "/",
            "display": "standalone",
        }
    )


app = gr.mount_gradio_app(app, stream.ui, path="/")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
