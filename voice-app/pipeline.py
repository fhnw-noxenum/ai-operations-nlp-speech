"""
Zwei Pipeline-Modi zum direkten Vergleich:

  sequential
      STT → LLM (vollständige Antwort) → TTS (vollständiger Audio-Output)
      Einfach, langsam. TTFA = STT + LLM + TTS.

  streaming
      STT → LLM (Token-Stream) → Sentence-Splitter → TTS (chunked)
      Pro-Satz-TTS. TTFA = STT + LLM-TTFT + TTS-TTFB ≈ 70 % schneller.

Beide Modi instrumentieren TurnTrace; das Dashboard zeigt die Differenz.
"""

from __future__ import annotations

import os
from typing import Iterator

import numpy as np
from elevenlabs.client import ElevenLabs
from fastrtc import audio_to_bytes
from groq import Groq

from metrics import TurnTrace
from sentence_split import SentenceSplitter

GROQ = Groq()
ELEVEN = ElevenLabs()

LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-20b")
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "200"))
LLM_REASONING = os.environ.get("LLM_REASONING_EFFORT", "low")
LLM_TEMP = float(os.environ.get("LLM_TEMPERATURE", "0.7"))
LLM_SYSTEM = os.environ.get(
    "LLM_SYSTEM_PROMPT",
    "Du bist ein hilfsbereiter Voice-Assistent. Antworte immer auf Deutsch, auch wenn die Eingabe in einer anderen Sprache ist. Antworte in 1-2 kurzen Sätzen.",
)

GROQ_STT_MODEL = os.environ.get("GROQ_STT_MODEL", "whisper-large-v3-turbo")

TTS_MODEL = os.environ.get("TTS_MODEL", "eleven_flash_v2_5")
TTS_VOICE = os.environ.get("TTS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
TTS_FORMAT = os.environ.get("TTS_OUTPUT_FORMAT", "pcm_24000")
TTS_LANGUAGE = os.environ.get("TTS_LANGUAGE_CODE", "de").strip() or None

# ElevenLabs PCM 24000 → 24 kHz mono int16
TTS_SAMPLE_RATE = 24000


# ---------------------------------------------------------------- helpers


def _llm_kwargs() -> dict:
    return dict(
        model=LLM_MODEL,
        max_completion_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMP,
        reasoning_effort=LLM_REASONING,
        # gpt-oss family supports include_reasoning; harmless on others
        # (Groq erlaubt unbekannte Felder via extra_body; wir lassen es weg)
    )


def _tts_stream_to_audio(text: str) -> Iterator[np.ndarray]:
    """ElevenLabs TTS-Stream → numpy int16-Chunks mit (rate, samples)-Shape."""
    emitted_samples = 0
    stream = ELEVEN.text_to_speech.convert_as_stream(
        voice_id=TTS_VOICE,
        text=text,
        model_id=TTS_MODEL,
        output_format=TTS_FORMAT,
        language_code=TTS_LANGUAGE,
    )
    for chunk in stream:
        if not chunk:
            continue
        # chunk ist raw 16-bit little-endian PCM
        samples = np.frombuffer(chunk, dtype=np.int16)
        if samples.size == 0:
            continue
        emitted_samples += samples.size
        yield samples.reshape(1, -1)
    if emitted_samples == 0:
        print(f"[tts] no audio returned for text: {text!r}", flush=True)


def _tts_full_audio(text: str) -> np.ndarray:
    """Collect the full TTS result before returning audio."""
    chunks = list(_tts_stream_to_audio(text))
    if not chunks:
        return np.empty((1, 0), dtype=np.int16)
    return np.concatenate(chunks, axis=1)


def transcribe_audio(audio: tuple[int, np.ndarray]) -> str:
    """Transcribe a FastRTC audio turn using Groq Whisper."""
    transcription = GROQ.audio.transcriptions.create(
        file=("turn.wav", audio_to_bytes(audio)),
        model=GROQ_STT_MODEL,
        response_format="json",
        language="de",
    )
    return transcription.text or ""


# ---------------------------------------------------------------- pipelines


def run_sequential(transcript: str, trace: TurnTrace) -> Iterator:
    """STT done → vollständige LLM-Antwort sammeln → komplette TTS."""
    completion = GROQ.chat.completions.create(
        messages=[
            {"role": "system", "content": LLM_SYSTEM},
            {"role": "user", "content": transcript},
        ],
        stream=False,
        **_llm_kwargs(),
    )
    trace.stamp("t_llm_first")  # in sequential ≈ t_llm_done
    answer = completion.choices[0].message.content or ""
    trace.stamp("t_llm_done")
    trace.response = answer
    print(f"[seq] LLM: {answer!r}", flush=True)

    samples = _tts_full_audio(answer)
    trace.stamp("t_tts_first")
    if samples.size:
        yield (TTS_SAMPLE_RATE, samples)


def run_streaming(transcript: str, trace: TurnTrace) -> Iterator:
    """LLM-Token-Stream → Sentence-Splitter → sofort TTS pro Satz."""
    stream = GROQ.chat.completions.create(
        messages=[
            {"role": "system", "content": LLM_SYSTEM},
            {"role": "user", "content": transcript},
        ],
        stream=True,
        **_llm_kwargs(),
    )

    splitter = SentenceSplitter()
    full_response: list[str] = []
    first_tok_stamped = False
    first_audio_stamped = False

    for chunk in stream:
        try:
            delta = chunk.choices[0].delta.content or ""
        except (AttributeError, IndexError):
            continue
        if not delta:
            continue
        if not first_tok_stamped:
            trace.stamp("t_llm_first")
            first_tok_stamped = True
        full_response.append(delta)

        for sentence in splitter.feed(delta):
            print(f"[stream] sentence → TTS: {sentence!r}", flush=True)
            for samples in _tts_stream_to_audio(sentence):
                if not first_audio_stamped:
                    trace.stamp("t_tts_first")
                    first_audio_stamped = True
                yield (TTS_SAMPLE_RATE, samples)

    trace.stamp("t_llm_done")
    rest = splitter.flush()
    if rest:
        print(f"[stream] flush → TTS: {rest!r}", flush=True)
        for samples in _tts_stream_to_audio(rest):
            if not first_audio_stamped:
                trace.stamp("t_tts_first")
                first_audio_stamped = True
            yield (TTS_SAMPLE_RATE, samples)

    trace.response = "".join(full_response)
