# Speech Processing Lab · Voice-Pipeline mit Latenz-Messung

Ein vollständig containerisiertes Lab für das Modul **Speech Processing** im
CAS AI Operations (NLP Operations). Sie experimentieren mit den Bausteinen
einer Voice-Pipeline (STT, LLM, TTS) und messen, wie sich Streaming auf die
**Time-to-First-Audio (TTFA)** auswirkt.

## Was läuft

```
Browser  ──WebRTC──▶  voice-app (FastRTC)  ──▶  Groq Cloud (gpt-oss-20b)
                            │                ──▶  ElevenLabs API
                            │
                            ▼
                       SQLite metrics
                            ▲
                            │
                       metrics-ui (Streamlit, Port 8501)
```

* **voice-app** (Port 8000) – FastRTC + Groq Whisper STT + Groq LLM + ElevenLabs TTS
* **metrics-ui** (Port 8501) – Streamlit-Dashboard über die SQLite-Metriken
* Alles läuft lokal über `docker compose`; STT, LLM und TTS gehen in die Cloud
  (siehe `.env.example`).

## Voraussetzungen

* Docker und Docker Compose
* Ein **Groq API Key** (`https://console.groq.com`) – kostenlos für Tests
* Ein **ElevenLabs API Key** (`https://elevenlabs.io`) – Free Tier reicht
* Mikrofon im Browser

## Schritt 1 · Setup

```bash
git clone <dieses-repo>
cd lab
cp .env.example .env
# .env öffnen und beide API-Keys eintragen
docker compose up --build
```

Endpunkte nach dem Start:

| URL                                | Was                              |
|------------------------------------|----------------------------------|
| `http://localhost:8000`            | Voice-UI (Gradio-Wrapper)        |
| `http://localhost:8000/metrics`    | Letzte Turns als JSON            |
| `http://localhost:8501`            | Latenz-Dashboard (Streamlit)     |

Im Browser Mikrofon erlauben und einen Satz sprechen.

## Schritt 2 · Baseline (sequentiell)

Setzen Sie in `.env`:

```
PIPELINE_MODE=sequential
```

Container neu starten:

```bash
docker compose up -d --force-recreate voice-app
```

In diesem Modus wartet die TTS auf die **vollständige** LLM-Antwort,
bevor sie zu sprechen beginnt. Notieren Sie die typischen TTFA-Werte
aus dem Dashboard. Erwartung: 1.5 – 3 s, je nach Antwortlänge.

## Schritt 3 · Streaming

Setzen Sie in `.env`:

```
PIPELINE_MODE=streaming
```

`docker compose up -d --force-recreate voice-app` und nochmal sprechen.
Jetzt streamt der LLM Token-für-Token, der Sentence-Splitter füttert
abgeschlossene Sätze sofort an ElevenLabs. Erwartung: TTFA ~ 400 – 800 ms.

## Schritt 4 · Tuning

Variieren Sie folgende Variablen in `.env` und beobachten Sie die Wirkung:

| Variable                       | Default | Wirkung                                          |
|--------------------------------|---------|--------------------------------------------------|
| `VAD_PAUSE_DURATION`           | 0.6     | Kürzer = schnellerer Turn, aber Schnitt-Gefahr  |
| `LLM_MAX_TOKENS`               | 200     | Kürzer = schnelleres LLM-Ende                    |
| `LLM_REASONING_EFFORT`         | low     | low spart hidden CoT bei gpt-oss                 |
| `TTS_MODEL`                    | eleven_flash_v2_5 | flash = ~75 ms TTFB, turbo = ~250 ms   |
| `TTS_LANGUAGE_CODE`            | de      | ElevenLabs explizit auf Deutsch setzen           |
| `TTS_VOICE_ID`                 | C4QktxZ39Uccr2xNdFHg | Deutsche Frauenstimme in ElevenLabs |

Tipp: Eine Variable nach der anderen ändern, jeweils 5 – 10 Turns sammeln.

## Schritt 5 · Auswertung

Im Dashboard sehen Sie:

* **Latenz pro Stufe** (STT, LLM-TTFT, TTS-TTFB) als Bar-Chart
* **P50 / P95** über die letzten 50 Turns
* **Tabelle der letzten Turns** mit Transcript und Response
* Mode-Vergleich `sequential` vs `streaming` nebeneinander

Identifizieren Sie den Bottleneck. Ist es:
* STT (bei Groq Whisper)?
* LLM (bei wortreichen Antworten)?
* TTS (bei langsamem Modell)?

## Aufbau im Detail

```
lab/
├── docker-compose.yml
├── .env.example
├── voice-app/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app.py            # FastRTC Stream + Handler
│   ├── pipeline.py       # Sequential vs. Streaming Pipeline
│   ├── metrics.py        # TurnTrace + SQLite
│   └── sentence_split.py # Streaming-Sentence-Splitter
└── metrics-ui/
    ├── Dockerfile
    ├── requirements.txt
    └── dashboard.py      # Streamlit Dashboard
```

## Troubleshooting

**`Connection refused` auf Port 8000:** Container braucht kurz für den Start.
`docker compose logs -f voice-app` beobachten.

**Kein Mikrofon-Zugriff:** Browser braucht `localhost` oder HTTPS. Bei
`http://192.168.x.x` blockiert Chrome WebRTC. `localhost` verwenden.

**Whisper findet kein Audio:** Wenn der Browser unter Windows kein Mikrofon
sendet, Edge oder Chrome verwenden. Firefox hat manchmal Probleme mit FastRTC.

**Groq Rate-Limit:** Free Tier ist begrenzt. Bei `429` etwas warten oder
einen Bezahl-Account verwenden.
