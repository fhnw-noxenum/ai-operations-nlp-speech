"""
Latenz-Dashboard für die Voice-Pipeline.

Liest read-only aus der SQLite-Datei, die voice-app schreibt.
Zeigt pro Mode (sequential / streaming) Latenz-Verteilung und Bottleneck.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


DB_PATH = os.environ.get("METRICS_DB", "/data/metrics.sqlite")

FHNW_YELLOW = "#FFCC00"
DARK = "#2C2C2C"
GREY = "#E5E5E5"

st.set_page_config(page_title="Speech Lab · Latency Dashboard",
                   page_icon="🎤", layout="wide")


# ---------------------------------------------------------------- styling

st.markdown(
    f"""
    <style>
      .stApp {{ font-family: Arial, sans-serif; }}
      h1, h2, h3 {{ color: {DARK}; }}
      h1 {{ border-bottom: 4px solid {FHNW_YELLOW};
            padding-bottom: 0.3em; }}
      [data-testid="stMetricValue"] {{ color: {DARK}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- data

@st.cache_data(ttl=2.0)
def load(limit: int = 200) -> pd.DataFrame:
    if not Path(DB_PATH).exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql(
            "SELECT * FROM turns ORDER BY ts_unix DESC LIMIT ?",
            con, params=(limit,),
        )
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts_unix"], unit="s")
    for col in ("t_stt_done", "t_llm_first", "t_llm_done",
                "t_tts_first", "t_total"):
        df[col + "_ms"] = (df[col] * 1000).round(0)
    return df


def reset_metrics() -> None:
    if not Path(DB_PATH).exists():
        return
    with sqlite3.connect(DB_PATH) as con:
        con.execute("DELETE FROM turns")
        con.commit()
    with sqlite3.connect(DB_PATH, isolation_level=None) as con:
        con.execute("VACUUM")


# ---------------------------------------------------------------- UI

st.title("🎤  Speech Processing Lab – Latency")

col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
with col1:
    if st.button("🔄  Refresh", width="stretch"):
        st.cache_data.clear()
        st.rerun()
with col2:
    if st.button("Reset metrics", width="stretch"):
        reset_metrics()
        st.cache_data.clear()
        st.rerun()
with col3:
    auto = st.toggle("Auto-Refresh (5 s)", value=False)
with col4:
    st.caption(f"DB: `{DB_PATH}`")

if auto:
    st.html("<meta http-equiv='refresh' content='5'>")

df = load(limit=200)

if df.empty:
    st.warning(
        "Noch keine Turns aufgezeichnet. "
        "Starten Sie voice-app und sprechen Sie etwas in den Browser."
    )
    st.stop()


# ---------------------- summary metrics per mode ----------------------

st.header("Übersicht pro Mode")

mode_summary = (
    df.groupby("mode")
    .agg(
        n_turns=("ts_unix", "count"),
        ttft_p50=("t_llm_first_ms", lambda s: s.quantile(0.50)),
        ttft_p95=("t_llm_first_ms", lambda s: s.quantile(0.95)),
        ttfa_p50=("t_tts_first_ms", lambda s: s.quantile(0.50)),
        ttfa_p95=("t_tts_first_ms", lambda s: s.quantile(0.95)),
        total_p50=("t_total_ms", lambda s: s.quantile(0.50)),
    )
    .reset_index()
)

cols = st.columns(len(mode_summary))
for c, (_, row) in zip(cols, mode_summary.iterrows()):
    with c:
        st.subheader(f"Mode: `{row['mode']}`")
        st.metric("Turns", int(row["n_turns"]))
        a, b = st.columns(2)
        a.metric("TTFT P50", f"{row['ttft_p50']:.0f} ms")
        b.metric("TTFT P95", f"{row['ttft_p95']:.0f} ms")
        a.metric("TTFA P50", f"{row['ttfa_p50']:.0f} ms")
        b.metric("TTFA P95", f"{row['ttfa_p95']:.0f} ms")
        st.metric("Total P50", f"{row['total_p50']:.0f} ms")


# ---------------------- latency per stage ----------------------

st.header("Latenz pro Stufe (P50)")

stage_long = (
    df.groupby("mode")[["t_stt_done_ms", "t_llm_first_ms", "t_tts_first_ms"]]
    .median()
    .rename(columns={
        "t_stt_done_ms": "STT",
        "t_llm_first_ms": "LLM TTFT",
        "t_tts_first_ms": "TTS TTFA",
    })
    .reset_index()
    .melt(id_vars="mode", var_name="stage", value_name="ms")
)

bar = (
    alt.Chart(stage_long)
    .mark_bar()
    .encode(
        x=alt.X("stage:N", title=None,
                sort=["STT", "LLM TTFT", "TTS TTFA"]),
        y=alt.Y("ms:Q", title="Latenz (ms, P50)"),
        color=alt.Color(
            "mode:N",
            scale=alt.Scale(
                domain=["sequential", "streaming"],
                range=[DARK, FHNW_YELLOW],
            ),
        ),
        xOffset="mode:N",
        tooltip=["mode", "stage", "ms"],
    )
    .properties(height=320)
)
st.altair_chart(bar, width="stretch")


# ---------------------- timeline ----------------------

st.header("Zeitlicher Verlauf · TTFA pro Turn")

line = (
    alt.Chart(df.sort_values("ts"))
    .mark_line(point=True)
    .encode(
        x=alt.X("ts:T", title="Zeit"),
        y=alt.Y("t_tts_first_ms:Q", title="TTFA (ms)"),
        color=alt.Color(
            "mode:N",
            scale=alt.Scale(
                domain=["sequential", "streaming"],
                range=[DARK, FHNW_YELLOW],
            ),
        ),
        tooltip=["ts:T", "mode", "t_tts_first_ms",
                 "transcript"],
    )
    .properties(height=300)
)
st.altair_chart(line, width="stretch")


# ---------------------- recent turns ----------------------

st.header("Letzte Turns")

cols_to_show = ["ts", "mode", "t_stt_done_ms", "t_llm_first_ms",
                "t_tts_first_ms", "t_total_ms", "transcript", "response"]
table = df[cols_to_show].head(20).rename(columns={
    "t_stt_done_ms": "STT (ms)",
    "t_llm_first_ms": "TTFT (ms)",
    "t_tts_first_ms": "TTFA (ms)",
    "t_total_ms": "Total (ms)",
})
st.dataframe(table, width="stretch", hide_index=True)


# ---------------------- insight box ----------------------

if len(mode_summary) == 2:
    seq = mode_summary[mode_summary["mode"] == "sequential"]
    stm = mode_summary[mode_summary["mode"] == "streaming"]
    if not seq.empty and not stm.empty:
        diff = float(seq["ttfa_p50"].iloc[0] - stm["ttfa_p50"].iloc[0])
        pct = 100 * diff / max(float(seq["ttfa_p50"].iloc[0]), 1.0)
        st.success(
            f"📉  Streaming spart {diff:.0f} ms TTFA (≈ {pct:.0f} % "
            f"schneller) gegenüber sequentiell."
        )
