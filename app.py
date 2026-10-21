"""Streamlit interface for the YouTube YouTube InsightRAG."""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from ingestion import IngestionError, ingest_playlist
from qdrant_manager import QdrantManager
from rag_chain import RevisionRAG

st.set_page_config(
    page_title="YouTube InsightRAG",
    page_icon="🎓",
    layout="centered",
)

st.title("🎓 YouTube InsightRAG")
st.write(
    "Turn a YouTube playlist into a searchable study companion. "
    "Answers are grounded in its transcripts and link back to the relevant moments."
)

with st.sidebar:
    st.header("Setup status")
    st.caption("API keys are read from your `.env` file and are never displayed.")
    st.write(
        f"{'✅' if os.getenv('GROQ_API_KEY') else '⚠️'} Groq API key "
        f"{'configured' if os.getenv('GROQ_API_KEY') else 'missing'}"
    )
    st.write(
        f"{'✅' if os.getenv('QDRANT_URL') else '⚠️'} Qdrant URL "
        f"{'configured' if os.getenv('QDRANT_URL') else 'missing'}"
    )
    st.write(f"🗣️ Local Whisper model: `{os.getenv('WHISPER_MODEL', 'base')}`")
    st.divider()
    st.caption(
        "Playlist audio is transcribed locally and stored in the configured Qdrant "
        "collection. Only process videos you are authorized to transcribe."
    )

ingest_tab, ask_tab = st.tabs(["📥 Add a playlist", "💬 Ask a question"])

with ingest_tab:
    st.subheader("Index playlist transcripts")
    st.caption(
        "Paste a YouTube playlist link or playlist ID. yt-dlp extracts audio and "
        "Whisper transcribes it locally. Videos that cannot be processed are listed "
        "in the results."
    )
    with st.form("ingest_playlist_form"):
        playlist = st.text_input(
            "YouTube playlist",
            placeholder="https://www.youtube.com/playlist?list=…",
        )
        ingest_submitted = st.form_submit_button(
            "Fetch transcripts and index",
            type="primary",
            use_container_width=True,
        )

    if ingest_submitted:
        if not playlist.strip():
            st.error("Enter a YouTube playlist URL or playlist ID.")
        else:
            try:
                with st.spinner(
                    "Extracting audio, transcribing with Whisper, and embedding chunks… "
                    "The first run downloads the model; large playlists can take a while."
                ):
                    report = ingest_playlist(playlist.strip())
                st.success(
                    f"Indexed {report.videos_ingested} of {report.videos_found} videos "
                    f"and stored {report.chunks_stored} transcript chunks."
                )
                if report.skipped_videos:
                    with st.expander(f"Skipped videos ({len(report.skipped_videos)})"):
                        for skipped in report.skipped_videos:
                            st.write(f"- {skipped}")
            except (ValueError, IngestionError, RuntimeError) as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")
                st.exception(exc)

with ask_tab:
    st.subheader("Ask your study question")
    st.caption(
        "Questions search the configured Qdrant collection and include clickable "
        "video title and timestamp citations."
    )
    with st.form("ask_question_form"):
        question = st.text_area(
            "Question",
            placeholder="What are the main steps of gradient descent?",
            height=100,
        )
        top_k = st.slider("Transcript chunks to retrieve", min_value=3, max_value=5, value=4)
        ask_submitted = st.form_submit_button(
            "Get an answer",
            type="primary",
            use_container_width=True,
        )

    if ask_submitted:
        if not question.strip():
            st.error("Enter a study question.")
        else:
            try:
                with st.spinner("Searching your indexed transcripts and generating an answer…"):
                    answer = RevisionRAG(qdrant_manager=QdrantManager()).ask(
                        question.strip(),
                        top_k=top_k,
                    )
                st.markdown("### Answer")
                st.markdown(answer)
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Could not answer the question: {exc}")
                st.exception(exc)
