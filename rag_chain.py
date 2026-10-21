"""Groq-powered answers grounded in retrieved playlist transcript chunks."""

from __future__ import annotations

import os
import re
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from qdrant_manager import QdrantManager

SOURCE_MARKER = re.compile(r"\[S(\d+)\]")


def format_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"
    return f"{minutes:02d}:{seconds_part:02d}"


class RevisionRAG:
    def __init__(
        self,
        qdrant_manager: QdrantManager | None = None,
        groq_client: Any | None = None,
        model: str | None = None,
    ) -> None:
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if groq_client is None and not api_key:
            raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")
        self.client = groq_client or Groq(api_key=api_key)
        self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        self.qdrant = qdrant_manager or QdrantManager()

    def ask(self, question: str, top_k: int = 4) -> str:
        question = question.strip()
        if not question:
            raise ValueError("Enter a study question.")

        sources = self.qdrant.search(question, limit=top_k)
        if not sources:
            return "I couldn't find relevant transcript context in the indexed playlist."

        context = self._format_context(sources)
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise study revision assistant. Answer using only the "
                        "provided transcript context; if it does not contain the answer, say so. "
                        "Do not invent details or citations. Cite every substantive claim with "
                        "one or more source markers exactly like [S1] using the source numbers "
                        "provided. The application resolves each marker to a clickable Markdown "
                        "link formatted [Exact video title at HH:MM](the provided URL). Be concise "
                        "and explain concepts clearly."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nTranscript context:\n{context}",
                },
            ],
        )
        answer = completion.choices[0].message.content
        if not answer:
            raise RuntimeError("Groq returned an empty response.")
        answer = self._replace_source_markers(answer, sources)
        links = "\n".join(
            f"- [{item.get('video_title', item.get('video_id', 'Video'))} at "
            f"{format_timestamp(float(item['start_time']))}]({item['youtube_url']})"
            for item in sources
        )
        return f"{answer.strip()}\n\nSources:\n{links}"

    @staticmethod
    def _format_context(sources: list[dict[str, Any]]) -> str:
        entries: list[str] = []
        for index, source in enumerate(sources, start=1):
            entries.append(
                f"[S{index}] Title: {source.get('video_title', source.get('video_id', 'Video'))}\n"
                f"Timestamp: {format_timestamp(float(source['start_time']))}\n"
                f"Text: {source['text_chunk']}\n"
                f"URL: {source['youtube_url']}"
            )
        return "\n\n".join(entries)

    @staticmethod
    def _replace_source_markers(answer: str, sources: list[dict[str, Any]]) -> str:
        def replace(match: re.Match[str]) -> str:
            index = int(match.group(1)) - 1
            if not 0 <= index < len(sources):
                return match.group(0)
            source = sources[index]
            title = source.get("video_title", source.get("video_id", "Video"))
            timestamp = format_timestamp(float(source["start_time"]))
            return f"[{title} at {timestamp}]({source['youtube_url']})"

        return SOURCE_MARKER.sub(replace, answer)
