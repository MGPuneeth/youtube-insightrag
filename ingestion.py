"""Local YouTube audio transcription and timestamp-aware playlist chunking."""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import parse_qs, quote, urlparse

from dotenv import load_dotenv
import yt_dlp

from qdrant_manager import QdrantManager


class IngestionError(RuntimeError):
    """Raised when playlist extraction or transcription cannot continue."""


class VideoTranscriptionError(IngestionError):
    """Raised when a playlist video's audio cannot be transcribed."""


@dataclass(frozen=True)
class TranscriptChunk:
    video_id: str
    video_title: str
    start_time: float
    end_time: float
    text_chunk: str

    @property
    def timestamp_url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}&t={int(self.start_time)}s"

    def to_payload(self) -> dict[str, Any]:
        return {
            "text_chunk": self.text_chunk,
            "video_id": self.video_id,
            "video_title": self.video_title,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "youtube_url": self.timestamp_url,
        }


@dataclass(frozen=True)
class IngestionReport:
    videos_found: int
    videos_ingested: int
    chunks_stored: int
    skipped_videos: tuple[str, ...]


def normalize_playlist_input(playlist_input: str) -> str:
    """Return the playlist ID for a standard YouTube URL, or preserve an ID."""
    value = playlist_input.strip()
    if not value:
        raise ValueError("Provide a YouTube playlist URL or playlist ID.")

    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc:
        playlist_ids = parse_qs(parsed.query).get("list")
        if playlist_ids and playlist_ids[0].strip():
            return playlist_ids[0].strip()
    return value


def _to_finite_number(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise IngestionError(f"Transcript segment has an invalid {field_name}.") from exc
    if not math.isfinite(number):
        raise IngestionError(f"Transcript segment has an invalid {field_name}.")
    return number


def build_time_chunks(
    video_id: str,
    video_title: str,
    segments: list[dict[str, Any]],
    target_seconds: float = 100.0,
    max_seconds: float = 120.0,
) -> list[TranscriptChunk]:
    """Group consecutive caption segments into roughly 100-second chunks."""
    if target_seconds <= 0 or max_seconds < target_seconds:
        raise ValueError("max_seconds must be greater than or equal to a positive target_seconds.")

    normalized: list[tuple[float, float, str]] = []
    for segment in segments:
        if not isinstance(segment, dict):
            raise IngestionError("Transcript contains an invalid segment.")
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        start = _to_finite_number(segment.get("start"), "start time")
        duration = _to_finite_number(segment.get("duration"), "duration")
        if start < 0 or duration < 0:
            raise IngestionError("Transcript segment times cannot be negative.")
        if duration > max_seconds:
            raise IngestionError(
                f"Transcript segment duration ({duration:g}s) exceeds the "
                f"{max_seconds:g}s maximum chunk duration."
            )
        normalized.append((start, duration, text))

    normalized.sort(key=lambda segment: segment[0])
    chunks: list[TranscriptChunk] = []
    current: list[tuple[float, float, str]] = []

    def flush() -> None:
        if not current:
            return
        start_time = current[0][0]
        end_time = max(start + duration for start, duration, _ in current)
        chunks.append(
            TranscriptChunk(
                video_id=video_id,
                video_title=video_title,
                start_time=start_time,
                end_time=end_time,
                text_chunk=" ".join(text for _, _, text in current),
            )
        )
        current.clear()

    for segment in normalized:
        start, duration, _ = segment
        if current and start + duration - current[0][0] > max_seconds:
            flush()
        current.append(segment)
        if max(start + duration for start, duration, _ in current) - current[0][0] >= target_seconds:
            flush()
    flush()
    return chunks


class WhisperYouTubeClient:
    """Extract playlist audio with yt-dlp and transcribe locally with faster-whisper."""

    def __init__(self, model_size: str | None = None, model: Any | None = None) -> None:
        load_dotenv()
        self.model_size = model_size or os.getenv("WHISPER_MODEL", "base").strip()
        self._model = model

    @staticmethod
    def _playlist_url(playlist_input: str) -> str:
        playlist = normalize_playlist_input(playlist_input)
        parsed = urlparse(playlist_input.strip())
        if parsed.scheme and parsed.netloc:
            return playlist_input.strip()
        return f"https://www.youtube.com/playlist?list={quote(playlist, safe='')}"

    def _get_model(self) -> Any:
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
            )
        return self._model

    def iter_playlist_videos(self, playlist_input: str) -> Iterator[dict[str, Any]]:
        options = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": "in_playlist",
            "skip_download": True,
        }
        try:
            with yt_dlp.YoutubeDL(options) as downloader:
                playlist_info = downloader.extract_info(
                    self._playlist_url(playlist_input),
                    download=False,
                )
        except yt_dlp.utils.DownloadError as exc:
            raise IngestionError(f"Could not read the YouTube playlist: {exc}") from exc

        entries = playlist_info.get("entries") if isinstance(playlist_info, dict) else None
        if entries is None:
            raise IngestionError("yt-dlp did not return a playlist. Check the playlist URL or ID.")

        found_video = False
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            video_id = entry.get("id")
            if not video_id:
                continue
            found_video = True
            yield {
                "video_id": str(video_id),
                "title": str(entry.get("title") or video_id),
            }
        if not found_video:
            raise IngestionError("The playlist contains no accessible videos.")

    def fetch_transcript(self, video_id: str) -> dict[str, Any]:
        video_url = f"https://www.youtube.com/watch?v={quote(video_id, safe='')}"
        with tempfile.TemporaryDirectory(prefix="playlist-rag-") as temp_dir:
            output_template = str(Path(temp_dir) / "%(id)s.%(ext)s")
            options = {
                "quiet": True,
                "no_warnings": True,
                "noplaylist": True,
                "format": "bestaudio/best",
                "outtmpl": output_template,
            }
            try:
                with yt_dlp.YoutubeDL(options) as downloader:
                    video_info = downloader.extract_info(video_url, download=True)
                    if not isinstance(video_info, dict):
                        raise VideoTranscriptionError(
                            f"yt-dlp returned no video data for {video_id}."
                        )
                    audio_path = Path(downloader.prepare_filename(video_info))
            except yt_dlp.utils.DownloadError as exc:
                raise VideoTranscriptionError(
                    f"Could not download audio for {video_id}: {exc}"
                ) from exc

            if not audio_path.is_file():
                raise VideoTranscriptionError(
                    f"yt-dlp did not produce an audio file for {video_id}."
                )

            try:
                whisper_model = self._get_model()
                whisper_segments, _ = whisper_model.transcribe(
                    str(audio_path),
                    beam_size=5,
                    vad_filter=True,
                )
                segments = [
                    {
                        "start": float(segment.start),
                        "duration": max(0.0, float(segment.end) - float(segment.start)),
                        "text": segment.text.strip(),
                    }
                    for segment in whisper_segments
                    if segment.text.strip()
                ]
            except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
                raise VideoTranscriptionError(
                    f"Whisper could not transcribe {video_id}: {exc}"
                ) from exc

        return {
            "transcript": segments,
            "metadata": {"title": str(video_info.get("title") or video_id)},
        }


def ingest_playlist(
    playlist_input: str,
    transcription_client: WhisperYouTubeClient | None = None,
    qdrant_manager: QdrantManager | None = None,
) -> IngestionReport:
    """Fetch playlist transcripts, chunk them, and upsert their vectors."""
    transcription_client = transcription_client or WhisperYouTubeClient()
    qdrant_manager = qdrant_manager or QdrantManager()
    qdrant_manager.ensure_collection()

    videos_found = 0
    videos_ingested = 0
    chunks_stored = 0
    skipped: list[str] = []
    for video in transcription_client.iter_playlist_videos(playlist_input):
        videos_found += 1
        video_id = video["video_id"]
        try:
            response = transcription_client.fetch_transcript(video_id)
        except VideoTranscriptionError as exc:
            skipped.append(f"{video_id}: {exc}")
            continue

        segments = response.get("transcript")
        if not isinstance(segments, list) or not segments:
            skipped.append(f"{video_id}: no timestamped transcript segments returned")
            continue
        metadata = response.get("metadata")
        metadata_title = metadata.get("title") if isinstance(metadata, dict) else None
        title = str(metadata_title or video["title"])
        chunks = build_time_chunks(video_id, title, segments)
        if not chunks:
            skipped.append(f"{video_id}: transcript contains no usable caption text")
            continue
        stored = qdrant_manager.add_chunks(chunks)
        chunks_stored += stored
        videos_ingested += 1

    return IngestionReport(
        videos_found=videos_found,
        videos_ingested=videos_ingested,
        chunks_stored=chunks_stored,
        skipped_videos=tuple(skipped),
    )
