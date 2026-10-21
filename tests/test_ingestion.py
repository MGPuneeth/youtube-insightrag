import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ingestion import (
    IngestionError,
    IngestionReport,
    VideoTranscriptionError,
    WhisperYouTubeClient,
    build_time_chunks,
    ingest_playlist,
    normalize_playlist_input,
)


class PlaylistInputTests(unittest.TestCase):
    def test_extracts_playlist_id_from_url(self) -> None:
        self.assertEqual(
            normalize_playlist_input("https://www.youtube.com/playlist?list=PLabc123"),
            "PLabc123",
        )

    def test_preserves_playlist_id(self) -> None:
        self.assertEqual(normalize_playlist_input("PLabc123"), "PLabc123")

    def test_rejects_empty_playlist(self) -> None:
        with self.assertRaises(ValueError):
            normalize_playlist_input(" ")

    def test_playlist_videos_are_read_with_yt_dlp(self) -> None:
        client = WhisperYouTubeClient()
        with patch("ingestion.yt_dlp.YoutubeDL") as youtube_dl:
            downloader = youtube_dl.return_value.__enter__.return_value
            downloader.extract_info.return_value = {
                "entries": [{"id": "abcdefghijk", "title": "Lesson"}],
            }

            videos = list(client.iter_playlist_videos("PLabc123"))

        options = youtube_dl.call_args.args[0]
        self.assertTrue(options["skip_download"])
        downloader.extract_info.assert_called_once_with(
            "https://www.youtube.com/playlist?list=PLabc123",
            download=False,
        )
        self.assertEqual(videos, [{"video_id": "abcdefghijk", "title": "Lesson"}])

    def test_audio_is_transcribed_to_timestamped_segments(self) -> None:
        class FakeModel:
            @staticmethod
            def transcribe(audio_path, **_kwargs):
                self_path = Path(audio_path)
                if not self_path.is_file():
                    raise AssertionError("yt-dlp should create audio before transcription")
                return iter(
                    [
                        SimpleNamespace(start=12.5, end=15.0, text=" lesson caption "),
                    ]
                ), SimpleNamespace()

        class FakeDownloader:
            def __init__(self, options):
                self.output_path = Path(
                    options["outtmpl"].replace("%(id)s.%(ext)s", "abcdefghijk.webm")
                )

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def extract_info(self, _url, download):
                if download:
                    self.output_path.write_bytes(b"fake audio")
                return {"id": "abcdefghijk", "title": "Lesson"}

            def prepare_filename(self, _info):
                return str(self.output_path)

        client = WhisperYouTubeClient(model=FakeModel())
        with patch("ingestion.yt_dlp.YoutubeDL", side_effect=FakeDownloader):
            transcript = client.fetch_transcript("abcdefghijk")

        self.assertEqual(
            transcript["transcript"],
            [{"start": 12.5, "duration": 2.5, "text": "lesson caption"}],
        )
        self.assertEqual(transcript["metadata"]["title"], "Lesson")

    def test_transcription_type_errors_are_reported_as_video_errors(self) -> None:
        class FakeModel:
            @staticmethod
            def transcribe(_audio_path, **_kwargs):
                raise TypeError("unsupported audio decoder argument")

        class FakeDownloader:
            def __init__(self, options):
                self.output_path = Path(
                    options["outtmpl"].replace("%(id)s.%(ext)s", "abcdefghijk.webm")
                )

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def extract_info(self, _url, download):
                if download:
                    self.output_path.write_bytes(b"fake audio")
                return {"id": "abcdefghijk", "title": "Lesson"}

            def prepare_filename(self, _info):
                return str(self.output_path)

        client = WhisperYouTubeClient(model=FakeModel())
        with patch("ingestion.yt_dlp.YoutubeDL", side_effect=FakeDownloader):
            with self.assertRaisesRegex(
                VideoTranscriptionError,
                "Whisper could not transcribe abcdefghijk: unsupported audio decoder argument",
            ):
                client.fetch_transcript("abcdefghijk")


class TimeChunkTests(unittest.TestCase):
    def test_groups_segments_and_preserves_timestamp_metadata(self) -> None:
        segments = [
            {"start": index * 10, "duration": 5, "text": f"caption {index}"}
            for index in range(11)
        ]
        chunks = build_time_chunks("abcdefghijk", "Lesson", segments)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].start_time, 0)
        self.assertEqual(chunks[0].end_time, 105)
        self.assertIn("caption 10", chunks[0].text_chunk)
        self.assertEqual(
            chunks[0].timestamp_url,
            "https://www.youtube.com/watch?v=abcdefghijk&t=0s",
        )
        self.assertEqual(chunks[0].to_payload()["youtube_url"], chunks[0].timestamp_url)

    def test_splits_chunks_at_target_duration(self) -> None:
        segments = [
            {"start": index * 10, "duration": 5, "text": f"caption {index}"}
            for index in range(25)
        ]
        chunks = build_time_chunks("abcdefghijk", "Lesson", segments)

        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(chunk.end_time - chunk.start_time <= 120 for chunk in chunks))
        self.assertEqual(chunks[0].start_time, 0)
        self.assertEqual(chunks[-1].end_time, 245)

    def test_ignores_blank_segments(self) -> None:
        chunks = build_time_chunks(
            "abcdefghijk",
            "Lesson",
            [{"start": 0, "duration": 1, "text": " "}],
        )
        self.assertEqual(chunks, [])

    def test_rejects_chunk_configuration_over_maximum(self) -> None:
        with self.assertRaises(ValueError):
            build_time_chunks("abcdefghijk", "Lesson", [], target_seconds=130, max_seconds=120)


class IngestionTests(unittest.TestCase):
    class FakeTranscriptClient:
        def iter_playlist_videos(self, _playlist):
            yield {"video_id": "abcdefghijk", "title": "Available lesson"}
            yield {"video_id": "lmnopqrstuv", "title": "No transcript"}

        def fetch_transcript(self, video_id):
            if video_id == "lmnopqrstuv":
                raise VideoTranscriptionError("Audio unavailable.")
            return {
                "transcript": [
                    {"start": 0, "duration": 5, "text": "A caption about study."}
                ]
            }

    class FakeQdrantManager:
        def __init__(self):
            self.chunks = []

        def ensure_collection(self):
            return None

        def add_chunks(self, chunks):
            self.chunks.extend(chunks)
            return len(chunks)

    def test_skips_missing_transcript_and_stores_available_chunks(self) -> None:
        qdrant = self.FakeQdrantManager()
        report = ingest_playlist(
            "PLabc123",
            transcription_client=self.FakeTranscriptClient(),
            qdrant_manager=qdrant,
        )

        self.assertEqual(
            report,
            IngestionReport(
                videos_found=2,
                videos_ingested=1,
                chunks_stored=1,
                skipped_videos=("lmnopqrstuv: Audio unavailable.",),
            ),
        )
        self.assertEqual(qdrant.chunks[0].video_id, "abcdefghijk")


if __name__ == "__main__":
    unittest.main()
