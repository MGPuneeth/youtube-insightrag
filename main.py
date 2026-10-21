"""Command-line interface for ingesting and querying a YouTube playlist."""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Index YouTube playlist transcripts and ask timestamp-grounded study questions."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a YouTube playlist")
    ingest_parser.add_argument("playlist", help="YouTube playlist URL or playlist ID")

    ask_parser = subparsers.add_parser("ask", help="Ask a revision question")
    ask_parser.add_argument("question", help="Study or revision question")
    ask_parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        choices=range(3, 6),
        metavar="{3,4,5}",
        help="Number of transcript chunks to retrieve (default: 4)",
    )
    return parser


def main() -> int:
    load_dotenv()
    from ingestion import IngestionError, ingest_playlist
    from qdrant_manager import QdrantManager
    from rag_chain import RevisionRAG

    args = build_parser().parse_args()
    try:
        if args.command == "ingest":
            report = ingest_playlist(args.playlist)
            print(
                f"Playlist videos: {report.videos_found}; "
                f"transcribed videos indexed: {report.videos_ingested}; "
                f"chunks stored: {report.chunks_stored}."
            )
            if report.skipped_videos:
                print("Skipped videos:")
                for skipped in report.skipped_videos:
                    print(f"  - {skipped}")
            return 0

        rag = RevisionRAG(qdrant_manager=QdrantManager())
        print(rag.ask(args.question, top_k=args.top_k))
        return 0
    except (ValueError, IngestionError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
