"""Structural type shared by ingestion chunks and the Qdrant manager."""

from typing import Any, Protocol


class ChunkLike(Protocol):
    video_id: str
    start_time: float
    end_time: float
    text_chunk: str

    def to_payload(self) -> dict[str, Any]: ...
