"""Qdrant collection and FastEmbed vector operations."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterable
from typing import Any

from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

from ingestion_types import ChunkLike

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE = 384


class QdrantManager:
    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str | None = None,
        client: QdrantClient | None = None,
        embedder: Any | None = None,
    ) -> None:
        load_dotenv()
        self.url = url or os.getenv("QDRANT_URL", "").strip()
        self.api_key = api_key if api_key is not None else os.getenv("QDRANT_API_KEY", "").strip()
        self.collection_name = collection_name or os.getenv(
            "QDRANT_COLLECTION", "youtube_revision_chunks"
        )
        if client is None and not self.url:
            raise ValueError("QDRANT_URL is missing. Set it in your .env file.")
        self.client = client or QdrantClient(
            url=self.url,
            api_key=self.api_key or None,
            timeout=30,
        )
        self.embedder = embedder or TextEmbedding(model_name=EMBEDDING_MODEL)

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )

    def add_chunks(self, chunks: Iterable[ChunkLike], batch_size: int = 64) -> int:
        self.ensure_collection()
        stored = 0
        batch: list[ChunkLike] = []
        for chunk in chunks:
            batch.append(chunk)
            if len(batch) >= batch_size:
                self._upsert_batch(batch)
                stored += len(batch)
                batch.clear()
        if batch:
            self._upsert_batch(batch)
            stored += len(batch)
        return stored

    def _upsert_batch(self, chunks: list[ChunkLike]) -> None:
        vectors = list(self.embedder.embed([chunk.text_chunk for chunk in chunks]))
        if len(vectors) != len(chunks):
            raise RuntimeError("Embedding model returned a different number of vectors than chunks.")
        points = []
        for chunk, vector in zip(chunks, vectors):
            point_id = uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{chunk.video_id}:{chunk.start_time:.3f}:{chunk.end_time:.3f}",
            )
            points.append(
                models.PointStruct(
                    id=str(point_id),
                    vector=vector.tolist(),
                    payload=chunk.to_payload(),
                )
            )
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    def search(self, query: str, limit: int = 4) -> list[dict[str, Any]]:
        if not 3 <= limit <= 5:
            raise ValueError("Retrieval limit must be between 3 and 5.")
        self.ensure_collection()
        query_vector = next(self.embedder.query_embed(query)).tolist()
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        return [
            {
                **(point.payload or {}),
                "score": point.score,
            }
            for point in results.points
        ]
