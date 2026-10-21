import unittest

import numpy as np
from qdrant_client import QdrantClient

from ingestion import TranscriptChunk
from qdrant_manager import QdrantManager


class StubEmbedder:
    @staticmethod
    def _vector(_text):
        vector = np.zeros(384, dtype=np.float32)
        vector[0] = 1
        return vector

    def embed(self, texts):
        return iter(self._vector(text) for text in texts)

    def query_embed(self, text):
        return iter([self._vector(text)])


class QdrantManagerTests(unittest.TestCase):
    def test_upserted_chunk_is_retrievable_with_payload(self) -> None:
        manager = QdrantManager(
            collection_name="test_chunks",
            client=QdrantClient(":memory:"),
            embedder=StubEmbedder(),
        )
        chunk = TranscriptChunk(
            video_id="abcdefghijk",
            video_title="Lesson",
            start_time=20,
            end_time=110,
            text_chunk="A test chunk about vector search.",
        )

        self.assertEqual(manager.add_chunks([chunk]), 1)
        results = manager.search("vector search", limit=3)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["video_id"], "abcdefghijk")
        self.assertEqual(results[0]["text_chunk"], chunk.text_chunk)
        self.assertEqual(results[0]["youtube_url"], chunk.timestamp_url)

    def test_search_enforces_requested_top_k_range(self) -> None:
        manager = QdrantManager(
            collection_name="test_chunks",
            client=QdrantClient(":memory:"),
            embedder=StubEmbedder(),
        )
        with self.assertRaises(ValueError):
            manager.search("query", limit=2)


if __name__ == "__main__":
    unittest.main()
