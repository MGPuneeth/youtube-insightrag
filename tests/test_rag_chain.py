import unittest
from types import SimpleNamespace

from rag_chain import RevisionRAG


class FakeQdrant:
    def search(self, query, limit):
        return [
            {
                "video_id": "abcdefghijk",
                "video_title": "Lesson",
                "start_time": 135.0,
                "end_time": 180.0,
                "text_chunk": "Gradient descent updates model parameters.",
                "youtube_url": "https://www.youtube.com/watch?v=abcdefghijk&t=135s",
                "score": 0.9,
            }
        ]


class FakeGroq:
    class Chat:
        class Completions:
            @staticmethod
            def create(**kwargs):
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="It updates parameters. [S1]")
                        )
                    ]
                )

        completions = Completions()

    chat = Chat()


class RevisionRAGTests(unittest.TestCase):
    def test_answers_include_resolved_clickable_timestamp_citations(self) -> None:
        rag = RevisionRAG(qdrant_manager=FakeQdrant(), groq_client=FakeGroq())

        answer = rag.ask("What does gradient descent do?")

        expected_link = (
            "[Lesson at 02:15]"
            "(https://www.youtube.com/watch?v=abcdefghijk&t=135s)"
        )
        self.assertIn(expected_link, answer)
        self.assertIn(f"Sources:\n- {expected_link}", answer)

    def test_empty_question_is_rejected(self) -> None:
        rag = RevisionRAG(qdrant_manager=FakeQdrant(), groq_client=FakeGroq())

        with self.assertRaises(ValueError):
            rag.ask(" ")


if __name__ == "__main__":
    unittest.main()
