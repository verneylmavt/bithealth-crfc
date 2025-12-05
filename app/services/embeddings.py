# app/services/embeddings.py: Embedding Service
# to encapsulate embedding generation behind a class so it is easy to swap or mock.

from typing import List
import random


class EmbeddingService:
    """Abstract interface for embedding text into vectors."""

    def embed(self, text: str) -> List[float]:
        raise NotImplementedError
        

class FakeEmbeddingService(EmbeddingService):
    """
    Deterministic fake embedding service.

    This reproduces the behaviour of the original fake_embed function:
    - The embedding is a list of `dim` random floats.
    - A seed derived from the text ensures determinism across calls.
    """

    def __init__(self, dim: int = 128) -> None:
        self._dim = dim

    def embed(self, text: str) -> List[float]:
        # Seed based on input so it's deterministic per text
        seed = abs(hash(text)) % 10000
        rng = random.Random(seed)
        return [rng.random() for _ in range(self._dim)]
