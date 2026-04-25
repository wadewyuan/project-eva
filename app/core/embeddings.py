import struct

import numpy as np
from fastembed import TextEmbedding


class EmbeddingModel:
    """Lightweight wrapper around fastembed for Chinese text embeddings."""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5") -> None:
        self.model = TextEmbedding(model_name)
        self.dim: int = self.model.embedding_size

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of float vectors."""
        return [list(vec) for vec in self.model.embed(texts)]

    def encode_one(self, text: str) -> list[float]:
        """Embed a single text."""
        return self.encode([text])[0]

    # ---------- Serialization for SQLite BLOB ----------

    def vector_to_blob(self, vector: list[float]) -> bytes:
        """Pack a float vector into bytes for SQLite BLOB storage."""
        return struct.pack(f"<{len(vector)}f", *vector)

    def blob_to_vector(self, blob: bytes) -> list[float]:
        """Unpack a BLOB back into a float vector."""
        count = len(blob) // 4
        return list(struct.unpack(f"<{count}f", blob))

    # ---------- Similarity ----------

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """
        Compute cosine similarity between two vectors.
        For L2-normalized vectors (BGE), this is just the dot product.
        """
        return sum(x * y for x, y in zip(a, b))

    @staticmethod
    def cosine_similarity_np(a: np.ndarray, b: np.ndarray) -> float:
        """NumPy-accelerated cosine similarity."""
        return float(np.dot(a, b))


embedding_model = EmbeddingModel()
