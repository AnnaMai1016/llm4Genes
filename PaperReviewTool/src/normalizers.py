"""
Normalize ``typing.Literal[...]`` string labels via fuzzy or embedding similarity.

Author: Zewei Ma <zewei.ma@outlook.com>
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable
import difflib
import numpy as np

class LiteralNormalizer(ABC):
    @abstractmethod
    def normalize(self, value: str, allowed: list[str]) -> str | None:
        ...


# --------------------------------------------------
# Fuzzy (string-based)
# --------------------------------------------------

class FuzzyNormalizer(LiteralNormalizer):
    def __init__(self, cutoff: float = 0.7):
        self.cutoff = cutoff

    def normalize(self, value: str, allowed: list[str]) -> str | None:
        matches = difflib.get_close_matches(
            value, allowed, n=1, cutoff=self.cutoff
        )
        return matches[0] if matches else None


# --------------------------------------------------
# Embedding-based (semantic)
# --------------------------------------------------

class EmbeddingNormalizer(LiteralNormalizer):
    def __init__(
        self,
        *,
        encoder: Callable[[str | list[str]], np.ndarray] | None = None,
        similarity_threshold: float = 0.75,
    ):
        """
        encoder: callable(str | list[str]) -> np.ndarray
        """
        if encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as e:
                raise ImportError(
                    "EmbeddingNormalizer requires the 'sentence-transformers' package. "
                    "Install it with 'pip install sentence-transformers' (and ensure PyTorch is available)."
                ) from e

            # Default to a simple sentence transformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            encoder = lambda texts: model.encode(texts, normalize_embeddings=True)

        self.encoder = encoder
        self.similarity_threshold = similarity_threshold
        self._cache: dict[str, np.ndarray] = {}

    def _embed(self, text: str) -> np.ndarray:
        if text not in self._cache:
            vec = self.encoder(text)
            # Some encoders return a 2D array when a list is provided; ensure 1D vector
            try:
                if isinstance(vec, np.ndarray) and getattr(vec, "ndim", 1) > 1:
                    vec = vec[0]
            except Exception:
                pass
            self._cache[text] = vec
        return self._cache[text]

    def normalize(self, value: str, allowed: list[str]) -> str | None:
        v = self._embed(value)

        best_score = -1.0
        best_label = None

        for label in allowed:
            l = self._embed(label)
            score = cosine_similarity(v, l)

            if score > best_score:
                best_score = score
                best_label = label

        if best_score >= self.similarity_threshold:
            return best_label

        return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.dot(a, b) /
        (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    )


# --------------------------------------------------
# Hybrid
# --------------------------------------------------

class HybridNormalizer(LiteralNormalizer):
    def __init__(
        self,
        fuzzy: FuzzyNormalizer,
        embedding: EmbeddingNormalizer | None = None,
    ):
        self.fuzzy = fuzzy
        self.embedding = embedding

    def normalize(self, value: str, allowed: list[str]) -> str | None:
        # 1️⃣ cheap lexical match first
        out = self.fuzzy.normalize(value, allowed)
        if out is not None:
            return out

        # 2️⃣ semantic fallback
        if self.embedding:
            return self.embedding.normalize(value, allowed)

        return None
