"""Deterministic eval providers — no models, no network, reproducible runs.

HashedBowEmbed gives vector search real (if crude) semantics: tokens hash to
buckets, synonyms collapse to one bucket, vectors are L2-normalized counts.
Paraphrases that share vocabulary (or synonyms) land near each other; unrelated
text doesn't. That's enough signal to exercise the retrieval pipeline
deterministically.
"""

import hashlib
import math
import re

# Synonym groups: all words in a group collapse to the first entry's bucket.
# This is what makes the embedder "semantic" beyond exact keyword overlap.
DEFAULT_SYNONYMS = [
    ["written", "implemented", "coded", "built", "developed"],
    ["language", "programming"],
    ["based", "located", "situated"],
    ["leads", "heads", "runs", "manages"],
    ["prefers", "likes", "favors"],
    ["communicate", "communication", "talk"],
    ["engineer", "developer", "programmer"],
    ["database", "datastore"],
    ["nightly", "daily"],
    ["shipped", "launched", "released"],
]

_WORD_RE = re.compile(r"[a-z0-9]+")


class HashedBowEmbed:
    """Deterministic bag-of-words embedder over hashed vocabulary buckets."""

    def __init__(self, dims: int = 256, synonyms: list[list[str]] | None = None):
        self.dims = dims
        self._canon: dict[str, str] = {}
        for group in (synonyms if synonyms is not None else DEFAULT_SYNONYMS):
            for word in group:
                self._canon[word] = group[0]

    def _bucket(self, token: str) -> int:
        digest = hashlib.md5(token.encode()).hexdigest()
        return int(digest, 16) % self.dims

    def _tokens(self, text: str) -> list[str]:
        words = _WORD_RE.findall(text.lower())
        return [self._canon.get(w, w) for w in words]

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        for tok in self._tokens(text):
            vec[self._bucket(tok)] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0:
            vec[0] = 1.0
            return vec
        return [x / norm for x in vec]

    def embed_query(self, text: str) -> list[float]:
        return self.embed(text)


class NullLLM:
    """LLM stub — eval corpus is built directly, extraction never runs."""

    def call(self, messages, model, max_tokens, source=""):
        return "[]"
