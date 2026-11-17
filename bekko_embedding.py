#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""32-dimensional static embeddings for English/ASCII text.

Run as a uv script (``uv run bekko_embedding.py "hello world"``) or import
``encode`` from code that has numpy. Model data: ``data/data.tar.gz``.
"""

from __future__ import annotations

import argparse
import json
import tarfile
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "StaticEmbedding",
    "WordPieceTokenizer",
    "encode",
    "load_model",
]

_DATA_FILE = Path(__file__).resolve().parent / "data" / "data.tar.gz"


@lru_cache(maxsize=1)
def _archive() -> dict[str, dict]:
    """Load the JSON assets from the tarball once, keyed by member name."""
    with tarfile.open(_DATA_FILE, "r:gz") as tar:
        return {
            member.name: json.load(tar.extractfile(member))
            for member in tar.getmembers()
            if member.isfile()
        }


def _asset(name: str, data_dir: str | Path | None = None) -> dict:
    """Read one JSON asset from the bundled archive or a custom raw-data dir."""
    if data_dir is None:
        asset = _archive()[name]
    else:
        asset = json.loads((Path(data_dir) / name).read_text(encoding="utf-8"))
    if not isinstance(asset, dict):
        raise ValueError(f"{name} must contain a JSON object")
    if asset.get("encoding", "raw") != "raw":
        raise ValueError(f"{name} must use raw JSON encoding")
    return asset


def _ascii_punctuation(char: str) -> bool:
    code = ord(char)
    return (
        33 <= code <= 47
        or 58 <= code <= 64
        or 91 <= code <= 96
        or 123 <= code <= 126
    )


def _normalize(text: str) -> str:
    """Bert-style clean/strip-accents/lowercase, restricted to ASCII output."""
    cleaned: list[str] = []
    for char in text:
        if char in "\t\n\r":
            cleaned.append(" ")
        elif unicodedata.category(char) == "Cc":
            continue
        else:
            cleaned.append(char)
    decomposed = unicodedata.normalize("NFKD", "".join(cleaned))
    stripped = "".join(
        char for char in decomposed if unicodedata.category(char)[0] != "M"
    ).lower()
    return "".join(char if char.isascii() else " " for char in stripped)


def _pre_tokenize(text: str) -> Iterable[str]:
    """Split on whitespace and ASCII punctuation, like BertPreTokenizer."""
    word: list[str] = []
    for char in text:
        if char.isspace() or _ascii_punctuation(char):
            if word:
                yield "".join(word)
                word.clear()
            if _ascii_punctuation(char):
                yield char
        else:
            word.append(char)
    if word:
        yield "".join(word)


class WordPieceTokenizer:
    """Greedy longest-match-first WordPiece tokenizer (ASCII vocabulary)."""

    def __init__(
        self,
        tokens: Sequence[str],
        unk_token: str = "[UNK]",
        max_input_chars_per_word: int = 100,
    ) -> None:
        self.tokens = tuple(tokens)
        self.vocab = {token: index for index, token in enumerate(self.tokens)}
        self.unk_token = unk_token
        self.unk_id = self.vocab[unk_token]
        self.max_input_chars_per_word = max_input_chars_per_word

    @classmethod
    def from_packaged(cls, data_dir: str | Path | None = None) -> "WordPieceTokenizer":
        return cls(_asset("vocab.json", data_dir)["tokens"])

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for word in _pre_tokenize(_normalize(text)):
            ids.extend(self._encode_word(word))
        return ids

    def tokenize(self, text: str) -> list[str]:
        return [self.tokens[index] for index in self.encode(text)]

    def _encode_word(self, word: str) -> list[int]:
        if len(word) > self.max_input_chars_per_word:
            return [self.unk_id]
        chars = list(word)
        ids: list[int] = []
        start = 0
        while start < len(chars):
            end = len(chars)
            while end > start:
                token = "".join(chars[start:end])
                if start:
                    token = "##" + token
                index = self.vocab.get(token)
                if index is not None:
                    ids.append(index)
                    break
                end -= 1
            else:
                start += 1
                continue
            start = end
        return ids if ids else [self.unk_id]


class StaticEmbedding:
    """Token lookup followed by mean pooling, with numpy doing the vector math."""

    def __init__(
        self,
        tokens: Sequence[str],
        weights: np.ndarray,
        scales: np.ndarray,
        dim: int = 32,
    ) -> None:
        # Weights are stored as raw bytes; reinterpret them as signed int8.
        weights = (np.asarray(weights, dtype=np.int16).reshape(-1, dim) & 0xFF)
        weights = weights.astype(np.uint8).view(np.int8)
        scales = np.asarray(scales, dtype=np.float32).reshape(-1, 1)
        if len(weights) != len(tokens) or len(scales) != len(tokens):
            raise ValueError("weight/scale tables do not match the vocabulary")
        self.tokenizer = WordPieceTokenizer(tokens)
        self.tokens = self.tokenizer.tokens
        self.dim = dim
        self._vectors = weights.astype(np.float32) * scales

    @classmethod
    def from_packaged(cls, data_dir: str | Path | None = None) -> "StaticEmbedding":
        """Load from the bundled tarball or from a directory of raw JSON assets."""
        tokens = _asset("vocab.json", data_dir)["tokens"]
        weights_asset = _asset("weights.json", data_dir)
        return cls(
            tokens,
            weights_asset["weights"],
            weights_asset["scales"],
            dim=int(weights_asset.get("dim", 32)),
        )

    def encode_one(self, text: str, *, normalize: bool = False) -> list[float]:
        """Encode a single string into a list of ``dim`` floats."""
        ids = self.tokenizer.encode(text)
        totals = np.zeros(self.dim, dtype=np.float32)
        for index in ids:
            totals += self._vectors[index]
        if ids:
            totals /= np.float32(len(ids))
        if normalize:
            norm = float(np.linalg.norm(totals.astype(np.float64)))
            if norm:
                totals = totals / norm
        return totals.tolist()

    def encode(
        self, texts: str | Iterable[str], *, normalize: bool = False
    ) -> list[float] | list[list[float]]:
        """Encode one string or an iterable of strings."""
        if isinstance(texts, str):
            return self.encode_one(texts, normalize=normalize)
        return [self.encode_one(text, normalize=normalize) for text in texts]


_MODEL: StaticEmbedding | None = None


def load_model() -> StaticEmbedding:
    """Load and cache the bundled model."""
    global _MODEL
    if _MODEL is None:
        _MODEL = StaticEmbedding.from_packaged()
    return _MODEL


def encode(
    texts: str | Iterable[str], *, normalize: bool = False
) -> list[float] | list[list[float]]:
    """Encode a string, or a batch of strings, into 32-dimensional vectors."""
    return load_model().encode(texts, normalize=normalize)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="text to embed")
    parser.add_argument(
        "--normalize", action="store_true", help="return a unit-length vector"
    )
    args = parser.parse_args()
    print(json.dumps(encode(args.text, normalize=args.normalize)))


if __name__ == "__main__":
    main()
