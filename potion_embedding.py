#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""64-dimensional static embeddings from minishlab/potion-base-2M.

The upstream model is a Model2Vec / static embedding model: a BERT WordPiece
tokenizer followed by mean pooling and L2 normalization.  This module runs the
same computation with numpy alone; no ONNX runtime is needed.  Model data:
``data/potion.tar.gz`` next to this script.

Source: https://huggingface.co/minishlab/potion-base-2M/blob/main/onnx/model.onnx
"""

from __future__ import annotations

import argparse
import json
import tarfile
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

__all__ = [
    "StaticEmbedding",
    "WordPieceTokenizer",
    "encode",
    "load_model",
]

_DATA_FILE = Path(__file__).resolve().parent / "data" / "potion.tar.gz"


@lru_cache(maxsize=1)
def _archive() -> dict[str, bytes]:
    """Load the tarball's members once, keyed by member name."""
    with tarfile.open(_DATA_FILE, "r:gz") as tar:
        return {
            member.name: tar.extractfile(member).read()
            for member in tar.getmembers()
            if member.isfile()
        }


def _asset(name: str, data_dir: str | Path | None = None) -> bytes:
    """Read one asset from the bundled archive or a custom raw-data dir."""
    if data_dir is None:
        return _archive()[name]
    return (Path(data_dir) / name).read_bytes()


# --- BERT tokenizer ----------------------------------------------------------


def _is_chinese_char(char: str) -> bool:
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B920 <= code <= 0x2CEAF
        or 0xF900 <= code <= 0xFAFF
        or 0x2F800 <= code <= 0x2FA1F
    )


def _normalize(text: str) -> str:
    """BertNormalizer: clean, split CJK chars, strip accents, lowercase."""
    cleaned: list[str] = []
    for char in text:
        if char == "\0" or char == "\ufffd":
            continue
        category = unicodedata.category(char)
        if char not in "\t\n\r" and category in {"Cc", "Cf", "Cs", "Co"}:
            # BERT's clean_text drops control/format/private-use characters,
            # but leaves unassigned (Cn) code points for WordPiece to reject.
            continue
        cleaned.append(" " if char.isspace() else char)

    spaced: list[str] = []
    for char in cleaned:
        if _is_chinese_char(char):
            spaced.extend((" ", char, " "))
        else:
            spaced.append(char)

    decomposed = unicodedata.normalize("NFD", "".join(spaced))
    stripped = "".join(
        char for char in decomposed if unicodedata.category(char) != "Mn"
    )
    return stripped.lower()


def _is_punctuation(char: str) -> bool:
    code = ord(char)
    return (
        33 <= code <= 47
        or 58 <= code <= 64
        or 91 <= code <= 96
        or 123 <= code <= 126
        or unicodedata.category(char)[0] == "P"
    )


def _pre_tokenize(text: str) -> Iterator[str]:
    """BertPreTokenizer: drop whitespace, isolate each punctuation char."""
    word: list[str] = []
    for char in text:
        if char.isspace() or _is_punctuation(char):
            if word:
                yield "".join(word)
                word.clear()
            if _is_punctuation(char):
                yield char
        else:
            word.append(char)
    if word:
        yield "".join(word)


class WordPieceTokenizer:
    """BERT-style WordPiece tokenizer matching the upstream tokenizer.json."""

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
        # Added tokens are matched verbatim against the raw text, before
        # normalization.  This model registers the five standard BERT ones.
        self.special_tokens = tuple(
            token
            for token in ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")
            if token in self.vocab
        )

    @classmethod
    def from_packaged(cls, data_dir: str | Path | None = None) -> "WordPieceTokenizer":
        tokens = json.loads(_asset("vocab.json", data_dir))["tokens"]
        return cls(tokens)

    def tokenize(self, text: str) -> list[str]:
        return [self.tokens[index] for index in self.encode(text)]

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for part, is_special in self._split_specials(text):
            if is_special:
                ids.append(self.vocab[part])
            else:
                for word in _pre_tokenize(_normalize(part)):
                    ids.extend(self._encode_word(word))
        return ids

    def _split_specials(self, text: str) -> Iterator[tuple[str, bool]]:
        start = 0
        pos = 0
        while pos < len(text):
            match = next(
                (token for token in self.special_tokens if text.startswith(token, pos)),
                None,
            )
            if match is None:
                pos += 1
                continue
            if pos > start:
                yield text[start:pos], False
            yield match, True
            pos += len(match)
            start = pos
        if start < len(text):
            yield text[start:], False

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
                # WordPiece emits [UNK] for the whole word if any character
                # cannot be matched, rather than skipping the bad character.
                return [self.unk_id]
            start = end
        return ids if ids else [self.unk_id]


# --- Static model ------------------------------------------------------------


class StaticEmbedding:
    """Token lookup, mean pooling, and L2 normalization with numpy."""

    def __init__(
        self,
        tokens: Sequence[str],
        weights: np.ndarray,
        dim: int = 64,
        normalize: bool = True,
    ) -> None:
        self.tokenizer = WordPieceTokenizer(tokens)
        self.tokens = self.tokenizer.tokens
        self.dim = dim
        self.normalize = normalize
        self.vectors = np.ascontiguousarray(weights, dtype=np.float32).reshape(
            len(self.tokens), dim
        )
        if self.vectors.shape[0] != len(tokens):
            raise ValueError("weight table does not match the vocabulary")

    @classmethod
    def from_packaged(cls, data_dir: str | Path | None = None) -> "StaticEmbedding":
        """Load from the bundled tarball or from a directory of raw assets."""
        config = json.loads(_asset("config.json", data_dir))
        tokens = json.loads(_asset("vocab.json", data_dir))["tokens"]
        raw = _asset("weights.bin", data_dir)
        return cls(
            tokens,
            np.frombuffer(raw, dtype="<f4"),
            dim=int(config["dim"]),
            normalize=bool(config.get("normalize", True)),
        )

    def encode_one(self, text: str, *, normalize: bool | None = None) -> list[float]:
        """Encode a single string into a list of ``dim`` floats."""
        if normalize is None:
            normalize = self.normalize
        ids = [
            index
            for index in self.tokenizer.encode(text)
            if index != self.tokenizer.unk_id
        ]
        if not ids:
            return [0.0] * self.dim
        vector = self.vectors[ids].mean(axis=0, dtype=np.float32)
        if normalize:
            norm = np.sqrt(np.sum(vector * vector, dtype=np.float32))
            if norm:
                vector = vector / max(np.float32(norm), np.float32(1e-12))
        return vector.tolist()

    def encode(
        self, texts: str | Iterable[str], *, normalize: bool | None = None
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
    texts: str | Iterable[str], *, normalize: bool | None = None
) -> list[float] | list[list[float]]:
    """Encode a string, or a batch of strings, into 64-dimensional vectors."""
    return load_model().encode(texts, normalize=normalize)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="text to embed")
    parser.add_argument(
        "--no-normalize", action="store_true", help="return the raw mean vector"
    )
    args = parser.parse_args()
    print(json.dumps(encode(args.text, normalize=not args.no_normalize)))


if __name__ == "__main__":
    main()
