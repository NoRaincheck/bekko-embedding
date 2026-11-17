"""Tests for the single uv script: ``uv run --with numpy python -m unittest discover -s tests``."""

from __future__ import annotations

import json
import math
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bekko_embedding import StaticEmbedding, WordPieceTokenizer, encode, load_model

ROOT = Path(__file__).resolve().parents[1]

HELLO_WORLD = [
    -27.942177,
    6.835876,
    -13.188484,
    9.262549,
    14.0844,
    3.472441,
    6.04503,
    0.765748,
    11.015256,
    -17.485483,
    9.632874,
    25.131889,
    -3.834154,
    -20.661419,
    -2.002214,
    -10.636319,
    19.054626,
    16.002708,
    -9.908218,
    7.70128,
    3.059793,
    -28.421507,
    -8.321851,
    -7.160679,
    -22.934303,
    12.896654,
    -0.124754,
    4.362205,
    -20.244833,
    -18.275593,
    3.421507,
    -3.255167,
]


class TokenizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer: WordPieceTokenizer = load_model().tokenizer

    def test_wordpiece_tokens(self) -> None:
        self.assertEqual(
            self.tokenizer.tokenize("The weather is lovely today."),
            ["the", "weather", "is", "lovely", "today", "."],
        )
        self.assertEqual(self.tokenizer.tokenize("abc123"), ["abc", "##123"])

    def test_uncased_and_accents(self) -> None:
        self.assertEqual(self.tokenizer.tokenize("naïve café"), ["naive", "cafe"])
        self.assertEqual(self.tokenizer.tokenize("HELLO"), ["hello"])

    def test_non_ascii_tokens_are_stripped(self) -> None:
        self.assertEqual(self.tokenizer.tokenize("日本語"), [])
        self.assertEqual(self.tokenizer.tokenize("hello\ue000world"), ["hello", "world"])

    def test_empty_input(self) -> None:
        self.assertEqual(self.tokenizer.encode(""), [])


class ModelTests(unittest.TestCase):
    def test_known_vector(self) -> None:
        vector = encode("hello world")
        self.assertIsInstance(vector, list)
        self.assertEqual(len(vector), 32)
        for actual, expected in zip(vector, HELLO_WORLD):
            self.assertAlmostEqual(actual, expected, places=5)

    def test_batch_and_shape(self) -> None:
        vectors = encode(["hello world", "goodbye world"])
        self.assertEqual(len(vectors), 2)
        self.assertTrue(all(len(vector) == 32 for vector in vectors))
        self.assertEqual(encode(""), [0.0] * 32)

    def test_normalization(self) -> None:
        vector = encode("hello world", normalize=True)
        norm = math.sqrt(math.fsum(value * value for value in vector))
        self.assertAlmostEqual(norm, 1.0, places=6)

    def test_similarity_with_numpy(self) -> None:
        vector = np.asarray(encode("Python is a programming language.", normalize=True))
        self.assertAlmostEqual(float(np.dot(vector, vector)), 1.0, places=6)
        self.assertAlmostEqual(float(np.dot([1.0, 0.0], [0.0, 1.0])), 0.0)
        self.assertAlmostEqual(float(np.dot(np.zeros(32), vector)), 0.0)

    def test_word_order_does_not_matter(self) -> None:
        first = encode("sushi rice seafood")
        second = encode("seafood sushi rice")
        # Float32 accumulation order changes the last few bits.
        for actual, expected in zip(first, second):
            self.assertAlmostEqual(actual, expected, places=5)


class DataFormatTests(unittest.TestCase):
    TOKENS = ["[UNK]", "hello", "world", "##s"]
    WEIGHTS = [1, -2, 3, 4, -5, 6, 7, 8]
    SCALES = [0.25, 0.5, 0.75, 1.0]

    def test_default_archive_is_a_gzip_tarball(self) -> None:
        with tarfile.open(ROOT / "data" / "data.tar.gz", "r:gz") as tar:
            self.assertEqual(sorted(tar.getnames()), ["vocab.json", "weights.json"])

    def test_custom_raw_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            vocab = {"encoding": "raw", "tokens": self.TOKENS}
            weights = {
                "encoding": "raw",
                "dim": 2,
                "weights": [self.WEIGHTS[i : i + 2] for i in range(0, 8, 2)],
                "scales": self.SCALES,
            }
            (directory / "vocab.json").write_text(json.dumps(vocab))
            (directory / "weights.json").write_text(json.dumps(weights))

            model = StaticEmbedding.from_packaged(directory)
            self.assertEqual(model.encode("hello world"), [-1.125, 3.25])
            self.assertEqual(
                WordPieceTokenizer.from_packaged(directory).tokenize("hellos"),
                ["hello", "##s"],
            )


if __name__ == "__main__":
    unittest.main()
