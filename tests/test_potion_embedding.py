"""Tests for potion_embedding.py: ``uv run --with numpy python -m unittest discover -s tests``."""

from __future__ import annotations

import json
import math
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np

from potion_embedding import StaticEmbedding, WordPieceTokenizer, encode, load_model

ROOT = Path(__file__).resolve().parents[1]

HELLO_WORLD = [
    -0.06458018,
    -0.15089768,
    -0.27086204,
    0.15954897,
    0.03685451,
    0.26745671,
    0.08301116,
    -0.23883809,
    0.18415618,
    0.17872564,
    -0.13338013,
    0.04583888,
    -0.0600642,
    0.02837727,
    0.12276495,
    0.25639531,
    0.22441089,
    0.18676485,
    -0.02269646,
    -0.115517,
    0.26406193,
    -0.03858538,
    0.06903323,
    -0.1102808,
    -0.22781704,
    0.13314919,
    -0.0445487,
    0.05234626,
    -0.15821479,
    -0.02542822,
    0.18443377,
    -0.06647159,
    0.15869765,
    0.04327974,
    0.04295943,
    0.089316,
    0.06454378,
    0.13101783,
    0.09101013,
    0.04526148,
    -0.08371773,
    -0.12598702,
    -0.01366845,
    -0.05346945,
    -0.13809437,
    -0.00401836,
    0.03196536,
    -0.04304823,
    -0.03502447,
    0.02014134,
    -0.04377087,
    0.02862394,
    0.00273166,
    0.02887256,
    0.06940684,
    0.18640453,
    -0.0161008,
    0.0815006,
    0.04166317,
    0.19700463,
    0.02292891,
    -0.06871781,
    0.06768266,
    0.14915158,
]


class TokenizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer: WordPieceTokenizer = load_model().tokenizer

    def test_wordpiece_tokens(self) -> None:
        self.assertEqual(
            self.tokenizer.tokenize("The weather is lovely today."),
            ["the", "weather", "is", "lovely", "today", "."],
        )
        self.assertEqual(self.tokenizer.tokenize("abc123"), ["abc", "##12", "##3"])

    def test_uncased_accents_and_cjk(self) -> None:
        self.assertEqual(self.tokenizer.tokenize("naïve café"), ["naive", "cafe"])
        self.assertEqual(self.tokenizer.tokenize("HELLO"), ["hello"])
        self.assertEqual(self.tokenizer.tokenize("日本語"), ["日", "本", "語"])

    def test_special_tokens_and_unknowns(self) -> None:
        self.assertEqual(
            self.tokenizer.tokenize("a[CLS]b [UNK] 🎈"),
            ["a", "[CLS]", "b", "[UNK]", "[UNK]"],
        )
        self.assertEqual(self.tokenizer.tokenize("ab\u0378cd"), ["[UNK]"])

    def test_empty_input(self) -> None:
        self.assertEqual(self.tokenizer.encode(""), [])


class ModelTests(unittest.TestCase):
    def test_known_vector(self) -> None:
        vector = encode("hello world")
        self.assertIsInstance(vector, list)
        self.assertEqual(len(vector), 64)
        for actual, expected in zip(vector, HELLO_WORLD):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_batch_and_shape(self) -> None:
        vectors = encode(["hello world", "goodbye world"])
        self.assertEqual(len(vectors), 2)
        self.assertTrue(all(len(vector) == 64 for vector in vectors))
        self.assertEqual(encode(""), [0.0] * 64)
        self.assertEqual(encode("😀"), [0.0] * 64)

    def test_normalization(self) -> None:
        vector = encode("hello world")
        norm = math.sqrt(math.fsum(value * value for value in vector))
        self.assertAlmostEqual(norm, 1.0, places=6)
        raw = encode("hello world", normalize=False)
        self.assertGreater(math.sqrt(math.fsum(v * v for v in raw)), 1.0)
        self.assertEqual(encode("", normalize=False), [0.0] * 64)

    def test_similarity_with_numpy(self) -> None:
        vector = np.asarray(encode("Python is a programming language."))
        self.assertAlmostEqual(float(np.dot(vector, vector)), 1.0, places=6)
        self.assertAlmostEqual(float(np.dot([1.0, 0.0], [0.0, 1.0])), 0.0)
        self.assertAlmostEqual(float(np.dot(np.zeros(64), vector)), 0.0)

    def test_word_order_does_not_matter(self) -> None:
        first = encode("sushi rice seafood")
        second = encode("seafood sushi rice")
        for actual, expected in zip(first, second):
            self.assertAlmostEqual(actual, expected, places=6)


class DataFormatTests(unittest.TestCase):
    TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "##s"]
    WEIGHTS = [
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ]

    def test_default_archive_members(self) -> None:
        with tarfile.open(ROOT / "data" / "potion.tar.gz", "r:gz") as tar:
            self.assertEqual(
                sorted(tar.getnames()), ["config.json", "vocab.json", "weights.bin"]
            )

    def test_custom_raw_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            vectors = np.asarray(self.WEIGHTS, dtype="<f4")
            (directory / "config.json").write_text(
                json.dumps({"dim": 2, "normalize": False})
            )
            (directory / "vocab.json").write_text(json.dumps({"tokens": self.TOKENS}))
            (directory / "weights.bin").write_bytes(vectors.tobytes())

            model = StaticEmbedding.from_packaged(directory)
            self.assertEqual(model.dim, 2)
            self.assertFalse(model.normalize)
            self.assertEqual(model.encode("hello world"), [2.0, 3.0])
            self.assertAlmostEqual(
                model.encode("hello", normalize=True)[0], 1 / math.sqrt(5), places=6
            )
            self.assertEqual(
                WordPieceTokenizer.from_packaged(directory).tokenize("hellos"),
                ["hello", "##s"],
            )


if __name__ == "__main__":
    unittest.main()
