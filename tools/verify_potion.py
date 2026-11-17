#!/usr/bin/env python3
"""Dev-only check of potion_embedding.py against the upstream ONNX graph.

Requires ``onnxruntime``, ``tokenizers`` and ``numpy`` (dev-only).  Download
``model.onnx`` and ``tokenizer.json`` from

    https://huggingface.co/minishlab/potion-base-2M

and run from the repository root:

    uv run --with numpy --with onnxruntime --with tokenizers \
        python tools/verify_potion.py --onnx model.onnx --tokenizer tokenizer.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer
except ImportError as exc:
    raise SystemExit(
        "this check needs dev-only packages: onnxruntime, tokenizers and numpy"
    ) from exc

from potion_embedding import WordPieceTokenizer, encode

SENTENCES = [
    "hello world",
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
    "What are the characteristics of sushi?",
    "A warm noodle soup served in broth with sliced toppings.",
    "A Japanese dish made with vinegared rice, often shaped with seafood.",
    "Python is a programming language known for its readability.",
    "Who designed the famous basilica in Barcelona?",
    "",
    "a_b",
    "#hashtag",
    "it'sso sunny",
    "  Hello   World  ",
    "123.456-789",
    "@user_name /path/to/file.html?x=1&y=2",
    "naïve café",
    "日本語",
    "中文测试",
    "🎈 party",
    "😀",
    "[CLS] hello [SEP]",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--tokenizer", required=True)
    args = parser.parse_args()

    reference = Tokenizer.from_file(args.tokenizer)
    ours = WordPieceTokenizer.from_packaged()
    for text in SENTENCES:
        expected_tokens = reference.encode(text, add_special_tokens=False).tokens
        if ours.tokenize(text) != expected_tokens:
            raise AssertionError(f"tokenization differs for {text!r}")

    encoded = reference.encode_batch(SENTENCES, add_special_tokens=False)
    ids: list[int] = []
    offsets: list[int] = []
    for item in encoded:
        offsets.append(len(ids))
        ids.extend(token_id for token_id in item.ids if token_id != 1)  # drop [UNK]

    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    expected = session.run(
        ["embeddings"],
        {
            "input_ids": np.asarray(ids, dtype=np.int64),
            "offsets": np.asarray(offsets, dtype=np.int64),
        },
    )[0]
    actual = np.asarray(encode(SENTENCES), dtype=np.float32)
    difference = float(np.abs(expected - actual).max())
    print(f"compared {len(SENTENCES)} sentences; max abs difference: {difference}")
    if difference > 1e-5:
        raise AssertionError("outputs differ from the upstream ONNX graph")


if __name__ == "__main__":
    main()
