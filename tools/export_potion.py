#!/usr/bin/env python3
"""Convert minishlab/potion-base-2M's ONNX model into plain assets.

This is a one-off build tool, not a runtime dependency.  It reads the ONNX
file and tokenizer linked below from the Hugging Face repository and writes
``data/potion.tar.gz``.  The ONNX initializer is Protobuf, so the small generic
scanner in ``export.py`` is reused to pull out the float32 embedding table.

    https://huggingface.co/minishlab/potion-base-2M/blob/main/onnx/model.onnx

Usage:
    python tools/export_potion.py --onnx path/to/model.onnx \
        --tokenizer path/to/tokenizer.json
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

if __package__:
    from .export import read_initializers
else:
    from export import read_initializers


SOURCE = "https://huggingface.co/minishlab/potion-base-2M"


def export(onnx_path: Path, tokenizer_path: Path, data_dir: Path) -> None:
    tensors = read_initializers(onnx_path)
    weights = tensors["embedding_bag.weight"]
    vocab = json.loads(tokenizer_path.read_text(encoding="utf-8"))["model"]["vocab"]
    tokens = [token for token, _ in sorted(vocab.items(), key=lambda item: item[1])]

    dim = 64
    if len(weights) != len(tokens) * dim * 4:
        raise ValueError(
            f"expected {len(tokens) * dim * 4} weight bytes, got {len(weights)}"
        )

    data_dir.mkdir(parents=True, exist_ok=True)
    assets = {
        "config.json": json.dumps(
            {
                "format": "potion-stdlib-v1",
                "source": SOURCE,
                "dim": dim,
                "normalize": True,
            },
            separators=(",", ":"),
        )
        + "\n",
        "vocab.json": json.dumps(
            {"format": "potion-stdlib-vocab-v1", "tokens": tokens},
            separators=(",", ":"),
        )
        + "\n",
    }
    with tarfile.open(data_dir / "potion.tar.gz", "w:gz") as tar:
        for name, text in assets.items():
            payload = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            tar.addfile(info, io.BytesIO(payload))
        info = tarfile.TarInfo("weights.bin")
        info.size = len(weights)
        info.mtime = 0
        tar.addfile(info, io.BytesIO(weights))

    print(f"wrote {len(tokens)} tokens x {dim} floats")
    print(f"archive: {data_dir / 'potion.tar.gz'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    export(args.onnx, args.tokenizer, args.data_dir)


if __name__ == "__main__":
    main()
