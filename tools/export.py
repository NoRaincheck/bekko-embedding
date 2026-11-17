#!/usr/bin/env python3
"""Convert the bekko-static INT8 ONNX model into plain JSON assets.

This is a one-off build tool, not a runtime dependency.  It reads the two
files from the upstream Hugging Face repository and writes a ``data.tar.gz``
archive of raw JSON into ``data/``.  It uses only the standard library: ONNX
files are Protobuf messages and the two tensors this model needs are extracted
with a small generic Protobuf scanner below.

Usage:
    python tools/export.py \
        --onnx path/to/model_int8_dim32.onnx \
        --tokenizer path/to/tokenizer.json
"""

from __future__ import annotations

import argparse
import io
import json
import struct
import tarfile
from pathlib import Path
from typing import Iterator


# --- Minimal Protobuf reader -------------------------------------------------

def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")


def _fields(data: bytes) -> Iterator[tuple[int, int, int | bytes]]:
    """Yield (field_number, wire_type, value) for a Protobuf message."""
    pos = 0
    while pos < len(data):
        key, pos = _read_varint(data, pos)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _read_varint(data, pos)
        elif wire == 1:
            value, pos = data[pos : pos + 8], pos + 8
        elif wire == 2:
            length, pos = _read_varint(data, pos)
            value, pos = data[pos : pos + length], pos + length
        elif wire == 5:
            value, pos = data[pos : pos + 4], pos + 4
        else:
            raise ValueError(f"unsupported Protobuf wire type {wire}")
        yield field, wire, value


def _message_fields(data: bytes, field_number: int) -> Iterator[bytes]:
    for field, wire, value in _fields(data):
        if field == field_number and wire == 2 and isinstance(value, bytes):
            yield value


def _tensor_fields(data: bytes) -> tuple[str, bytes]:
    name = ""
    raw = b""
    for field, wire, value in _fields(data):
        if field == 8 and wire == 2:  # name
            name = value.decode("utf-8")
        elif field == 9 and wire == 2:  # raw_data
            raw = value
    return name, raw


def read_initializers(path: Path) -> dict[str, bytes]:
    """Return the raw tensor bytes of the ONNX graph's initializers."""
    model = path.read_bytes()
    graph = next(_message_fields(model, 7))
    tensors: dict[str, bytes] = {}
    for tensor in _message_fields(graph, 5):
        name, raw = _tensor_fields(tensor)
        if not raw:
            raise ValueError(f"tensor {name!r} has no raw_data")
        tensors[name] = raw
    return tensors


# --- Asset writing -----------------------------------------------------------

def select_tokens(vocab: dict[str, int], mode: str) -> list[tuple[str, int]]:
    """Select the tokens to keep, in original vocabulary order."""
    if mode == "ascii":
        return [(token, index) for token, index in vocab.items() if token.isascii()]
    if mode == "english":
        # The upstream vocab keeps a 30k-token English core at the front.  In
        # addition, every ASCII single character (and its "##" form) is kept so
        # that WordPiece can always fall back to characters without emitting
        # [UNK].  Single characters live at the front of the vocabulary too.
        kept = [(token, index) for token, index in vocab.items() if index < 30_005]
        core = {token for token, _ in kept}
        for token, index in vocab.items():
            body = token[2:] if token.startswith("##") else token
            if len(body) == 1 and body.isascii() and token not in core:
                kept.append((token, index))
                core.add(token)
        kept.sort(key=lambda item: item[1])
        return kept
    raise ValueError(f"unknown vocabulary mode {mode!r}")


def export(
    onnx_path: Path,
    tokenizer_path: Path,
    raw_dir: Path,
    mode: str,
) -> None:
    tensors = read_initializers(onnx_path)
    weights = tensors["embedding_int8"]
    scales = tensors["row_scales"]
    vocab = json.loads(tokenizer_path.read_text())["model"]["vocab"]
    dim = 32
    if len(weights) != len(vocab) * dim:
        raise ValueError("unexpected embedding table size")
    if len(scales) != len(vocab) * 4:
        raise ValueError("unexpected scale table size")

    selected = select_tokens(vocab, mode)
    tokens = [token for token, _ in selected]
    if any("\n" in token for token in tokens):
        raise ValueError("cannot store tokens containing newlines")

    rows = b"".join(weights[index * dim : (index + 1) * dim] for _, index in selected)
    row_scales = b"".join(scales[index * 4 : (index + 1) * 4] for _, index in selected)

    raw_dir.mkdir(parents=True, exist_ok=True)

    # Raw JSON assets, packed into one gzip tarball (the only data source).
    raw_assets = {
        "vocab.json": json.dumps(
            {"format": "bekko-stdlib-vocab-v1", "encoding": "raw", "tokens": tokens},
            separators=(",", ":"),
        )
        + "\n",
        "weights.json": json.dumps(
            {
                "format": "bekko-stdlib-weights-v1",
                "encoding": "raw",
                "dim": dim,
                "weights": [list(rows[i : i + dim]) for i in range(0, len(rows), dim)],
                "scales": list(struct.unpack(f"<{len(row_scales) // 4}f", row_scales)),
            },
            separators=(",", ":"),
        )
        + "\n",
    }
    with tarfile.open(raw_dir / "data.tar.gz", "w:gz") as tar:
        for name, text in raw_assets.items():
            payload = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            tar.addfile(info, io.BytesIO(payload))

    print(f"wrote {len(tokens)} tokens ({mode})")
    print(f"archive: {raw_dir / 'data.tar.gz'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path("data"))
    parser.add_argument("--vocab", choices=("ascii", "english"), default="ascii")
    args = parser.parse_args()
    export(args.onnx, args.tokenizer, args.raw_dir, args.vocab)


if __name__ == "__main__":
    main()
