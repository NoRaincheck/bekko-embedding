# bekko-embedding

A single [uv script](https://docs.astral.sh/uv/guides/scripts/) implementing the
32-dimensional INT8 model from
[`hotchpotch/bekko-static-embedding-v1`](https://huggingface.co/hotchpotch/bekko-static-embedding-v1):
a WordPiece tokenizer and mean-pooled static embeddings. Its only runtime
dependency is numpy (declared inline), and model data lives in
`data/data.tar.gz` next to the script.

## Usage

```bash
uv run bekko_embedding.py --normalize "hello world"
# or: ./bekko_embedding.py --normalize "hello world"
```

```python
from bekko_embedding import encode

vector = encode("What are the characteristics of sushi?", normalize=True)  # 32 floats
vectors = encode(["first text", "second text"])  # batch
```

`StaticEmbedding` and `WordPieceTokenizer` can also be imported directly; both
have `from_packaged()` constructors.

## Data

The tarball holds two raw JSON assets: `vocab.json` with `tokens`, and
`weights.json` with `dim`, `weights` and `scales`. `from_packaged("./my_data")`
loads the same files from any directory.

## Development

```bash
uv run --with numpy python -m unittest discover -s tests -v
```

`tools/export.py` rebuilds `data/data.tar.gz` from the upstream ONNX model and
tokenizer; `tools/verify_reference.py` compares outputs against the upstream
graph.

MIT. See `NOTICE` for upstream attribution.
