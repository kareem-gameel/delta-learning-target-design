# Contributing

Open an issue before changing scientific defaults or data contracts. Method
changes require focused numerical tests and must preserve molecule-ID
alignment. Never infer correspondence from CSV, sparse-matrix, LMDB, or extxyz
row order.

Before submitting a change, run:

```bash
make verify
```

Do not commit raw third-party datasets, LMDB files, checkpoints, embeddings,
prediction arrays, credentials, cluster logs, or local caches.
