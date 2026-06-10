# Scour

[![CI](https://github.com/SA-Ark/scour/actions/workflows/ci.yml/badge.svg)](https://github.com/SA-Ark/scour/actions/workflows/ci.yml)
![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![dependencies: 0](https://img.shields.io/badge/dependencies-0-brightgreen.svg)

**Embeddable hybrid search primitives for Rust — BM25, HNSW, reciprocal-rank fusion, and UTF-8-safe
chunking. Zero dependencies.**

Scour is the in-process retrieval core extracted from a production semantic-memory engine that serves
68,000+ documents. It gives you the four primitives every RAG / search pipeline needs, in plain Rust with
no dependency tree at all — embed it in a CLI, a server, a WASM target, or a test harness without dragging
in tokio, an ANN library, and a tokenizer framework.

```rust
use scour::HybridIndex;

let mut index = HybridIndex::new(384); // your embedding width

index.add("doc-1", "the rust borrow checker enforces memory safety", &embedding_1);
index.add("doc-2", "slow-simmered ragu builds flavor over hours",   &embedding_2);

// lexical + semantic legs, fused with Reciprocal Rank Fusion
let hits = index.search("memory safety in rust", &query_embedding, 10);
```

## Architecture

```
                              HybridIndex
        ┌──────────────────────────┴──────────────────────────┐
        │                                                     │
   Bm25Index                                             HnswIndex
   ┌─────────────────────────┐                ┌──────────────────────────────┐
   │ inverted index           │                │ hierarchical NSW graph       │
   │ Okapi BM25 (k1=1.2,b=.75)│                │ cosine distance              │
   │                          │                │ soft deletes + revive        │
   │   text::tokenize         │                │ deterministic construction   │
   │   ├─ lowercase/split     │                │ tunable M / ef params        │
   │   ├─ stopword filter     │                └──────────────┬───────────────┘
   │   └─ Porter stemmer      │                               │
   └────────────┬─────────────┘                               │
                │       ranked lexical        ranked semantic │
                └──────────────┐      ┌───────────────────────┘
                               ▼      ▼
                        fuse::rrf_fuse  (1 / (k + rank), k = 60)
                               │
                               ▼
                      fused, deterministic top-k

   chunk::chunk_text — boundary-aware (¶ → sentence → line), UTF-8-safe,
   lossless: chunks always reassemble to the original text byte-for-byte
```

## What's in the box

| Module | What it is |
|---|---|
| `bm25` | Okapi BM25 over an in-memory inverted index; deterministic tie-breaks; add/remove/replace documents |
| `hnsw` | Pure-Rust HNSW ANN index: cosine distance, soft deletes with revive-on-reinsert, reproducible builds (per-index PRNG), tunable `HnswParams` |
| `fuse` | Reciprocal Rank Fusion (Cormack et al., 2009) — merge any number of ranked lists without score normalization; generic over id type |
| `text` | Unicode-aware tokenizer + English stopwords + a full five-step Porter stemmer |
| `chunk` | Chunking for embedding pipelines: paragraph/sentence/line boundary preference, never splits a UTF-8 code point, optional overlap, lossless round-trip |
| `HybridIndex` | All of it behind one type: `add`, `remove`, `search` (fused), `search_lexical`, `search_semantic` |

## Benchmarks

`cargo run --release --example bench` — 20,000 docs, 384-dim embeddings, 1,000 queries, single thread,
Intel i7-13700H. Reproducible: the workload is seeded.

| Operation | Result |
|---|---|
| HNSW build (20K × 384-dim, ef_construction=200) | ~46 s (≈ 430 inserts/s) |
| HNSW search, k=10 | **~1.0 ms/query** |
| HNSW recall@10 vs exact scan | **≥ 90%** (enforced by a unit test, not just claimed) |
| BM25 build (~100-word docs) | ≈ 20,000 docs/s |
| BM25 search, selective terms (typical) | **70 µs/query** |
| BM25 search, dense terms (worst case: every query term in ~half the corpus) | 10.4 ms/query |
| Hybrid (BM25 + HNSW + RRF), k=10 | dominated by the legs above |
| Chunking | ≈ 460 MB/s |

Worst-case BM25 numbers are listed deliberately: posting-list density is the cost driver, and a benchmark
that hides the dense case isn't a benchmark.

## Design decisions

- **Zero dependencies.** The entire crate is `std`. This is not minimalism theater — it makes the crate
  auditable in one sitting, portable to WASM, and immune to dependency churn.
- **Determinism everywhere.** BM25 ties break lexicographically; HNSW level assignment uses a per-index
  seeded PRNG (same insertion order → same graph → same results); RRF ties break by first appearance.
  Deterministic retrieval is the difference between a debuggable pipeline and a haunted one.
- **Soft deletes in HNSW.** Deleting from a navigable small-world graph properly requires re-linking;
  soft-delete + filter at query time is the production-pragmatic answer, and `insert` revives a deleted
  id in place.
- **Bring your own embeddings.** Any `&[f32]` works. Scour deliberately does not pick an embedding model,
  an HTTP client, or a runtime for you.
- **Lossless chunking.** `chunk_text(t, n).concat() == t` is a tested invariant — chunkers that silently
  drop or duplicate bytes corrupt RAG corpora in ways that surface weeks later.

## Quickstart

```toml
[dependencies]
scour = { git = "https://github.com/SA-Ark/scour" } # crates.io publish pending
```

```rust
use scour::{chunk_text, Bm25Index, HnswIndex, HybridIndex, rrf_fuse};

// 1. chunk a document for your embedding pipeline
let chunks = chunk_text(&long_document, 2000);

// 2. or use the pieces standalone
let mut bm25 = Bm25Index::new();
bm25.add_document("a", "fearless concurrency in rust");
let lexical_hits = bm25.search("rust concurrency", 5);

let mut ann = HnswIndex::new(384);
ann.insert("a", &embedding);
let semantic_hits = ann.search(&query_embedding, 5);

// 3. or fuse arbitrary ranked lists from anywhere
let fused = rrf_fuse(&[list_a, list_b, list_c], scour::DEFAULT_RRF_K);
```

Run the test suite and benchmark:

```bash
cargo test
cargo run --release --example bench
```

## Provenance

Extracted from the retrieval core of a production memory system (68K+ documents, hybrid
semantic + keyword recall). Hardened during extraction: the chunker gained UTF-8 code-point safety
(the original could split a multibyte character), HNSW construction became deterministic per index,
and every module carries its tests.

## License

MIT — see [LICENSE](LICENSE).
