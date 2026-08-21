# scour

[![Crates.io](https://img.shields.io/crates/v/scour-search.svg)](https://crates.io/crates/scour-search)
[![Docs.rs](https://img.shields.io/docsrs/scour-search)](https://docs.rs/scour-search)
[![CI](https://github.com/SA-Ark/scour/actions/workflows/ci.yml/badge.svg)](https://github.com/SA-Ark/scour/actions/workflows/ci.yml)
[![Downloads](https://img.shields.io/crates/d/scour-search.svg)](https://crates.io/crates/scour-search)
[![License](https://img.shields.io/crates/l/scour-search.svg)](LICENSE)

scour is a zero-dependency hybrid search engine for Rust, combining BM25 keyword search and HNSW vector search with reciprocal-rank fusion.

Drop it into a CLI, a server, a WASM target, or a test harness. There's no tokio, no ANN library, no tokenizer framework underneath — the whole crate is `std`, so you get the four primitives every RAG or search pipeline ends up needing without dragging in a dependency tree to get them.

```rust
use scour::HybridIndex;

let mut index = HybridIndex::new(384); // your embedding width

index.add("doc-1", "the rust borrow checker enforces memory safety", &embedding_1);
index.add("doc-2", "slow-simmered ragu builds flavor over hours",   &embedding_2);

// lexical + semantic legs, fused with Reciprocal Rank Fusion
let hits = index.search("memory safety in rust", &query_embedding, 10);
```

## Features

- **BM25 keyword search** — Okapi BM25 over an in-memory inverted index, deterministic tie-breaks, add / remove / replace documents.
- **HNSW vector search** — pure-Rust approximate nearest neighbors: cosine distance, soft deletes with revive-on-reinsert, reproducible per-index builds, tunable `M` / `ef`.
- **Reciprocal-rank fusion** — merge any number of ranked lists without score normalization; generic over id type.
- **UTF-8-safe chunking** — paragraph → sentence → line boundary preference, never splits a code point, lossless round-trip (`chunk_text(t, n).concat() == t` is a tested invariant).
- **Zero dependencies** — audit it in one sitting, compile it to WASM, and never get woken up by someone else's dependency churn.
- **Deterministic** — same inputs, same graph, same results. BM25 ties break lexicographically, HNSW level assignment uses a per-index seeded PRNG, RRF ties break by first appearance.

## Benchmarks

`cargo run --release --example bench` — 20,000 docs, 384-dim embeddings, 1,000 queries, single thread, Intel i7-13700H. The workload is seeded, so the numbers reproduce.

| Operation | Result |
|---|---|
| HNSW build (20K × 384-dim, ef_construction=200) | ~45 s (≈ 447 inserts/s) |
| HNSW search, k=10 | **~0.9 ms/query** |
| HNSW recall@10 vs exact scan | **≥ 90%** (enforced by a unit test, not just claimed) |
| BM25 build (~100-word docs) | ≈ 22,000 docs/s |
| BM25 search, selective terms (typical) | **106 µs/query** |
| BM25 search, dense terms (worst case: every query term in ~half the corpus) | 10.5 ms/query |
| Hybrid (BM25 + HNSW + RRF), k=10 | dominated by the legs above |
| Chunking | ≈ 497 MB/s |

The worst-case BM25 row is there on purpose. Posting-list density drives the cost, and a benchmark that quietly drops the dense case isn't telling you anything.

## Installation

```toml
[dependencies]
scour-search = "0.1"
```

Or `cargo add scour-search`. Try the three lanes side by side on a live corpus at **[scour.chakrakali.com](https://scour.chakrakali.com)** — one query, ranked by BM25, by vectors, and by RRF, each hit tagged with which leg found it and at what rank.

## Usage

Reach for the whole `HybridIndex`, or use any piece standalone:

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

The vector lane takes any `&[f32]` — scour won't pick an embedding model, an HTTP client, or a runtime for you.

Run the demo, the tests, or the benchmark locally:

```bash
cargo run --release --bin scour-demo   # then open http://127.0.0.1:8087
cargo test
cargo run --release --example bench
```

## How it works

A `HybridIndex` wraps two indexes over the same documents. `Bm25Index` is an inverted index scoring Okapi BM25 (k1=1.2, b=0.75) with a Unicode-aware tokenizer, English stopwords, and a full five-step Porter stemmer. `HnswIndex` is a hierarchical navigable small-world graph over cosine distance; deletes are soft (mark-and-filter, revived on reinsert) because truly removing a node means re-linking its neighbours. A search runs both legs, then `rrf_fuse` merges the ranked lists with reciprocal-rank fusion (`1 / (k + rank)`, k=60), which needs no score normalization and recovers the best of keyword precision and semantic recall.

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

| Module | What it is |
|---|---|
| `bm25` | Okapi BM25 over an in-memory inverted index; deterministic tie-breaks; add / remove / replace documents |
| `hnsw` | Pure-Rust HNSW ANN index: cosine distance, soft deletes with revive-on-reinsert, reproducible builds (per-index PRNG), tunable `HnswParams` |
| `fuse` | Reciprocal Rank Fusion (Cormack et al., 2009) — merge any number of ranked lists without score normalization; generic over id type |
| `text` | Unicode-aware tokenizer + English stopwords + a full five-step Porter stemmer |
| `chunk` | Chunking for embedding pipelines: paragraph/sentence/line boundary preference, never splits a UTF-8 code point, optional overlap, lossless round-trip |
| `HybridIndex` | All of it behind one type: `add`, `remove`, `search` (fused), `search_lexical`, `search_semantic` |

## License

MIT — see [LICENSE](LICENSE).
