//! Reproducible micro-benchmark for the README table.
//!
//! Run with: `cargo run --release --example bench`
//!
//! Synthetic but realistic workload: 384-dim embeddings (the common
//! sentence-transformer width) and ~100-word documents.

use scour::{chunk_text, Bm25Index, HnswIndex, HybridIndex};
use std::time::Instant;

fn pseudo_random_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let mut values = Vec::with_capacity(dim);
    for _ in 0..dim {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        values.push((((x >> 40) as u32) as f32 / ((1u32 << 24) - 1) as f32) * 2.0 - 1.0);
    }
    values
}

const WORDS: [&str; 32] = [
    "system",
    "vector",
    "search",
    "index",
    "memory",
    "retrieval",
    "ranking",
    "query",
    "token",
    "stream",
    "buffer",
    "shard",
    "merge",
    "graph",
    "node",
    "edge",
    "weight",
    "score",
    "fusion",
    "lexical",
    "semantic",
    "neural",
    "embed",
    "cluster",
    "cache",
    "latency",
    "throughput",
    "pipeline",
    "batch",
    "filter",
    "recall",
    "precision",
];

/// ~16k-term vocabulary (32 stems x 512 numeric variants) with a skew:
/// half the tokens come from the small common-stem pool, half from the
/// long tail — a rough approximation of natural term frequency.
fn pseudo_random_doc(seed: u64, words: usize) -> String {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(7);
    let mut doc = String::new();
    for i in 0..words {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        let stem = WORDS[(x >> 33) as usize % WORDS.len()];
        if x & 1 == 0 {
            doc.push_str(stem);
        } else {
            let variant = (x >> 17) % 512;
            doc.push_str(stem);
            doc.push_str(&variant.to_string());
        }
        doc.push(if i % 12 == 11 { '.' } else { ' ' });
    }
    doc
}

fn main() {
    let dim = 384;
    let n = 20_000usize;
    let queries = 1_000usize;

    println!("scour bench — {n} docs, {dim}-dim embeddings, {queries} queries\n");

    // ---- HNSW ----
    let entries: Vec<(String, Vec<f32>)> = (0..n)
        .map(|i| (format!("doc-{i}"), pseudo_random_vec(dim, i as u64 + 1)))
        .collect();

    let t = Instant::now();
    let hnsw = HnswIndex::build_from_embeddings(&entries);
    let build = t.elapsed();
    println!(
        "HNSW  build:  {:>8.2?}  ({:.0} inserts/s)",
        build,
        n as f64 / build.as_secs_f64()
    );

    let qvecs: Vec<Vec<f32>> = (0..queries)
        .map(|q| pseudo_random_vec(dim, 1_000_000 + q as u64))
        .collect();
    let t = Instant::now();
    let mut sink = 0usize;
    for q in &qvecs {
        sink += hnsw.search(q, 10).len();
    }
    let search = t.elapsed();
    println!(
        "HNSW  search: {:>8.2?} total, {:.0} µs/query (k=10)  [{sink}]",
        search,
        search.as_micros() as f64 / queries as f64
    );

    // ---- BM25 ----
    let docs: Vec<(String, String)> = (0..n)
        .map(|i| (format!("doc-{i}"), pseudo_random_doc(i as u64, 100)))
        .collect();

    let t = Instant::now();
    let bm25 = Bm25Index::build_from_documents(&docs);
    let build = t.elapsed();
    println!(
        "BM25  build:  {:>8.2?}  ({:.0} docs/s, ~100 words/doc)",
        build,
        n as f64 / build.as_secs_f64()
    );

    // Worst case: queries built from the densest terms (huge posting lists).
    let t = Instant::now();
    let mut sink = 0usize;
    for q in 0..queries {
        sink += bm25
            .search(&pseudo_random_doc(9_000_000 + q as u64, 4), 10)
            .len();
    }
    let search = t.elapsed();
    println!(
        "BM25  search (dense terms, worst case): {:>8.2?} total, {:.0} µs/query (k=10)  [{sink}]",
        search,
        search.as_micros() as f64 / queries as f64
    );

    // Typical case: selective terms from the vocabulary tail.
    let t = Instant::now();
    let mut sink = 0usize;
    for q in 0..queries {
        let mut x = (q as u64 + 77).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut query = String::new();
        for _ in 0..4 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
            query.push_str(WORDS[(x >> 33) as usize % WORDS.len()]);
            query.push_str(&((x >> 17) % 512).to_string());
            query.push(' ');
        }
        sink += bm25.search(&query, 10).len();
    }
    let search = t.elapsed();
    println!(
        "BM25  search (selective terms, typical): {:>7.2?} total, {:.0} µs/query (k=10)  [{sink}]",
        search,
        search.as_micros() as f64 / queries as f64
    );

    // ---- Hybrid ----
    let mut hybrid = HybridIndex::new(dim);
    for i in 0..n {
        hybrid.add(&docs[i].0, &docs[i].1, &entries[i].1);
    }
    let t = Instant::now();
    let mut sink = 0usize;
    for (q, qvec) in qvecs.iter().enumerate() {
        let text = pseudo_random_doc(9_000_000 + q as u64, 4);
        sink += hybrid.search(&text, qvec, 10).len();
    }
    let search = t.elapsed();
    println!(
        "HYBRID search: {:>7.2?} total, {:.0} µs/query (k=10, RRF)  [{sink}]",
        search,
        search.as_micros() as f64 / queries as f64
    );

    // ---- Chunking ----
    let long_text = pseudo_random_doc(42, 200_000); // ~1.3 MB
    let t = Instant::now();
    let chunks = chunk_text(&long_text, 2000);
    let elapsed = t.elapsed();
    println!(
        "CHUNK 1.3 MB → {} chunks in {:>6.2?} ({:.0} MB/s)",
        chunks.len(),
        elapsed,
        long_text.len() as f64 / 1e6 / elapsed.as_secs_f64()
    );
}
