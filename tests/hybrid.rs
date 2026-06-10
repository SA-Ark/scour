//! End-to-end test of the full hybrid pipeline: chunk → index → fused
//! retrieval, on a small but meaningful corpus with hand-checkable
//! expectations.

use scour::{chunk_text, HybridIndex};

/// Toy embedding: bag-of-topic axes so the test is fully deterministic.
/// axis 0 = systems/rust, axis 1 = cooking, axis 2 = astronomy.
fn embed(text: &str) -> Vec<f32> {
    let lower = text.to_lowercase();
    let score =
        |words: &[&str]| -> f32 { words.iter().filter(|w| lower.contains(**w)).count() as f32 };
    let v = [
        score(&["rust", "memory", "compiler", "systems", "borrow"]),
        score(&["cook", "sauce", "pasta", "flavor", "boil"]),
        score(&["star", "galaxy", "telescope", "orbit", "cosmic"]),
    ];
    let norm = (v.iter().map(|x| x * x).sum::<f32>()).sqrt().max(1e-6);
    v.iter().map(|x| x / norm).collect()
}

const CORPUS: [(&str, &str); 6] = [
    (
        "rust-borrow",
        "The Rust borrow checker enforces memory safety at compile time, \
         eliminating whole classes of systems bugs.",
    ),
    (
        "rust-async",
        "Async Rust schedules cooperative tasks onto worker threads; the \
         compiler turns futures into state machines.",
    ),
    (
        "pasta",
        "Cooking pasta well means salted boiling water and finishing the \
         noodles in the sauce for flavor.",
    ),
    (
        "ragu",
        "A slow-simmered ragu builds flavor over hours; the sauce should \
         barely bubble while it cooks.",
    ),
    (
        "galaxies",
        "Telescopes resolve distant galaxies whose light has traveled for \
         billions of years across cosmic distances.",
    ),
    (
        "orbits",
        "Stable orbits arise when gravitational pull balances tangential \
         velocity around a star.",
    ),
];

fn build_index() -> HybridIndex {
    let mut index = HybridIndex::new(3);
    for (id, text) in CORPUS {
        index.add(id, text, &embed(text));
    }
    index
}

#[test]
fn hybrid_retrieval_finds_topical_documents() {
    let index = build_index();

    let query = "rust memory safety";
    let results = index.search(query, &embed(query), 3);
    assert_eq!(results[0].0, "rust-borrow");

    let query = "simmering a flavorful sauce";
    let results = index.search(query, &embed(query), 3);
    assert!(results[0].0 == "ragu" || results[0].0 == "pasta");

    let query = "light from distant galaxies";
    let results = index.search(query, &embed(query), 3);
    assert_eq!(results[0].0, "galaxies");
}

#[test]
fn semantic_leg_rescues_vocabulary_mismatch() {
    let index = build_index();

    // Query with zero lexical overlap with the astronomy docs, but whose
    // embedding lights up the astronomy axis.
    let query = "telescope orbit cosmic";
    let lexical = index.search_lexical("celestial observation equipment", 3);
    let hybrid = index.search("celestial observation equipment", &embed(query), 3);

    // Lexical alone finds nothing; hybrid still returns astronomy docs.
    assert!(lexical.is_empty());
    assert!(!hybrid.is_empty());
    assert!(hybrid
        .iter()
        .any(|(id, _)| id == "galaxies" || id == "orbits"));
}

#[test]
fn chunked_document_round_trips_through_index() {
    let long_doc = CORPUS
        .iter()
        .map(|(_, text)| *text)
        .collect::<Vec<_>>()
        .join("\n\n");

    let chunks = chunk_text(&long_doc, 120);
    assert!(chunks.len() > 2);
    assert_eq!(chunks.concat(), long_doc);

    let mut index = HybridIndex::new(3);
    for (i, chunk) in chunks.iter().enumerate() {
        index.add(&format!("chunk-{i}"), chunk, &embed(chunk));
    }

    let query = "borrow checker memory safety";
    let results = index.search(query, &embed(query), 2);
    assert!(!results.is_empty());
    // The winning chunk must actually contain the borrow-checker sentence.
    let winner: usize = results[0]
        .0
        .strip_prefix("chunk-")
        .unwrap()
        .parse()
        .unwrap();
    assert!(chunks[winner].to_lowercase().contains("borrow"));
}
