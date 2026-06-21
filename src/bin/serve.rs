//! Scour live demo server — a dependency-free HTTP server that indexes the
//! seeded corpus and serves three retrieval lanes (BM25, vector, RRF-fused)
//! for any typed query, plus a single-page UI to explore them side by side.
//!
//! Pure `std`: a tiny blocking HTTP/1.1 handler on a thread pool, a hand
//! rolled JSON writer, and an embedded HTML/CSS/JS front end. No web
//! framework, no async runtime, no template engine — same zero-dependency
//! discipline as the library it demonstrates.
//!
//! Run: `cargo run --release --bin serve` then open http://127.0.0.1:8087
//! Override the bind address with `SCOUR_ADDR` (e.g. `0.0.0.0:8087`).

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::thread;

use scour::corpus::{Doc, CORPUS, EXAMPLE_QUERIES};
use scour::{embed, Bm25Index, HnswIndex, DEFAULT_RRF_K, DEMO_DIM};

const UI_HTML: &str = include_str!("../../assets/index.html");

/// Everything the request handlers need, built once at startup.
struct AppState {
    bm25: Bm25Index,
    hnsw: HnswIndex,
    docs: HashMap<String, &'static Doc>,
}

impl AppState {
    fn build() -> Self {
        let mut bm25 = Bm25Index::new();
        let mut hnsw = HnswIndex::new(DEMO_DIM);
        let mut docs = HashMap::new();
        for doc in CORPUS {
            let (id, title, body) = *doc;
            // Index title + body together so titles contribute to recall.
            let full = format!("{title}. {body}");
            bm25.add_document(id, &full);
            hnsw.insert(id, &embed(&full, DEMO_DIM));
            docs.insert(id.to_string(), doc);
        }
        AppState { bm25, hnsw, docs }
    }

    fn doc(&self, id: &str) -> Option<&'static Doc> {
        self.docs.get(id).copied()
    }
}

fn main() {
    // Resolve the bind address. `SCOUR_ADDR` wins (full host:port for local
    // use); otherwise bind 0.0.0.0:$PORT for deploy behind a reverse proxy;
    // default to localhost:8087 for a bare `cargo run`.
    let addr = std::env::var("SCOUR_ADDR").unwrap_or_else(|_| {
        let port = std::env::var("PORT").unwrap_or_else(|_| "8087".to_string());
        format!("0.0.0.0:{port}")
    });
    let state = Arc::new(AppState::build());
    eprintln!(
        "scour demo: indexed {} documents (BM25 + HNSW {}-dim), listening on http://{addr}",
        CORPUS.len(),
        DEMO_DIM
    );

    let listener = TcpListener::bind(&addr).unwrap_or_else(|e| {
        eprintln!("failed to bind {addr}: {e}");
        std::process::exit(1);
    });

    let pool = ThreadPool::new(8);
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let state = Arc::clone(&state);
                pool.execute(move || handle(stream, state));
            }
            Err(e) => eprintln!("connection error: {e}"),
        }
    }
}

fn handle(mut stream: TcpStream, state: Arc<AppState>) {
    let mut reader = BufReader::new(match stream.try_clone() {
        Ok(s) => s,
        Err(_) => return,
    });

    // Request line: METHOD PATH VERSION
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() || request_line.is_empty() {
        return;
    }
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let path = parts.next().unwrap_or("/");

    // Drain headers (we don't need bodies; the API is GET with a query string).
    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).is_err() {
            break;
        }
        if line == "\r\n" || line == "\n" || line.is_empty() {
            break;
        }
        if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
            content_length = v.trim().parse().unwrap_or(0);
        }
    }
    if content_length > 0 {
        let mut body = vec![0u8; content_length.min(1 << 20)];
        let _ = reader.read_exact(&mut body);
    }

    let (route, query) = split_path(path);
    let response = match (method, route) {
        ("GET", "/") | ("GET", "/index.html") => {
            http_response("200 OK", "text/html; charset=utf-8", UI_HTML.as_bytes())
        }
        ("GET", "/health") => http_response("200 OK", "application/json", b"{\"status\":\"ok\"}"),
        ("GET", "/api/examples") => {
            http_response("200 OK", "application/json", examples_json().as_bytes())
        }
        ("GET", "/api/search") => {
            let q = query_param(query, "q").unwrap_or_default();
            let k: usize = query_param(query, "k")
                .and_then(|v| v.parse().ok())
                .unwrap_or(8)
                .clamp(1, 25);
            let json = search_json(&state, &q, k);
            http_response("200 OK", "application/json; charset=utf-8", json.as_bytes())
        }
        _ => http_response("404 Not Found", "application/json", b"{\"error\":\"not found\"}"),
    };

    let _ = stream.write_all(&response);
    let _ = stream.flush();
}

// ---------------------------------------------------------------------------
// Search → JSON
// ---------------------------------------------------------------------------

fn search_json(state: &AppState, raw_q: &str, k: usize) -> String {
    let q = raw_q.trim();
    if q.is_empty() {
        return format!(
            "{{\"query\":\"\",\"k\":{k},\"count\":{},\"lexical\":[],\"semantic\":[],\"fused\":[]}}",
            CORPUS.len()
        );
    }

    let fetch = k.saturating_mul(3).max(k);

    // Lexical leg: BM25.
    let lexical = state.bm25.search(q, fetch);

    // Semantic leg: embed the query, search HNSW (cosine distance: smaller is
    // closer). Convert to a 0..1 similarity for display.
    let qvec = embed(q, DEMO_DIM);
    let semantic = state.hnsw.search(&qvec, fetch);

    // Fused leg: RRF over the two rank orders.
    let lex_ids: Vec<String> = lexical.iter().map(|(id, _)| id.clone()).collect();
    let sem_ids: Vec<String> = semantic.iter().map(|(id, _)| id.clone()).collect();
    let fused = scour::rrf_fuse(&[lex_ids.clone(), sem_ids.clone()], DEFAULT_RRF_K);

    // Rank lookups so the fused lane can show *why* a doc ranked (which legs
    // contributed and at what rank).
    let lex_rank: HashMap<&str, usize> =
        lex_ids.iter().enumerate().map(|(i, id)| (id.as_str(), i + 1)).collect();
    let sem_rank: HashMap<&str, usize> =
        sem_ids.iter().enumerate().map(|(i, id)| (id.as_str(), i + 1)).collect();

    let mut out = String::with_capacity(4096);
    out.push('{');
    out.push_str(&format!("\"query\":{},", json_str(q)));
    out.push_str(&format!("\"k\":{k},"));
    out.push_str(&format!("\"count\":{},", CORPUS.len()));

    // lexical lane
    out.push_str("\"lexical\":[");
    for (i, (id, score)) in lexical.iter().take(k).enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&lane_item_json(state, id, *score, None));
    }
    out.push_str("],");

    // semantic lane (distance → similarity)
    out.push_str("\"semantic\":[");
    for (i, (id, dist)) in semantic.iter().take(k).enumerate() {
        if i > 0 {
            out.push(',');
        }
        let sim = (1.0 - *dist as f64).clamp(0.0, 1.0);
        out.push_str(&lane_item_json(state, id, sim, None));
    }
    out.push_str("],");

    // fused lane (with provenance: which legs and at what rank)
    out.push_str("\"fused\":[");
    for (i, (id, score)) in fused.iter().take(k).enumerate() {
        if i > 0 {
            out.push(',');
        }
        let prov = Provenance {
            lex: lex_rank.get(id.as_str()).copied(),
            sem: sem_rank.get(id.as_str()).copied(),
        };
        out.push_str(&lane_item_json(state, id, *score, Some(prov)));
    }
    out.push(']');

    out.push('}');
    out
}

struct Provenance {
    lex: Option<usize>,
    sem: Option<usize>,
}

fn lane_item_json(state: &AppState, id: &str, score: f64, prov: Option<Provenance>) -> String {
    let (title, body) = match state.doc(id) {
        Some((_, t, b)) => (*t, *b),
        None => ("(unknown)", ""),
    };
    let mut s = String::new();
    s.push('{');
    s.push_str(&format!("\"id\":{},", json_str(id)));
    s.push_str(&format!("\"title\":{},", json_str(title)));
    s.push_str(&format!("\"body\":{},", json_str(body)));
    s.push_str(&format!("\"score\":{:.4}", score));
    if let Some(p) = prov {
        s.push_str(",\"from\":{");
        s.push_str(&format!("\"lexical\":{},", opt_num(p.lex)));
        s.push_str(&format!("\"semantic\":{}", opt_num(p.sem)));
        s.push('}');
    }
    s.push('}');
    s
}

fn opt_num(v: Option<usize>) -> String {
    match v {
        Some(n) => n.to_string(),
        None => "null".to_string(),
    }
}

fn examples_json() -> String {
    let mut s = String::from("{\"examples\":[");
    for (i, q) in EXAMPLE_QUERIES.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&json_str(q));
    }
    s.push_str("]}");
    s
}

// ---------------------------------------------------------------------------
// Minimal HTTP + JSON + URL helpers (std only)
// ---------------------------------------------------------------------------

fn http_response(status: &str, content_type: &str, body: &[u8]) -> Vec<u8> {
    let header = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\
         Cache-Control: no-cache\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let mut out = header.into_bytes();
    out.extend_from_slice(body);
    out
}

/// Split "/api/search?q=foo&k=8" → ("/api/search", "q=foo&k=8").
fn split_path(path: &str) -> (&str, &str) {
    match path.split_once('?') {
        Some((route, query)) => (route, query),
        None => (path, ""),
    }
}

/// Extract and URL-decode a query parameter.
fn query_param(query: &str, key: &str) -> Option<String> {
    for pair in query.split('&') {
        if let Some((k, v)) = pair.split_once('=') {
            if k == key {
                return Some(url_decode(v));
            }
        }
    }
    None
}

fn url_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                let hi = hex_val(bytes[i + 1]);
                let lo = hex_val(bytes[i + 2]);
                if let (Some(h), Some(l)) = (hi, lo) {
                    out.push(h << 4 | l);
                    i += 3;
                } else {
                    out.push(bytes[i]);
                    i += 1;
                }
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Serialize a string as a JSON string literal (escaping the required chars).
fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

// ---------------------------------------------------------------------------
// Tiny fixed-size thread pool (std only)
// ---------------------------------------------------------------------------

struct ThreadPool {
    sender: mpsc::Sender<Box<dyn FnOnce() + Send + 'static>>,
}

impl ThreadPool {
    fn new(size: usize) -> Self {
        let (sender, receiver) = mpsc::channel::<Box<dyn FnOnce() + Send + 'static>>();
        let receiver = Arc::new(Mutex::new(receiver));
        for _ in 0..size.max(1) {
            let receiver: Arc<Mutex<Receiver<_>>> = Arc::clone(&receiver);
            thread::spawn(move || loop {
                let job = {
                    let guard = match receiver.lock() {
                        Ok(g) => g,
                        Err(_) => break,
                    };
                    guard.recv()
                };
                match job {
                    Ok(job) => job(),
                    Err(_) => break,
                }
            });
        }
        ThreadPool { sender }
    }

    fn execute<F: FnOnce() + Send + 'static>(&self, f: F) {
        let _ = self.sender.send(Box::new(f));
    }
}
