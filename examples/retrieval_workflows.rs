use a3s_vec::{
    initialize, shutdown, Collection, CollectionSchema, DataType, Doc, FieldSchema, Fts,
    IndexParams, MetricType, MultiQuery, SearchQuery, SubQuery,
};
use std::path::{Path, PathBuf};

struct ExampleDirectory(PathBuf);

impl ExampleDirectory {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!("{name}-{}", std::process::id()));
        if path.exists() {
            std::fs::remove_dir_all(&path).expect("stale example directory must be removable");
        }
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for ExampleDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[allow(clippy::too_many_lines)]
fn main() -> a3s_vec::Result<()> {
    initialize(None)?;
    let directory = ExampleDirectory::new("a3s-vec-retrieval-workflows");
    let data_path = directory.path().to_string_lossy();

    let mut category = FieldSchema::new("category", DataType::String, false, 0)?;
    category.set_index_params(&IndexParams::invert(true, true)?)?;
    let mut body = FieldSchema::new("body", DataType::String, false, 0)?;
    body.set_index_params(&IndexParams::fts(
        Some("standard"),
        Some(&["lowercase"]),
        None,
    )?)?;
    let schema = CollectionSchema::builder("retrieval_workflows")
        .add_field(FieldSchema::new("id", DataType::String, false, 0)?)
        .add_field(category)
        .add_field(body)
        .add_vector_field(
            "embedding",
            DataType::VectorFp32,
            3,
            IndexParams::hnsw(MetricType::Cosine, 8, 64)?,
        )
        .build()?;
    let collection = Collection::create_and_open(&data_path, &schema, None)?;

    let fixtures = [
        (
            "rust-index",
            "systems",
            "Rust builds a safe in-process vector index",
            [1.0, 0.0, 0.0],
        ),
        (
            "database",
            "systems",
            "A durable database stores vectors and text",
            [0.9, 0.1, 0.0],
        ),
        (
            "search",
            "retrieval",
            "Full text search uses BM25 ranking",
            [0.1, 0.9, 0.0],
        ),
        (
            "agent",
            "retrieval",
            "Coding agents retrieve workspace context",
            [0.0, 1.0, 0.1],
        ),
    ];
    let mut docs = Vec::new();
    for (id, category, body, embedding) in fixtures {
        let mut doc = Doc::with_pk(id)?;
        doc.add_string("id", id)?;
        doc.add_string("category", category)?;
        doc.add_string("body", body)?;
        doc.add_vector_f32("embedding", &embedding)?;
        docs.push(doc);
    }
    let inserted = collection.insert(&docs.iter().collect::<Vec<_>>())?;
    assert_eq!(inserted.success_count, 4);
    assert_eq!(inserted.error_count, 0);

    let mut fts = Fts::new()?;
    fts.set_query_string("rust OR database")?;
    let lexical = SearchQuery::fts("body", &fts, 3)?;
    let lexical_hits = collection.query(&lexical)?;
    assert_eq!(lexical_hits.len(), 2);
    assert!(lexical_hits
        .iter()
        .any(|doc| doc.get_pk() == Some("rust-index")));
    assert!(lexical_hits
        .iter()
        .any(|doc| doc.get_pk() == Some("database")));

    let vector = SearchQuery::new("embedding", &[1.0, 0.0, 0.0], 2)?;
    let vector_hits = collection.query(&vector)?;
    assert_eq!(vector_hits[0].get_pk(), Some("rust-index"));

    let mut vector_branch = SubQuery::new()?;
    vector_branch.set_field_name("embedding")?;
    vector_branch.set_query_vector(&[1.0, 0.0, 0.0])?;
    vector_branch.set_num_candidates(4)?;

    let mut lexical_branch = SubQuery::new()?;
    lexical_branch.set_field_name("body")?;
    lexical_branch.set_fts(&fts)?;
    lexical_branch.set_num_candidates(4)?;

    let mut hybrid = MultiQuery::new()?;
    hybrid.set_topk(3)?;
    hybrid.add_sub_query(&vector_branch)?;
    hybrid.add_sub_query(&lexical_branch)?;
    hybrid.set_rerank_rrf(60)?;
    let rrf_hits = collection.multi_query(&hybrid)?;
    assert!(rrf_hits
        .iter()
        .any(|doc| doc.get_pk() == Some("rust-index")));
    assert!(rrf_hits.iter().any(|doc| doc.get_pk() == Some("database")));

    hybrid.set_rerank_weighted(&[1.0, 0.0])?;
    hybrid.set_normalization("minmax")?;
    let weighted_hits = collection.multi_query(&hybrid)?;
    assert_eq!(weighted_hits[0].get_pk(), Some("rust-index"));

    collection.close()?;
    shutdown()?;
    println!("FTS, vector, and hybrid retrieval workflows passed.");
    Ok(())
}
