use a3s_vec::{
    initialize, shutdown, Collection, CollectionSchema, DataType, Doc, FieldSchema,
    GroupBySearchQuery, IndexParams, MetricType,
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

fn main() -> a3s_vec::Result<()> {
    initialize(None)?;
    let directory = ExampleDirectory::new("a3s-vec-group-by");
    let data_path = directory.path().to_string_lossy();
    let schema = CollectionSchema::builder("group_by")
        .add_field(FieldSchema::new("category", DataType::String, false, 0)?)
        .add_field(FieldSchema::new("title", DataType::String, false, 0)?)
        .add_vector_field(
            "embedding",
            DataType::VectorFp32,
            2,
            IndexParams::hnsw(MetricType::Cosine, 8, 64)?,
        )
        .build()?;
    let collection = Collection::create_and_open(&data_path, &schema, None)?;

    let fixtures = [
        ("alpha-1", "alpha", "Alpha one", [1.0, 0.0]),
        ("alpha-2", "alpha", "Alpha two", [0.98, 0.02]),
        ("beta-1", "beta", "Beta one", [0.9, 0.1]),
        ("beta-2", "beta", "Beta two", [0.85, 0.15]),
        ("gamma-1", "gamma", "Gamma one", [0.0, 1.0]),
        ("gamma-2", "gamma", "Gamma two", [0.1, 0.9]),
    ];
    let mut docs = Vec::new();
    for (id, category, title, embedding) in fixtures {
        let mut doc = Doc::with_pk(id)?;
        doc.add_string("category", category)?;
        doc.add_string("title", title)?;
        doc.add_vector_f32("embedding", &embedding)?;
        docs.push(doc);
    }
    let inserted = collection.insert(&docs.iter().collect::<Vec<_>>())?;
    assert_eq!(inserted.success_count, 6);

    let mut query = GroupBySearchQuery::new("embedding", "category", &[1.0, 0.0], 2, 2)?;
    query.set_output_fields(&["category", "title"])?;
    let groups = collection.group_by_query(&query)?;
    assert_eq!(groups.len(), 2);
    assert_eq!(groups["alpha"].len(), 2);
    assert_eq!(groups["beta"].len(), 2);
    assert!(groups
        .values()
        .flatten()
        .all(|doc| doc.vector("embedding").is_none()));

    collection.close()?;
    shutdown()?;
    println!("Group-by top-k workflow passed.");
    Ok(())
}
