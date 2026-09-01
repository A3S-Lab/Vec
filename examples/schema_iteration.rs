use a3s_vec::{
    initialize, shutdown, AlterColumnOption, Collection, CollectionSchema, DataType, Doc,
    FieldSchema, IndexParams, MetricType,
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

fn document(id: &str, title: &str, embedding: [f32; 2]) -> a3s_vec::Result<Doc> {
    let mut doc = Doc::with_pk(id)?;
    doc.add_string("id", id)?;
    doc.add_string("title", title)?;
    doc.add_vector_f32("embedding", &embedding)?;
    Ok(doc)
}

fn main() -> a3s_vec::Result<()> {
    initialize(None)?;
    let directory = ExampleDirectory::new("a3s-vec-schema-iteration");
    let data_path = directory.path().to_string_lossy();
    let schema = CollectionSchema::builder("schema_iteration")
        .add_field(FieldSchema::new("id", DataType::String, false, 0)?)
        .add_field(FieldSchema::new("title", DataType::String, false, 0)?)
        .add_vector_field(
            "embedding",
            DataType::VectorFp32,
            2,
            IndexParams::flat(MetricType::L2)?,
        )
        .build()?;
    let collection = Collection::create_and_open(&data_path, &schema, None)?;

    let docs = [
        document("doc-1", "One", [1.0, 0.0])?,
        document("doc-2", "Two", [0.0, 1.0])?,
        document("doc-3", "Three", [1.0, 1.0])?,
    ];
    let inserted = collection.insert(&docs.iter().collect::<Vec<_>>())?;
    assert_eq!(inserted.success_count, 3);

    let iterator = collection.iter_with_options(Some(&["title"]), false)?;
    let iterator_revision = iterator.revision();
    let late = document("doc-4", "Four", [0.5, 0.5])?;
    assert_eq!(collection.insert(&[&late])?.success_count, 1);
    let snapshot_docs = iterator.collect::<a3s_vec::Result<Vec<_>>>()?;
    assert_eq!(snapshot_docs.len(), 3);
    assert!(snapshot_docs
        .iter()
        .all(|doc| doc.get_string("title").is_ok_and(|value| value.is_some())));
    assert!(snapshot_docs
        .iter()
        .all(|doc| doc.vector("embedding").is_none()));
    assert!(collection.stats()?.revision > iterator_revision);

    let category = FieldSchema::new("category", DataType::String, false, 0)?;
    collection.add_column(&category, Some("'general'"))?;
    collection.rename_column("category", "kind")?;
    let nullable_kind = FieldSchema::new("kind", DataType::String, true, 0)?;
    collection.alter_column(&nullable_kind, AlterColumnOption::default())?;
    let evolved = collection.fetch(&["doc-1"])?;
    assert_eq!(evolved[0].get_string("kind")?, Some("general".to_string()));
    collection.drop_column("kind")?;
    collection.flush()?;
    collection.close()?;

    let reopened = Collection::open(&data_path, None)?;
    assert_eq!(reopened.count()?, 4);
    assert!(!reopened.schema()?.has_field("kind"));
    reopened.close()?;
    shutdown()?;
    println!("Iterator isolation and schema evolution workflows passed.");
    Ok(())
}
