use a3s_vec::{
    Collection, CollectionOptions, CollectionSchema, DataType, Doc, Durability, FieldSchema, Fts,
    IndexParams, IndexType, MetricType, MultiQuery, SearchQuery, SubQuery,
};
use tempfile::tempdir;

const DOCUMENTS: usize = 256;

fn manual_options() -> CollectionOptions {
    let mut options = CollectionOptions::new().expect("collection options must be valid");
    options
        .set_durability(Durability::Manual)
        .expect("manual durability must be valid");
    options
}

fn indexed_field(
    name: &str,
    data_type: DataType,
    nullable: bool,
    range: bool,
    wildcard: bool,
) -> FieldSchema {
    let mut field =
        FieldSchema::new(name, data_type, nullable, 0).expect("field schema must be valid");
    field
        .set_index_params(
            &IndexParams::invert(range, wildcard).expect("inverted descriptor must be valid"),
        )
        .expect("inverted index must be supported for this field");
    field
}

fn schema(name: &str, indexed: bool, ann: bool) -> CollectionSchema {
    let bucket = if indexed {
        indexed_field("bucket", DataType::Int32, false, true, false)
    } else {
        FieldSchema::new("bucket", DataType::Int32, false, 0).expect("bucket schema must be valid")
    };
    let active = if indexed {
        indexed_field("active", DataType::Bool, false, false, false)
    } else {
        FieldSchema::new("active", DataType::Bool, false, 0).expect("active schema must be valid")
    };
    let label = if indexed {
        indexed_field("label", DataType::String, false, true, true)
    } else {
        FieldSchema::new("label", DataType::String, false, 0).expect("label schema must be valid")
    };
    let optional = if indexed {
        indexed_field("optional", DataType::String, true, false, false)
    } else {
        FieldSchema::new("optional", DataType::String, true, 0)
            .expect("optional schema must be valid")
    };
    let mut embedding = FieldSchema::new("embedding", DataType::VectorFp32, false, 2)
        .expect("embedding schema must be valid");
    if ann {
        embedding
            .set_index_params(
                &IndexParams::hnsw(MetricType::L2, 8, 32).expect("HNSW descriptor must be valid"),
            )
            .expect("HNSW index must be supported");
    }
    let mut body =
        FieldSchema::new("body", DataType::String, true, 0).expect("body schema must be valid");
    body.set_index_params(
        &IndexParams::fts(Some("whitespace"), None, None).expect("FTS descriptor must be valid"),
    )
    .expect("FTS index configuration must be supported");
    CollectionSchema::builder(name)
        .add_field(bucket)
        .add_field(active)
        .add_field(label)
        .add_field(optional)
        .add_field(
            FieldSchema::new("tags", DataType::ArrayString, false, 0)
                .expect("tags schema must be valid"),
        )
        .add_field(body)
        .add_field(embedding)
        .build()
        .expect("collection schema must be valid")
}

fn document(index: usize) -> Doc {
    let mut doc = Doc::with_pk(format!("doc-{index:04}")).expect("document must be valid");
    let bucket = i32::try_from(index % 16).expect("bucket fits i32");
    doc.add_i32("bucket", bucket).expect("bucket must be valid");
    doc.add_bool("active", index % 2 == 0)
        .expect("active must be valid");
    doc.add_string("label", &format!("src/module-{bucket}/file-{index:03}.rs"))
        .expect("label must be valid");
    if index % 5 == 0 {
        doc.set_field_value("optional", a3s_vec::FieldValue::Null)
            .expect("explicit null must be valid");
    } else if index % 7 != 0 {
        doc.add_string("optional", if index % 2 == 0 { "even" } else { "odd" })
            .expect("optional value must be valid");
    }
    let parity = if index % 2 == 0 { "even" } else { "odd" };
    doc.add_array_string("tags", &["workspace", parity])
        .expect("tags must be valid");
    if index % 11 != 0 {
        doc.add_string(
            "body",
            &format!("workspace search bucket-{bucket} {parity}"),
        )
        .expect("body must be valid");
    }
    let coordinate = f32::from(u16::try_from(index).expect("fixture index fits u16"));
    doc.add_vector_f32(
        "embedding",
        &[
            coordinate,
            f32::from(i16::try_from(bucket).expect("bucket fits i16")),
        ],
    )
    .expect("embedding must be valid");
    doc
}

fn insert_fixture(collection: &Collection) {
    let docs: Vec<Doc> = (0..DOCUMENTS).map(document).collect();
    let refs: Vec<&Doc> = docs.iter().collect();
    let result = collection
        .insert(&refs)
        .expect("fixture insert must succeed");
    assert_eq!(
        result.success_count,
        u64::try_from(DOCUMENTS).expect("document count fits u64")
    );
}

fn query(collection: &Collection, filter: &str) -> Vec<Doc> {
    let mut query = SearchQuery::new(
        "embedding",
        &[0.0, 0.0],
        i32::try_from(DOCUMENTS).expect("document count fits i32"),
    )
    .expect("query must be valid");
    query
        .params
        .insert("metric".into(), serde_json::json!("l2"));
    query.set_filter(filter).expect("filter must be valid");
    collection.query(&query).expect("query must succeed")
}

fn assert_same_results(indexed: &Collection, fallback: &Collection, filter: &str) {
    let indexed = query(indexed, filter);
    let fallback = query(fallback, filter);
    let indexed_values: Vec<_> = indexed
        .iter()
        .map(|doc| (doc.get_pk().unwrap_or_default(), doc.get_score()))
        .collect();
    let fallback_values: Vec<_> = fallback
        .iter()
        .map(|doc| (doc.get_pk().unwrap_or_default(), doc.get_score()))
        .collect();
    assert_eq!(indexed_values, fallback_values, "filter={filter}");
}

fn comparable_results(docs: &[Doc]) -> Vec<(&str, f32)> {
    docs.iter()
        .map(|doc| (doc.get_pk().unwrap_or_default(), doc.get_score()))
        .collect()
}

#[test]
fn bitmap_filters_match_the_scan_oracle_for_scalar_and_boolean_semantics() {
    let temporary = tempdir().expect("temporary directory must be available");
    let options = manual_options();
    let indexed = Collection::create(
        temporary
            .path()
            .join("indexed")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema("indexed", true, false),
        Some(&options),
    )
    .expect("indexed collection must be created");
    let fallback = Collection::create(
        temporary
            .path()
            .join("fallback")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema("fallback", false, false),
        Some(&options),
    )
    .expect("fallback collection must be created");
    insert_fixture(&indexed);
    insert_fixture(&fallback);

    let indexed_filters = [
        "bucket == 3",
        "bucket >= 12 and active == true",
        "label has_prefix 'src/module-2/' or bucket < 2",
        "not (bucket in [1, 3, 5])",
        "optional is_null",
        "label like 'src/%/file-0_7.rs'",
        "bucket != 'not-a-number'",
        "bucket == 4 and tags contain_all ['workspace', 'even']",
    ];
    for filter in indexed_filters {
        assert_same_results(&indexed, &fallback, filter);
    }

    // OR cannot safely use one indexed side as a prefilter when the other side
    // is unindexed, so this expression must fall back without changing results.
    assert_same_results(
        &indexed,
        &fallback,
        "bucket == 4 or tags contain_all ['workspace', 'odd']",
    );
    // Complementing a partial AND bitmap would be a false-negative filter.
    // The planner must therefore fall back when NOT wraps an unindexed term.
    assert_same_results(
        &indexed,
        &fallback,
        "not (bucket == 4 and tags contain_all ['workspace', 'odd'])",
    );

    let stats = indexed.stats_snapshot().expect("stats must be readable");
    assert_eq!(stats.scalar_index_query_count, 8);
    assert!(
        stats.candidates_scanned
            < u64::try_from(DOCUMENTS * indexed_filters.len()).expect("candidate bound fits u64")
    );
    let scalar_stats: Vec<_> = stats
        .indexes
        .iter()
        .filter(|index| index.index_type == IndexType::Invert)
        .collect();
    assert_eq!(scalar_stats.len(), 4);
    assert!(scalar_stats
        .iter()
        .all(|index| index.source_revision == stats.revision));
}

#[test]
fn scalar_index_lifecycle_tracks_mutations_delete_by_filter_and_reopen() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection_path = temporary.path().join("lifecycle");
    let options = manual_options();
    let collection = Collection::create(
        collection_path
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema("lifecycle", true, false),
        Some(&options),
    )
    .expect("collection must be created");
    insert_fixture(&collection);

    let mut patch = Doc::with_pk("doc-0003").expect("patch must be valid");
    patch.add_i32("bucket", 9).expect("bucket must be valid");
    collection.update(&[&patch]).expect("update must succeed");
    collection
        .delete(&["doc-0019"])
        .expect("delete must succeed");
    let mut inserted = document(255);
    inserted.set_pk("doc-0999");
    inserted
        .set_field_value("bucket", a3s_vec::FieldValue::Int32(3))
        .expect("bucket replacement must be valid");
    collection
        .upsert(&[&inserted])
        .expect("upsert must succeed");

    let bucket_three = query(&collection, "bucket == 3");
    assert!(!bucket_three
        .iter()
        .any(|doc| doc.get_pk() == Some("doc-0003")));
    assert!(!bucket_three
        .iter()
        .any(|doc| doc.get_pk() == Some("doc-0019")));
    assert!(bucket_three
        .iter()
        .any(|doc| doc.get_pk() == Some("doc-0999")));

    collection
        .delete_by_filter("bucket >= 14 and active == true")
        .expect("indexed delete-by-filter must succeed");
    assert!(query(&collection, "bucket >= 14 and active == true").is_empty());
    collection.flush().expect("collection must flush");
    collection.close().expect("collection must close");

    let reopened = Collection::open(
        collection_path
            .to_str()
            .expect("temporary path must be UTF-8"),
        Some(&options),
    )
    .expect("collection must reopen");
    let reopened_three = query(&reopened, "bucket == 3");
    assert_eq!(
        reopened_three
            .iter()
            .filter(|doc| doc.get_pk() == Some("doc-0999"))
            .count(),
        1
    );

    reopened
        .drop_index("bucket")
        .expect("scalar index must be dropped");
    let before = reopened.stats_snapshot().expect("stats must be readable");
    let _ = query(&reopened, "bucket == 3");
    let after = reopened.stats_snapshot().expect("stats must be readable");
    assert_eq!(
        after.scalar_index_query_count,
        before.scalar_index_query_count
    );
    reopened
        .create_index(
            "bucket",
            &IndexParams::invert(true, false).expect("descriptor must be valid"),
        )
        .expect("scalar index must be recreated");
    let _ = query(&reopened, "bucket == 3");
    let recreated = reopened.stats_snapshot().expect("stats must be readable");
    assert_eq!(
        recreated.scalar_index_query_count,
        after.scalar_index_query_count + 1
    );
}

#[test]
fn retired_scalar_ordinals_compact_without_changing_filter_results() {
    const RETIRED: usize = 80;

    let temporary = tempdir().expect("temporary directory must be available");
    let options = manual_options();
    let indexed = Collection::create(
        temporary
            .path()
            .join("ordinal-indexed")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema("ordinal-indexed", true, false),
        Some(&options),
    )
    .expect("indexed collection must be created");
    let fallback = Collection::create(
        temporary
            .path()
            .join("ordinal-fallback")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema("ordinal-fallback", false, false),
        Some(&options),
    )
    .expect("fallback collection must be created");
    insert_fixture(&indexed);
    insert_fixture(&fallback);

    let retired: Vec<String> = (0..RETIRED)
        .map(|index| format!("doc-{index:04}"))
        .collect();
    let retired_refs: Vec<&str> = retired.iter().map(String::as_str).collect();
    indexed
        .delete(&retired_refs)
        .expect("indexed documents must be retired");
    fallback
        .delete(&retired_refs)
        .expect("fallback documents must be retired");

    let replacements: Vec<Doc> = (0..RETIRED).map(|index| document(1_000 + index)).collect();
    let replacement_refs: Vec<&Doc> = replacements.iter().collect();
    indexed
        .insert(&replacement_refs)
        .expect("indexed replacements must be inserted");
    fallback
        .insert(&replacement_refs)
        .expect("fallback replacements must be inserted");

    for filter in [
        "bucket == 3",
        "bucket >= 12 and active == true",
        "label has_prefix 'src/module-7/'",
        "optional is_null",
    ] {
        assert_same_results(&indexed, &fallback, filter);
    }
    let stats = indexed.stats().expect("statistics must be readable");
    let bucket = stats
        .indexes
        .iter()
        .find(|index| index.name == "bucket")
        .expect("bucket index stats must exist");
    assert_eq!(bucket.source_revision, stats.revision);
    assert_eq!(
        bucket.document_count,
        u64::try_from(DOCUMENTS).expect("document count fits u64")
    );
}

#[test]
fn selective_bitmap_filter_bypasses_ann_and_matches_exact_results() {
    let temporary = tempdir().expect("temporary directory must be available");
    let options = manual_options();
    let indexed = Collection::create(
        temporary
            .path()
            .join("ann-indexed")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema("ann-indexed", true, true),
        Some(&options),
    )
    .expect("indexed collection must be created");
    let fallback = Collection::create(
        temporary
            .path()
            .join("ann-fallback")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema("ann-fallback", false, false),
        Some(&options),
    )
    .expect("fallback collection must be created");
    insert_fixture(&indexed);
    insert_fixture(&fallback);

    assert_same_results(&indexed, &fallback, "bucket == 7 and active == false");
    let stats = indexed.stats_snapshot().expect("stats must be readable");
    assert_eq!(stats.scalar_index_query_count, 1);
    assert_eq!(stats.ann_query_count, 0);
    assert!(stats.candidates_scanned <= 16);
}

#[test]
fn bitmap_prefilters_apply_to_fts_and_multi_query_branches() {
    let temporary = tempdir().expect("temporary directory must be available");
    let options = manual_options();
    let indexed = Collection::create(
        temporary
            .path()
            .join("hybrid-indexed")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema("hybrid-indexed", true, false),
        Some(&options),
    )
    .expect("indexed collection must be created");
    let fallback = Collection::create(
        temporary
            .path()
            .join("hybrid-fallback")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema("hybrid-fallback", false, false),
        Some(&options),
    )
    .expect("fallback collection must be created");
    insert_fixture(&indexed);
    insert_fixture(&fallback);

    let mut fts = Fts::new().expect("FTS payload must be valid");
    fts.set_match_string("workspace search")
        .expect("FTS expression must be valid");
    let mut indexed_fts = SearchQuery::fts("body", &fts, 64).expect("FTS query must be valid");
    indexed_fts
        .set_filter("bucket == 3 and active == false")
        .expect("filter must be valid");
    let fallback_fts = indexed_fts.clone();
    let indexed_fts = indexed.query(&indexed_fts).expect("FTS query must succeed");
    let fallback_fts = fallback
        .query(&fallback_fts)
        .expect("fallback FTS query must succeed");
    assert_eq!(
        comparable_results(&indexed_fts),
        comparable_results(&fallback_fts)
    );

    let mut vector_branch = SubQuery::new().expect("vector branch must be valid");
    vector_branch
        .set_field_name("embedding")
        .expect("field must be valid");
    vector_branch
        .set_query_vector(&[0.0, 0.0])
        .expect("vector must be valid");
    vector_branch
        .set_num_candidates(64)
        .expect("candidate count must be valid");
    vector_branch
        .params
        .insert("metric".into(), serde_json::json!("l2"));
    let mut fts_branch = SubQuery::new().expect("FTS branch must be valid");
    fts_branch
        .set_field_name("body")
        .expect("field must be valid");
    fts_branch.set_fts(&fts).expect("FTS payload must be valid");
    fts_branch
        .set_num_candidates(64)
        .expect("candidate count must be valid");
    let mut multi = MultiQuery::new().expect("multi-query must be valid");
    multi
        .add_sub_query(&vector_branch)
        .expect("vector branch must be accepted");
    multi
        .add_sub_query(&fts_branch)
        .expect("FTS branch must be accepted");
    multi.set_topk(32).expect("topk must be valid");
    multi
        .set_filter("bucket == 3 and active == false")
        .expect("filter must be valid");
    multi
        .set_rerank_weighted(&[0.5, 0.5])
        .expect("weights must be valid");
    let indexed_multi = indexed
        .multi_query(&multi)
        .expect("multi-query must succeed");
    let fallback_multi = fallback
        .multi_query(&multi)
        .expect("fallback multi-query must succeed");
    assert_eq!(
        comparable_results(&indexed_multi),
        comparable_results(&fallback_multi)
    );

    let stats = indexed.stats_snapshot().expect("stats must be readable");
    assert_eq!(stats.scalar_index_query_count, 2);
    assert_eq!(stats.fts_query_count, 2);
    assert_eq!(stats.fts_index_query_count, 2);
}
