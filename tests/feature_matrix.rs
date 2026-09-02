//! End-to-end feature matrix for the advertised `a3s-vec` surface.
//!
//! The focused tests in the other integration modules remain the detailed
//! regression fixtures.  This file intentionally walks every public workflow
//! once in one small, deterministic corpus so a release cannot accidentally
//! pass unit tests while a route is disconnected from the collection planner.

use a3s_vec::{
    AlterColumnOption, Collection, CollectionOptions, CollectionSchema, DataType,
    DiskannQueryParams, Doc, Durability, ErrorCode, FieldSchema, Fts, GroupBySearchQuery,
    HnswQueryParams, IndexParams, IndexType, IoBackend, IvfQueryParams, IvfRabitqQueryParams,
    MetricType, MultiQuery, SearchQuery, SubQuery,
};
use tempfile::tempdir;

const DIMENSION: u32 = 8;
const DOCUMENTS: usize = 48;

fn manual_options() -> CollectionOptions {
    let mut options = CollectionOptions::new().expect("options must be valid");
    options
        .set_durability(Durability::Manual)
        .expect("manual durability must be valid");
    options
}

fn vector_for(index: usize) -> Vec<f32> {
    (0..usize::try_from(DIMENSION).expect("dimension fits usize"))
        .map(|dimension| {
            let raw = (index * 31 + dimension * 17 + index * dimension * 3) % 101;
            (f32::from(u16::try_from(raw).expect("fixture value fits u16")) - 50.0) / 50.0
        })
        .collect()
}

fn document(index: usize) -> Doc {
    let mut doc = Doc::with_pk(format!("doc-{index:03}")).expect("document must be valid");
    doc.add_string("category", if index % 2 == 0 { "even" } else { "odd" })
        .expect("category must be valid");
    doc.add_i32("bucket", i32::try_from(index % 6).expect("bucket fits i32"))
        .expect("bucket must be valid");
    doc.add_string(
        "body",
        &format!(
            "rust workspace vector retrieval document group{} token{}",
            index % 3,
            index % 7
        ),
    )
    .expect("body must be valid");
    doc.add_vector_f32("embedding", &vector_for(index))
        .expect("dense vector must be valid");
    let first = u32::try_from(index % 16).expect("sparse index fits u32");
    let second = (first + 3) % 16;
    doc.add_sparse_vector("sparse", &[first, second], &[1.0, 0.25])
        .expect("sparse vector must be valid");
    doc
}

fn ann_document(index: usize) -> Doc {
    let mut doc = Doc::with_pk(format!("doc-{index:03}")).expect("ANN document must be valid");
    doc.add_vector_f32("embedding", &vector_for(index))
        .expect("ANN vector must be valid");
    doc
}

fn indexed_schema() -> CollectionSchema {
    let mut category =
        FieldSchema::new("category", DataType::String, false, 0).expect("category schema");
    category
        .set_index_params(&IndexParams::invert(true, true).expect("scalar index"))
        .expect("category index");
    let mut bucket = FieldSchema::new("bucket", DataType::Int32, false, 0).expect("bucket schema");
    bucket
        .set_index_params(&IndexParams::invert(true, false).expect("scalar index"))
        .expect("bucket index");
    let mut body = FieldSchema::new("body", DataType::String, false, 0).expect("body schema");
    body.set_index_params(
        &IndexParams::fts(Some("standard"), Some(&["lowercase"]), None).expect("FTS index"),
    )
    .expect("body index");
    CollectionSchema::builder("feature-matrix")
        .add_field(category)
        .add_field(bucket)
        .add_field(body)
        .add_field(
            FieldSchema::new("embedding", DataType::VectorFp32, false, DIMENSION)
                .expect("embedding schema"),
        )
        .add_field(
            FieldSchema::new("sparse", DataType::SparseVectorFp32, false, 16)
                .expect("sparse schema"),
        )
        .build()
        .expect("schema must be valid")
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
    assert_eq!(result.error_count, 0);
}

fn ids(docs: &[Doc]) -> Vec<String> {
    docs.iter()
        .map(|doc| {
            doc.get_pk()
                .expect("result must have a primary key")
                .to_string()
        })
        .collect()
}

fn assert_same_ranking(left: &[Doc], right: &[Doc]) {
    assert_eq!(ids(left), ids(right));
    assert_eq!(
        left.iter().map(Doc::get_score).collect::<Vec<_>>(),
        right.iter().map(Doc::get_score).collect::<Vec<_>>()
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn every_public_query_route_and_lifecycle_has_a_small_oracle() {
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary.path().join("routes");
    let path_string = path.to_str().expect("path must be UTF-8");
    let collection = Collection::create(path_string, &indexed_schema(), Some(&manual_options()))
        .expect("collection must be created");
    insert_fixture(&collection);

    // CRUD, projection, fetch, and snapshot iteration.
    assert_eq!(collection.count().expect("count must succeed"), DOCUMENTS);
    let fetched = collection
        .fetch_with_options(&["doc-003"], Some(&["body"]), false)
        .expect("fetch must succeed");
    assert_eq!(fetched.len(), 1);
    assert!(fetched[0].vector("embedding").is_none());
    assert!(fetched[0]
        .get_string("body")
        .expect("body getter")
        .is_some());
    let iterated = collection
        .iter_with_options(Some(&["category"]), false)
        .expect("iterator must be created")
        .collect::<a3s_vec::Result<Vec<_>>>()
        .expect("iterator must succeed");
    assert_eq!(iterated.len(), DOCUMENTS);
    assert!(iterated.iter().all(|doc| doc.vector("embedding").is_none()));

    // Exact dense metrics and radius filtering share one independent route.
    for metric in ["l2", "ip", "cosine", "mips_l2"] {
        let mut query =
            SearchQuery::new("embedding", &vector_for(11), 5).expect("dense query must be valid");
        query
            .params
            .insert("metric".into(), serde_json::json!(metric));
        let result = collection.query(&query).expect("dense query must succeed");
        assert_eq!(result.len(), 5, "metric {metric} must return top-k");
        assert!(result.iter().all(|doc| doc.get_score().is_finite()));
    }
    let mut radius =
        SearchQuery::new("embedding", &vector_for(11), 10).expect("radius query must be valid");
    radius
        .set_radius(0.0)
        .expect("zero radius must be accepted");
    assert!(!collection.query(&radius).expect("radius query").is_empty());

    // Dense source-ID and explicit payload routes must agree for every metric.
    for metric in ["l2", "ip", "cosine", "mips_l2"] {
        let mut explicit = SearchQuery::new("embedding", &vector_for(11), 5)
            .expect("explicit dense query must be valid");
        explicit
            .params
            .insert("metric".into(), serde_json::json!(metric));
        let mut by_id = SearchQuery::by_id("embedding", "doc-011", 5).expect("dense source query");
        by_id
            .params
            .insert("metric".into(), serde_json::json!(metric));
        assert_same_ranking(
            &collection
                .query(&explicit)
                .expect("explicit dense query must succeed"),
            &collection
                .query(&by_id)
                .expect("dense source-ID query must succeed"),
        );
    }

    // Sparse explicit payload and source-document routes must agree.
    let sparse = SearchQuery::sparse("sparse", &[11, 14], &[1.0, 0.25], 5)
        .expect("sparse query must be valid");
    let explicit_sparse = collection
        .query(&sparse)
        .expect("sparse query must succeed");
    let source_sparse =
        SearchQuery::by_id("sparse", "doc-011", 5).expect("source-id query must be valid");
    let source_sparse = collection
        .query(&source_sparse)
        .expect("source-id query must succeed");
    assert_same_ranking(&explicit_sparse, &source_sparse);

    // Scalar + FTS and hybrid fusion routes.
    let mut fts = Fts::new().expect("FTS payload must be valid");
    fts.set_query_string("rust AND vector")
        .expect("FTS expression must be valid");
    let mut lexical = SearchQuery::fts("body", &fts, 8).expect("FTS query must be valid");
    lexical
        .set_filter("category == 'even'")
        .expect("FTS filter must be valid");
    let lexical_hits = collection.query(&lexical).expect("FTS query must succeed");
    assert!(!lexical_hits.is_empty());
    assert!(lexical_hits.iter().all(|doc| {
        doc.get_string("category")
            .expect("category getter")
            .as_deref()
            == Some("even")
    }));

    let mut vector_branch = SubQuery::new().expect("sub-query must be valid");
    vector_branch
        .set_field_name("embedding")
        .expect("sub-query field must be valid");
    vector_branch
        .set_query_vector(&vector_for(11))
        .expect("sub-query vector must be valid");
    vector_branch
        .set_num_candidates(12)
        .expect("candidate count must be valid");
    let mut fts_branch = SubQuery::new().expect("sub-query must be valid");
    fts_branch
        .set_field_name("body")
        .expect("sub-query field must be valid");
    fts_branch.set_fts(&fts).expect("FTS branch must be valid");
    fts_branch
        .set_num_candidates(12)
        .expect("candidate count must be valid");
    let mut hybrid = MultiQuery::new().expect("multi-query must be valid");
    hybrid.set_topk(6).expect("multi-query top-k must be valid");
    hybrid
        .set_filter("bucket >= 0")
        .expect("multi-query filter must be valid");
    hybrid
        .add_sub_query(&vector_branch)
        .expect("vector branch must be added");
    hybrid
        .add_sub_query(&fts_branch)
        .expect("FTS branch must be added");
    hybrid.set_rerank_rrf(60).expect("RRF rerank must be valid");
    let fused = collection
        .multi_query(&hybrid)
        .expect("multi-query must succeed");
    assert!(!fused.is_empty());
    assert!(fused
        .windows(2)
        .all(|window| window[0].get_pk() != window[1].get_pk()));

    let mut grouped = GroupBySearchQuery::new("embedding", "category", &vector_for(11), 2, 2)
        .expect("group query must be valid");
    grouped
        .set_output_fields(&["category"])
        .expect("group projection must be valid");
    let groups = collection
        .group_by(&grouped)
        .expect("group-by must succeed");
    assert_eq!(groups.len(), 2);
    assert!(groups.values().all(|values| values.len() <= 2));

    // Update, upsert, filtered delete, schema evolution, flush, and reopen.
    let mut update_doc = Doc::with_pk("doc-003").expect("patch must be valid");
    update_doc
        .add_i32("bucket", 5)
        .expect("patch bucket must be valid");
    collection
        .update(&[&update_doc])
        .expect("partial update must succeed");
    let replacement = document(3);
    collection
        .upsert(&[&replacement])
        .expect("replacement upsert must succeed");
    collection
        .delete_by_filter("bucket == 4 and category == 'even'")
        .expect("filtered delete must succeed");
    assert!(collection.count().expect("count must succeed") < DOCUMENTS);

    let extra = FieldSchema::new("extra", DataType::Int32, false, 0).expect("extra schema");
    collection
        .add_column(&extra, Some("7"))
        .expect("add column must succeed");
    collection
        .rename_column("extra", "renamed")
        .expect("rename column must succeed");
    let nullable = FieldSchema::new("renamed", DataType::Int32, true, 0).expect("alter schema");
    collection
        .alter_column(&nullable, AlterColumnOption::default())
        .expect("alter column must succeed");
    assert_eq!(
        collection.fetch(&["doc-001"]).expect("fetch evolved doc")[0]
            .get_i32("renamed")
            .expect("evolved getter"),
        Some(7)
    );
    collection
        .drop_column("renamed")
        .expect("drop column must succeed");
    collection.flush().expect("flush must succeed");
    collection.close().expect("collection must close");

    let reopened = Collection::open(path_string, None).expect("collection must reopen");
    assert!(!reopened
        .schema()
        .expect("schema must be readable")
        .has_field("renamed"));
    assert!(reopened
        .health()
        .expect("health must be readable")
        .is_healthy());
    reopened.close().expect("reopened collection must close");
}

fn ann_schema() -> CollectionSchema {
    CollectionSchema::builder("ann-matrix")
        .add_field(
            FieldSchema::new("embedding", DataType::VectorFp32, false, DIMENSION)
                .expect("embedding schema"),
        )
        .build()
        .expect("ANN schema must be valid")
}

#[derive(Clone, Copy)]
enum AnnCase {
    Hnsw,
    IvfSoar,
    HnswRabitq,
    IvfRabitq,
    Vamana,
    VamanaIp,
    VamanaCosine,
    VamanaMipsL2,
    Diskann,
    DiskannIp,
    DiskannCosine,
    DiskannMipsL2,
}

impl AnnCase {
    fn name(self) -> &'static str {
        match self {
            Self::Hnsw => "hnsw",
            Self::IvfSoar => "ivf_soar",
            Self::HnswRabitq => "hnsw_rabitq",
            Self::IvfRabitq => "ivf_rabitq",
            Self::Vamana => "vamana",
            Self::VamanaIp => "vamana_ip",
            Self::VamanaCosine => "vamana_cosine",
            Self::VamanaMipsL2 => "vamana_mips_l2",
            Self::Diskann => "diskann",
            Self::DiskannIp => "diskann_ip",
            Self::DiskannCosine => "diskann_cosine",
            Self::DiskannMipsL2 => "diskann_mips_l2",
        }
    }

    fn metric(self) -> MetricType {
        match self {
            Self::Hnsw
            | Self::IvfSoar
            | Self::HnswRabitq
            | Self::IvfRabitq
            | Self::VamanaCosine
            | Self::DiskannCosine => MetricType::Cosine,
            Self::Vamana | Self::Diskann => MetricType::L2,
            Self::VamanaIp | Self::DiskannIp => MetricType::Ip,
            Self::VamanaMipsL2 | Self::DiskannMipsL2 => MetricType::MipsL2,
        }
    }

    fn metric_name(self) -> &'static str {
        match self.metric() {
            MetricType::L2 => "l2",
            MetricType::Ip => "ip",
            MetricType::Cosine => "cosine",
            MetricType::MipsL2 => "mips_l2",
            MetricType::Undefined => "undefined",
        }
    }

    fn params(self) -> IndexParams {
        match self {
            Self::Hnsw => IndexParams::hnsw(MetricType::Cosine, 8, 32),
            Self::IvfSoar => IndexParams::ivf(MetricType::Cosine, 8, 4, true),
            Self::HnswRabitq => {
                IndexParams::hnsw_rabitq_with_options(MetricType::Cosine, 8, 32, 5, 8, 0)
            }
            Self::IvfRabitq => IndexParams::ivf_rabitq(MetricType::Cosine, 8, 5, 0),
            Self::Vamana | Self::VamanaIp | Self::VamanaCosine | Self::VamanaMipsL2 => {
                IndexParams::vamana(
                    self.metric(),
                    12,
                    i32::try_from(DOCUMENTS).expect("document count fits i32"),
                    1.2,
                )
            }
            Self::Diskann | Self::DiskannIp | Self::DiskannCosine | Self::DiskannMipsL2 => {
                IndexParams::diskann(
                    self.metric(),
                    12,
                    i32::try_from(DOCUMENTS).expect("document count fits i32"),
                    2,
                )
            }
        }
        .expect("ANN descriptor must be valid")
    }

    fn configure(self, query: &mut SearchQuery) {
        match self {
            Self::Hnsw | Self::HnswRabitq => query
                .set_hnsw_params(HnswQueryParams::new(
                    i32::try_from(DOCUMENTS).expect("document count fits i32"),
                    0.0,
                    false,
                    true,
                ))
                .expect("HNSW controls must be valid"),
            Self::IvfSoar => query
                .set_ivf_params(IvfQueryParams::new(8, true, 1.0))
                .expect("IVF controls must be valid"),
            Self::IvfRabitq => {
                let mut params = IvfRabitqQueryParams::new(8, 0.0, false, true);
                params
                    .set_scale_factor(1.0)
                    .expect("IVF RaBitQ scale must be valid");
                query
                    .set_ivf_rabitq_params(params)
                    .expect("IVF RaBitQ controls must be valid");
            }
            Self::Vamana
            | Self::VamanaIp
            | Self::VamanaCosine
            | Self::VamanaMipsL2
            | Self::Diskann
            | Self::DiskannIp
            | Self::DiskannCosine
            | Self::DiskannMipsL2 => {
                query
                    .params
                    .insert("metric".into(), serde_json::json!(self.metric_name()));
                query
                    .set_diskann_params(DiskannQueryParams::new(
                        i32::try_from(DOCUMENTS).expect("document count fits i32"),
                    ))
                    .expect("DiskANN controls must be valid");
            }
        }
    }

    fn expected_type(self) -> IndexType {
        match self {
            Self::Hnsw => IndexType::Hnsw,
            Self::IvfSoar => IndexType::Ivf,
            Self::HnswRabitq => IndexType::HnswRabitq,
            Self::IvfRabitq => IndexType::IvfRabitq,
            Self::Vamana | Self::VamanaIp | Self::VamanaCosine | Self::VamanaMipsL2 => {
                IndexType::Vamana
            }
            Self::Diskann | Self::DiskannIp | Self::DiskannCosine | Self::DiskannMipsL2 => {
                IndexType::Diskann
            }
        }
    }
}

#[test]
fn every_ann_family_matches_the_exact_oracle_at_exhaustive_controls() {
    for case in [
        AnnCase::Hnsw,
        AnnCase::IvfSoar,
        AnnCase::HnswRabitq,
        AnnCase::IvfRabitq,
        AnnCase::Vamana,
        AnnCase::VamanaIp,
        AnnCase::VamanaCosine,
        AnnCase::VamanaMipsL2,
        AnnCase::Diskann,
        AnnCase::DiskannIp,
        AnnCase::DiskannCosine,
        AnnCase::DiskannMipsL2,
    ] {
        let temporary = tempdir().expect("temporary directory must be available");
        let path = temporary.path().join(case.name());
        let path_string = path.to_str().expect("path must be UTF-8");
        let collection = Collection::create(path_string, &ann_schema(), None)
            .expect("ANN collection must be created");
        let docs: Vec<Doc> = (0..DOCUMENTS).map(ann_document).collect();
        let inserted = collection
            .insert(&docs.iter().collect::<Vec<_>>())
            .expect("ANN fixture insert must succeed");
        assert_eq!(
            inserted.success_count,
            u64::try_from(DOCUMENTS).expect("document count fits u64")
        );
        let mut query =
            SearchQuery::new("embedding", &vector_for(17), 12).expect("ANN query must be valid");
        if matches!(
            case,
            AnnCase::Vamana
                | AnnCase::VamanaIp
                | AnnCase::VamanaCosine
                | AnnCase::VamanaMipsL2
                | AnnCase::Diskann
                | AnnCase::DiskannIp
                | AnnCase::DiskannCosine
                | AnnCase::DiskannMipsL2
        ) {
            query
                .params
                .insert("metric".into(), serde_json::json!(case.metric_name()));
        }
        let exact = collection.query(&query).expect("exact oracle must succeed");
        collection
            .create_index("embedding", &case.params())
            .expect("ANN index must build");
        case.configure(&mut query);
        let indexed = collection.query(&query).expect("ANN query must succeed");
        assert_same_ranking(&exact, &indexed);
        let stats = collection.stats().expect("stats must be available");
        assert_eq!(stats.indexes[0].index_type, case.expected_type());
        assert_eq!(stats.indexes[0].state, "ready");
        assert_eq!(stats.indexes[0].source_revision, stats.revision);
        collection.close().expect("ANN collection must close");
    }
}

#[test]
fn cache_sidecar_health_and_unsupported_boundaries_are_explicit() {
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary.path().join("sidecar");
    let path_string = path.to_str().expect("path must be UTF-8");
    let collection =
        Collection::create(path_string, &ann_schema(), None).expect("collection must be created");
    let docs: Vec<Doc> = (0..DOCUMENTS).map(ann_document).collect();
    let inserted = collection
        .insert(&docs.iter().collect::<Vec<_>>())
        .expect("fixture insert must succeed");
    assert_eq!(
        inserted.success_count,
        u64::try_from(DOCUMENTS).expect("document count fits u64")
    );
    collection
        .create_index(
            "embedding",
            &IndexParams::diskann(
                MetricType::L2,
                12,
                i32::try_from(DOCUMENTS).expect("document count fits i32"),
                2,
            )
            .expect("DiskANN descriptor must be valid"),
        )
        .expect("DiskANN index must build");
    collection.flush().expect("flush must succeed");
    collection.close().expect("collection must close");

    let mut options = CollectionOptions::new().expect("options must be valid");
    options
        .set_read_only(true)
        .expect("read-only option must be valid");
    options
        .set_io_backend(IoBackend::Positioned)
        .expect("positioned backend must be valid");
    let reopened = Collection::open(path_string, Some(&options)).expect("reopen must succeed");
    assert!(
        reopened
            .stats()
            .expect("stats must be available")
            .index_cache_hit
    );
    assert!(reopened
        .health()
        .expect("health must be available")
        .is_healthy());
    let mut query = SearchQuery::new("embedding", &vector_for(4), 5).expect("query must be valid");
    query
        .params
        .insert("metric".into(), serde_json::json!("l2"));
    query
        .set_diskann_params(DiskannQueryParams::new(16))
        .expect("DiskANN controls must be valid");
    let sidecar_hits = reopened.query(&query).expect("sidecar query must succeed");
    assert!(
        !sidecar_hits.is_empty(),
        "sidecar query unexpectedly returned no hits: {:?}",
        reopened.stats().expect("stats must be available")
    );
    reopened.close().expect("read-only collection must close");

    // Binary execution is a deliberate typed boundary until its metric and
    // index contract is specified; the failure itself is regression-tested.
    let binary_schema = CollectionSchema::builder("binary-boundary")
        .add_field(
            FieldSchema::new("bits", DataType::VectorBinary32, false, 32)
                .expect("binary schema must be valid"),
        )
        .build()
        .expect("binary schema must build");
    let binary_path = temporary.path().join("binary");
    let binary = Collection::create(
        binary_path.to_str().expect("binary path must be UTF-8"),
        &binary_schema,
        None,
    )
    .expect("binary collection must be created");
    let mut binary_doc = Doc::with_pk("binary-1").expect("binary doc must be valid");
    binary_doc
        .add_vector_binary32("bits", &[0b1010_1010; 4])
        .expect("binary payload must be valid");
    binary
        .insert(&[&binary_doc])
        .expect("binary insert must succeed");
    let query = SearchQuery::new("bits", &[0.0; 32], 1).expect("query constructor must succeed");
    let error = binary
        .query(&query)
        .expect_err("binary query must be explicit");
    assert_eq!(error.code, ErrorCode::NotSupported);
    binary.close().expect("binary collection must close");
}
