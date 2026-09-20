#![allow(clippy::cast_precision_loss)]
use super::IndexRegistry;
use crate::doc::DocumentMap;
use crate::index::ordinals::OrdinalSet;
use crate::{
    CollectionSchema, DataType, Doc, FieldSchema, Fts, HnswQueryParams, IndexParams, MetricType,
    SearchQuery,
};
use roaring::RoaringTreemap;
use std::collections::BTreeSet;
use std::sync::Arc;

fn indexed_schema() -> CollectionSchema {
    indexed_schema_with(
        &IndexParams::hnsw(MetricType::L2, 4, 16).expect("descriptor must be valid"),
    )
}

fn indexed_schema_with(params: &IndexParams) -> CollectionSchema {
    let mut field =
        FieldSchema::new("embedding", DataType::VectorFp32, false, 2).expect("field must be valid");
    field
        .set_index_params(params)
        .expect("ANN index must be supported");
    CollectionSchema::builder("incremental")
        .add_field(field)
        .build()
        .expect("schema must be valid")
}

fn vector_doc(id: &str, vector: &[f32]) -> Arc<Doc> {
    let mut doc = Doc::with_pk(id).expect("document must be valid");
    doc.add_vector_f32("embedding", vector)
        .expect("vector must be valid");
    Arc::new(doc)
}

fn mixed_index_schema() -> CollectionSchema {
    let mut embedding =
        FieldSchema::new("embedding", DataType::VectorFp32, false, 2).expect("field must be valid");
    embedding
        .set_index_params(
            &IndexParams::hnsw(MetricType::L2, 4, 16).expect("HNSW params must be valid"),
        )
        .expect("HNSW index must be valid");
    let mut language =
        FieldSchema::new("language", DataType::String, false, 0).expect("field must be valid");
    language
        .set_index_params(&IndexParams::invert(false, false).expect("scalar params must be valid"))
        .expect("scalar index must be valid");
    let mut body =
        FieldSchema::new("body", DataType::String, false, 0).expect("field must be valid");
    body.set_index_params(
        &IndexParams::fts(Some("standard"), None, None).expect("FTS params must be valid"),
    )
    .expect("FTS index must be valid");
    CollectionSchema::builder("mixed-indexes")
        .add_field(embedding)
        .add_field(language)
        .add_field(body)
        .build()
        .expect("schema must be valid")
}

#[test]
fn targeted_rebuild_preserves_unrelated_ann_generation() {
    let schema = mixed_index_schema();
    let docs: DocumentMap = (0_u16..32)
        .map(|value| {
            let id = format!("doc-{value:03}");
            let mut doc = Doc::with_pk(&id).expect("document must be valid");
            doc.add_vector_f32("embedding", &[f32::from(value), 0.0])
                .expect("vector must be valid");
            doc.add_string("language", if value % 2 == 0 { "rust" } else { "go" })
                .expect("language must be valid");
            doc.add_string("body", &format!("workspace symbol{value}"))
                .expect("body must be valid");
            (id, Arc::new(doc))
        })
        .collect();
    let indexes = IndexRegistry::build(&schema, &docs, 1).expect("indexes must build");
    let initial = indexes.indexes.get("embedding").expect("ANN must exist");

    let scalar = indexes
        .rebuild_field(&schema, &docs, 1, "language")
        .expect("scalar rebuild must succeed");
    let after_scalar = scalar.indexes.get("embedding").expect("ANN must remain");
    assert!(Arc::ptr_eq(&initial.base, &after_scalar.base));

    let fts = scalar
        .rebuild_field(&schema, &docs, 1, "body")
        .expect("FTS rebuild must succeed");
    let after_fts = fts.indexes.get("embedding").expect("ANN must remain");
    assert!(Arc::ptr_eq(&initial.base, &after_fts.base));

    let ann = fts
        .rebuild_field(&schema, &docs, 1, "embedding")
        .expect("ANN rebuild must succeed");
    let rebuilt = ann.indexes.get("embedding").expect("ANN must remain");
    assert!(!Arc::ptr_eq(&initial.base, &rebuilt.base));
}

#[test]
fn indexed_fts_plan_keeps_scores_in_the_ordinal_generation() {
    let schema = mixed_index_schema();
    let docs: DocumentMap = [
        ("doc-z", "workspace rust"),
        ("doc-a", "workspace rust rust"),
        ("doc-m", "unrelated"),
    ]
    .into_iter()
    .map(|(id, body)| {
        let mut doc = Doc::with_pk(id).expect("document must be valid");
        doc.add_vector_f32("embedding", &[0.0, 0.0])
            .expect("vector must be valid");
        doc.add_string("language", "rust")
            .expect("language must be valid");
        doc.add_string("body", body).expect("body must be valid");
        (id.to_string(), Arc::new(doc))
    })
    .collect();
    let indexes = IndexRegistry::build(&schema, &docs, 1).expect("indexes must build");
    let mut fts = Fts::new().expect("FTS payload must be valid");
    fts.set_match_string("workspace rust")
        .expect("FTS expression must be valid");
    let query = SearchQuery::fts("body", &fts, 1).expect("FTS query must be valid");

    let plan = indexes
        .plan_candidates(&docs, 1, &query, None)
        .expect("candidate plan must build");
    let scores = plan.fts_scores.as_ref().expect("FTS scores must exist");

    assert!(plan.selection.is_none());
    assert!(plan.used_fts_index);
    assert_eq!(plan.candidate_count(docs.len()), 2);
    assert_eq!(scores.entries().count(), 1);
}

#[test]
fn stale_generation_declines_selection_for_exact_fallback() {
    let schema = indexed_schema();
    let mut doc = Doc::with_pk("one").expect("document must be valid");
    doc.add_vector_f32("embedding", &[1.0, 0.0])
        .expect("vector must be valid");
    let docs = [("one".to_string(), Arc::new(doc))]
        .into_iter()
        .collect::<DocumentMap>();
    let indexes = IndexRegistry::build(&schema, &docs, 3).expect("index must build");
    let query = SearchQuery::new("embedding", &[1.0, 0.0], 1).expect("query must be valid");
    assert!(indexes
        .candidates(&docs, 4, &query, None)
        .expect("selection must not fail")
        .is_none());
}

#[test]
fn incremental_generation_shares_base_and_shadows_updates_and_deletes() {
    let schema = indexed_schema();
    let docs = [
        ("base-a".to_string(), vector_doc("base-a", &[0.0, 0.0])),
        ("base-b".to_string(), vector_doc("base-b", &[10.0, 0.0])),
    ]
    .into_iter()
    .collect::<DocumentMap>();
    let indexes = IndexRegistry::build(&schema, &docs, 1).expect("index must build");

    let mut scalar_only = docs.clone();
    Arc::make_mut(scalar_only.get_mut("base-a").expect("document must exist"))
        .set_score(1.0)
        .expect("score must be valid");
    let scalar_generation = indexes
        .apply_document_changes(
            &schema,
            &docs,
            &scalar_only,
            2,
            &BTreeSet::from(["base-a".to_string()]),
        )
        .expect("scalar-only generation must publish");
    let initial = indexes.indexes.get("embedding").expect("index must exist");
    let scalar = scalar_generation
        .indexes
        .get("embedding")
        .expect("index must exist");
    assert!(Arc::ptr_eq(&initial.base, &scalar.base));
    assert!(scalar.delta.is_empty());
    assert!(scalar.tombstones.is_empty());

    let mut next_docs = scalar_only.clone();
    next_docs.insert("base-a".to_string(), vector_doc("base-a", &[100.0, 0.0]));
    next_docs.remove("base-b");
    next_docs.insert("delta-c".to_string(), vector_doc("delta-c", &[50.0, 0.0]));
    let changed = BTreeSet::from([
        "base-a".to_string(),
        "base-b".to_string(),
        "delta-c".to_string(),
    ]);
    let next = scalar_generation
        .apply_document_changes(&schema, &scalar_only, &next_docs, 3, &changed)
        .expect("incremental generation must publish");
    let incremental = next.indexes.get("embedding").expect("index must exist");
    assert!(Arc::ptr_eq(&initial.base, &incremental.base));
    assert_eq!(incremental.delta.len(), 2);
    let expected_tombstones: RoaringTreemap = ["base-a", "base-b"]
        .into_iter()
        .filter_map(|id| next.ordinals.ordinal(id))
        .collect();
    assert_eq!(incremental.tombstones, expected_tombstones);

    let mut query = SearchQuery::new("embedding", &[100.0, 0.0], 2).expect("query must be valid");
    query
        .set_hnsw_params(HnswQueryParams::new(2, 0.0, false, true))
        .expect("HNSW controls must be valid");
    let candidates = next
        .candidates(&next_docs, 3, &query, None)
        .expect("candidate selection must succeed")
        .expect("ANN generation must be selected");
    assert_eq!(
        candidates.selection.ids().collect::<BTreeSet<_>>(),
        BTreeSet::from(["base-a", "delta-c"])
    );
    let stat = &next.stats(&schema, &next_docs, 3)[0];
    assert_eq!(stat.document_count, 2);
    assert_eq!(stat.source_revision, 3);
    assert!(stat.estimated_payload_bytes.is_some_and(|bytes| bytes > 0));
}

#[test]
fn incremental_equal_scores_keep_primary_key_order_across_ordinals() {
    let schema = indexed_schema();
    let docs = [("z-base".to_string(), vector_doc("z-base", &[1.0, 0.0]))]
        .into_iter()
        .collect::<DocumentMap>();
    let indexes = IndexRegistry::build(&schema, &docs, 1).expect("index must build");

    let mut next_docs = docs.clone();
    next_docs.insert("a-delta".to_string(), vector_doc("a-delta", &[1.0, 0.0]));
    let next = indexes
        .apply_document_changes(
            &schema,
            &docs,
            &next_docs,
            2,
            &BTreeSet::from(["a-delta".to_string()]),
        )
        .expect("incremental generation must publish");
    assert!(
        next.ordinals.ordinal("z-base").expect("base ordinal")
            < next.ordinals.ordinal("a-delta").expect("delta ordinal")
    );

    let mut query = SearchQuery::new("embedding", &[1.0, 0.0], 1).expect("query must be valid");
    query
        .set_hnsw_params(HnswQueryParams::new(1, 0.0, false, true))
        .expect("HNSW controls must be valid");
    let candidates = next
        .candidates(&next_docs, 2, &query, None)
        .expect("candidate selection must succeed")
        .expect("ANN generation must be selected");
    assert_eq!(
        candidates.selection.ids().collect::<Vec<_>>(),
        vec!["a-delta"]
    );
}

#[test]
fn ordinal_compaction_rebuilds_ann_membership_without_filter_drift() {
    const DOCUMENTS: usize = 513;
    const DELETIONS: usize = 64;

    for params in [
        IndexParams::hnsw(MetricType::L2, 8, 32).expect("HNSW descriptor must be valid"),
        IndexParams::ivf(MetricType::L2, 16, 5, false).expect("IVF descriptor must be valid"),
    ] {
        let schema = indexed_schema_with(&params);
        let docs: DocumentMap = (0..DOCUMENTS)
            .map(|index| {
                let id = format!("doc-{index:04}");
                let coordinate = f32::from(u16::try_from(index).expect("fixture index fits u16"));
                (id.clone(), vector_doc(&id, &[coordinate, 0.0]))
            })
            .collect();
        let indexes = IndexRegistry::build(&schema, &docs, 1).expect("index must build");
        let initial = indexes.indexes.get("embedding").expect("index must exist");

        let mut next_docs = docs.clone();
        let changed: BTreeSet<String> = (0..DELETIONS)
            .map(|index| format!("doc-{index:04}"))
            .collect();
        for id in &changed {
            next_docs.remove(id);
        }
        let next = indexes
            .apply_document_changes(&schema, &docs, &next_docs, 2, &changed)
            .expect("compacted generation must publish");
        let compacted = next.indexes.get("embedding").expect("index must exist");

        // Sixty-four retired ordinals trigger ordinal compaction at this
        // collection size, while the vector overlay threshold is 65.
        assert!(!Arc::ptr_eq(&initial.base, &compacted.base));
        assert!(compacted.delta.is_empty());
        assert!(compacted.tombstones.is_empty());
        assert_eq!(
            compacted.base.vector_ordinals.len(),
            u64::try_from(DOCUMENTS - DELETIONS).expect("fixture size fits u64")
        );

        let allowed_bitmap: RoaringTreemap = (DELETIONS..DOCUMENTS)
            .filter(|index| index % 2 == 0)
            .filter_map(|index| next.ordinals.ordinal(&format!("doc-{index:04}")))
            .collect();
        let allowed = OrdinalSet::new(&next.ordinals, allowed_bitmap);
        let query = SearchQuery::new("embedding", &[512.0, 0.0], 5).expect("query must be valid");
        let candidates = next
            .candidates(&next_docs, 2, &query, Some(&allowed))
            .expect("candidate selection must succeed")
            .expect("ANN generation must remain usable");
        assert!(candidates.selection.count() >= 5);
        assert!(candidates.selection.ids().all(|id| allowed.contains(id)));
    }
}

#[test]
fn exact_unquantized_score_by_ordinal_matches_document_promotion() {
    let schema = indexed_schema();
    let docs: DocumentMap = [
        ("a", [1.0_f32, 0.0]),
        ("b", [0.0_f32, 1.0]),
        ("c", [0.5_f32, 0.5]),
    ]
    .into_iter()
    .map(|(id, vector)| (id.to_string(), vector_doc(id, &vector)))
    .collect();
    let indexes = IndexRegistry::build(&schema, &docs, 1).expect("indexes must build");
    let query = [0.75_f32, 0.25];
    let query_norm =
        f64::from(query[0]) * f64::from(query[0]) + f64::from(query[1]) * f64::from(query[1]);
    let query_norm = query_norm.sqrt();
    for (id, vector) in [
        ("a", [1.0_f32, 0.0]),
        ("b", [0.0_f32, 1.0]),
        ("c", [0.5_f32, 0.5]),
    ] {
        let ordinal = indexes
            .document_ordinal(id)
            .expect("fixture document must have an ordinal");
        let from_index = indexes
            .exact_unquantized_f32_score_at(
                "embedding",
                ordinal,
                &query,
                query_norm,
                MetricType::L2,
            )
            .expect("ordinal score must resolve");
        let from_document = super::quantization::score_dense_with_query_norm(
            &query,
            &vector,
            MetricType::L2,
            query_norm,
        );
        assert_eq!(from_index.to_bits(), from_document.to_bits());
    }
}

#[test]
fn flat_index_returns_exact_topk_candidates() {
    let schema = indexed_schema_with(
        &IndexParams::flat(MetricType::Cosine).expect("Flat params must be valid"),
    );
    let docs: DocumentMap = [
        ("a", [1.0_f32, 0.0]),
        ("b", [0.0_f32, 1.0]),
        ("c", [0.8_f32, 0.2]),
        ("d", [-1.0_f32, 0.0]),
    ]
    .into_iter()
    .map(|(id, vector)| (id.to_string(), vector_doc(id, &vector)))
    .collect();
    let indexes = IndexRegistry::build(&schema, &docs, 1).expect("Flat indexes must build");
    assert!(
        indexes.stats(&schema, &docs, 1).iter().any(|stat| {
            stat.name == "embedding"
                && stat.index_type == crate::types::IndexType::Flat
                && stat.state == "ready"
        }),
        "Flat field must appear as a ready index"
    );
    let query = SearchQuery::new("embedding", &[1.0, 0.0], 2).expect("query must be valid");
    let plan = indexes
        .plan_candidates(&docs, 1, &query, None)
        .expect("Flat plan must succeed");
    assert!(
        !plan.used_ann,
        "Flat must record as exact, not approximate ANN"
    );
    let selection = plan.selection.expect("Flat must return candidate ordinals");
    assert_eq!(selection.count(), 2);
    let ids: Vec<_> = selection.ids().collect();
    assert_eq!(ids, ["a", "c"]);
}

fn flat_schema() -> CollectionSchema {
    indexed_schema_with(&IndexParams::flat(MetricType::Cosine).expect("Flat must be valid"))
}

#[test]
fn flat_overlay_tombstones_still_select_exact_live_candidates() {
    let schema = flat_schema();
    let docs = [
        ("a".to_string(), vector_doc("a", &[1.0, 0.0])),
        ("b".to_string(), vector_doc("b", &[0.0, 1.0])),
        ("c".to_string(), vector_doc("c", &[0.7, 0.3])),
    ]
    .into_iter()
    .collect::<DocumentMap>();
    let indexes = IndexRegistry::build(&schema, &docs, 1).expect("Flat must build");
    let mut overlays = indexes.clone();
    let flat = overlays
        .indexes
        .get_mut("embedding")
        .expect("Flat field must exist");
    let tombstone = overlays
        .ordinals
        .ordinal("b")
        .expect("ordinal for deleted doc");
    flat.tombstones.insert(tombstone);
    // Keep source_revision matched so Flat is selected; overlay forces live scan.
    assert_eq!(flat.source_revision, 1);

    let query = SearchQuery::new("embedding", &[1.0, 0.0], 2).expect("query");
    let candidates = overlays
        .candidates(&docs, 1, &query, None)
        .expect("selection")
        .expect("Flat generation must be selected");
    let ids: BTreeSet<_> = candidates.selection.ids().collect();
    assert_eq!(ids, BTreeSet::from(["a", "c"]));
}

#[test]
fn flat_filtered_allowed_bitmap_uses_live_scan_path() {
    let schema = flat_schema();
    let docs = (0..64)
        .map(|value| {
            let id = format!("d{value:02}");
            (id.clone(), vector_doc(&id, &[value as f32, 0.0]))
        })
        .collect::<DocumentMap>();
    let indexes = IndexRegistry::build(&schema, &docs, 1).expect("Flat must build");
    let mut allowed = RoaringTreemap::new();
    for value in [1_u64, 2, 3, 10, 20, 30] {
        if let Some(ordinal) = indexes.ordinals.ordinal(&format!("d{value:02}")) {
            allowed.insert(ordinal);
        }
    }
    let allowed_set = OrdinalSet::new(&indexes.ordinals, allowed);
    let query = SearchQuery::new("embedding", &[30.0, 0.0], 6).expect("query");
    let candidates = indexes
        .candidates(&docs, 1, &query, Some(&allowed_set))
        .expect("selection")
        .expect("Flat with allow-list");
    let ids: BTreeSet<_> = candidates.selection.ids().collect();
    assert_eq!(
        ids,
        BTreeSet::from(["d01", "d02", "d03", "d10", "d20", "d30"])
    );
}

#[test]
fn rebuild_field_rejects_unknown_and_unsupported_targets() {
    let schema = mixed_index_schema();
    let docs: DocumentMap = (0_u16..8)
        .map(|value| {
            let id = format!("doc-{value}");
            let mut doc = Doc::with_pk(&id).expect("document must be valid");
            doc.add_vector_f32("embedding", &[f32::from(value), 0.0])
                .expect("vector must be valid");
            doc.add_string("language", "rust")
                .expect("language must be valid");
            doc.add_string("body", "workspace")
                .expect("body must be valid");
            (id, Arc::new(doc))
        })
        .collect();
    let indexes = IndexRegistry::build(&schema, &docs, 1).expect("indexes must build");
    assert!(indexes.rebuild_field(&schema, &docs, 1, "missing").is_err());

    // Unindexed scalar field is not a rebuild target.
    let mut schema_with_tag = mixed_index_schema();
    schema_with_tag
        .fields
        .push(FieldSchema::new("tag", DataType::String, false, 0).expect("tag"));
    assert!(indexes
        .rebuild_field(&schema_with_tag, &docs, 1, "tag")
        .is_err());
}

#[test]
fn build_rejects_incomplete_or_nonpositive_ann_construction_params() {
    use serde_json::json;

    let docs: DocumentMap = (0_u16..8)
        .map(|value| {
            let id = format!("doc-{value}");
            (id.clone(), vector_doc(&id, &[f32::from(value), 0.0]))
        })
        .collect();

    let mut schema = indexed_schema();
    schema.vectors[0]
        .index_params
        .as_mut()
        .expect("hnsw")
        .params
        .remove("m");
    assert!(IndexRegistry::build(&schema, &docs, 1).is_err());

    let mut schema = indexed_schema();
    schema.vectors[0]
        .index_params
        .as_mut()
        .expect("hnsw")
        .params
        .insert("ef_construction".into(), json!(0));
    assert!(IndexRegistry::build(&schema, &docs, 1).is_err());

    let mut schema =
        indexed_schema_with(&IndexParams::ivf(MetricType::L2, 4, 1, false).expect("ivf"));
    schema.vectors[0]
        .index_params
        .as_mut()
        .expect("ivf")
        .params
        .remove("use_soar");
    assert!(IndexRegistry::build(&schema, &docs, 1).is_err());

    let mut schema =
        indexed_schema_with(&IndexParams::diskann(MetricType::L2, 8, 16, 0).expect("diskann"));
    schema.vectors[0]
        .index_params
        .as_mut()
        .expect("diskann")
        .params
        .insert("alpha".into(), json!(f64::NAN));
    assert!(IndexRegistry::build(&schema, &docs, 1).is_err());

    let mut schema =
        indexed_schema_with(&IndexParams::vamana(MetricType::L2, 8, 16, 1.2).expect("vamana"));
    schema.vectors[0]
        .index_params
        .as_mut()
        .expect("vamana")
        .params
        .remove("saturate");
    assert!(IndexRegistry::build(&schema, &docs, 1).is_err());

    let mut schema =
        indexed_schema_with(&IndexParams::hnsw_rabitq(MetricType::L2, 8, 32).expect("hnsw-rabitq"));
    schema.vectors[0]
        .index_params
        .as_mut()
        .expect("hnsw-rabitq")
        .params
        .remove("total_bits");
    assert!(IndexRegistry::build(&schema, &docs, 1).is_err());

    let mut schema = indexed_schema_with(
        &IndexParams::ivf_rabitq(MetricType::L2, 8, 4, 32).expect("ivf-rabitq"),
    );
    schema.vectors[0]
        .index_params
        .as_mut()
        .expect("ivf-rabitq")
        .params
        .insert("n_list".into(), json!(0));
    assert!(IndexRegistry::build(&schema, &docs, 1).is_err());
}

#[test]
fn missing_packed_ann_reports_incomplete_stats_and_flat_absent_stays_ready() {
    let schema = indexed_schema();
    let docs: DocumentMap = (0_u16..8)
        .map(|value| {
            let id = format!("doc-{value}");
            (id.clone(), vector_doc(&id, &[f32::from(value), 0.0]))
        })
        .collect();
    let indexes = IndexRegistry::build(&schema, &docs, 1).expect("indexes must build");
    let mut stripped = indexes.clone();
    stripped.indexes.clear();
    let stats = stripped.stats(&schema, &docs, 1);
    assert!(
        stats.iter().any(|stat| {
            stat.name == "embedding"
                && stat.state == "missing"
                && (stat.completeness - 0.0).abs() < f32::EPSILON
        }),
        "absent HNSW must surface as missing, got {stats:?}"
    );

    let flat_schema = flat_schema();
    let flat_docs = docs.clone();
    let flat = IndexRegistry::build(&flat_schema, &flat_docs, 1).expect("flat must build");
    let mut flat_stripped = flat.clone();
    flat_stripped.indexes.clear();
    let flat_stats = flat_stripped.stats(&flat_schema, &flat_docs, 7);
    assert!(
        flat_stats.iter().any(|stat| {
            stat.name == "embedding"
                && stat.index_type == crate::types::IndexType::Flat
                && stat.state == "ready"
                && stat.source_revision == 7
        }),
        "absent packed Flat remains document-scan ready, got {flat_stats:?}"
    );
}

#[test]
fn binary_flat_stats_remain_document_backed() {
    let mut field = FieldSchema::new("bits", DataType::VectorBinary32, false, 32).expect("field");
    field
        .set_index_params(&IndexParams::flat(MetricType::L2).expect("flat"))
        .expect("binary flat");
    let schema = CollectionSchema::builder("binary-flat-stats")
        .add_field(field)
        .build()
        .expect("schema");
    let mut doc = Doc::with_pk("one").expect("doc");
    doc.add_vector_binary32("bits", &[0, 0, 0, 0])
        .expect("binary");
    let docs: DocumentMap = [("one".to_string(), Arc::new(doc))].into_iter().collect();
    let indexes = IndexRegistry::build(&schema, &docs, 3).expect("build");
    let stats = indexes.stats(&schema, &docs, 3);
    assert!(
        stats.iter().any(|stat| {
            stat.name == "bits"
                && stat.index_type == crate::types::IndexType::Flat
                && stat.state == "ready"
                && stat.document_count == 1
        }),
        "binary Flat must stay document-backed ready, got {stats:?}"
    );
}

#[test]
fn vector_index_classification_helpers_are_consistent() {
    use super::{
        builds_packed_vector_index, is_approximate_ann, is_in_memory_vector, is_incremental_vector,
    };
    use crate::types::IndexType;

    assert!(is_in_memory_vector(IndexType::Hnsw));
    assert!(is_in_memory_vector(IndexType::Flat));
    assert!(!is_in_memory_vector(IndexType::Invert));
    assert!(!is_in_memory_vector(IndexType::Fts));

    assert!(builds_packed_vector_index(
        IndexType::Hnsw,
        DataType::VectorFp32
    ));
    assert!(!builds_packed_vector_index(
        IndexType::Flat,
        DataType::VectorBinary32
    ));
    assert!(is_incremental_vector(IndexType::Hnsw, DataType::VectorFp32));
    assert!(!is_incremental_vector(
        IndexType::Flat,
        DataType::VectorFp32
    ));
    assert!(is_approximate_ann(IndexType::Ivf));
    assert!(!is_approximate_ann(IndexType::Flat));
}
