use a3s_vec::{
    Collection, CollectionSchema, DataType, Doc, ErrorCode, FieldSchema, GroupBySearchQuery,
    IndexParams, IndexType, MetricType, MultiQuery, SearchQuery, SubQuery,
};
use serde_json::json;
use std::collections::HashMap;
use tempfile::tempdir;

fn schema() -> CollectionSchema {
    let mut category =
        FieldSchema::new("category", DataType::String, false, 0).expect("category schema");
    category
        .set_index_params(&IndexParams::invert(true, true).expect("scalar index"))
        .expect("category index");
    let mut bits32 =
        FieldSchema::new("bits32", DataType::VectorBinary32, false, 32).expect("Binary32 schema");
    bits32
        .set_index_params(&IndexParams::flat(MetricType::L2).expect("Flat L2 index"))
        .expect("Binary32 Flat index");
    CollectionSchema::builder("binary-query-contract")
        .add_field(category)
        .add_field(bits32)
        .add_field(
            FieldSchema::new("bits64", DataType::VectorBinary64, false, 64)
                .expect("Binary64 schema"),
        )
        .add_field(FieldSchema::new("dense", DataType::VectorFp32, false, 2).expect("dense schema"))
        .build()
        .expect("binary query schema")
}

fn fixture_doc(id: &str, category: &str, bits32: [u8; 4], bits64: [u8; 8]) -> Doc {
    let mut doc = Doc::with_pk(id).expect("document id");
    doc.add_string("category", category).expect("category");
    doc.add_vector_binary32("bits32", &bits32)
        .expect("Binary32 payload");
    doc.add_vector_binary64("bits64", &bits64)
        .expect("Binary64 payload");
    doc.add_vector_f32("dense", &[0.0, 1.0])
        .expect("dense payload");
    doc
}

fn fixture_docs() -> Vec<Doc> {
    vec![
        fixture_doc("doc-0", "a", [0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]),
        fixture_doc("doc-1", "a", [1, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]),
        fixture_doc("doc-2", "b", [3, 0, 0, 0], [3, 0, 0, 0, 0, 0, 0, 0]),
        fixture_doc(
            "doc-3",
            "b",
            [u8::MAX, 0, 0, 0],
            [u8::MAX, 0, 0, 0, 0, 0, 0, 0],
        ),
        fixture_doc(
            "doc-4",
            "c",
            [0, u8::MAX, 0, 0],
            [0, u8::MAX, 0, 0, 0, 0, 0, 0],
        ),
    ]
}

fn ids(docs: &[Doc]) -> Vec<&str> {
    docs.iter()
        .map(|doc| doc.get_pk().expect("query result id"))
        .collect()
}

fn ranking(docs: &[Doc]) -> Vec<(String, u32)> {
    docs.iter()
        .map(|doc| {
            (
                doc.get_pk().expect("query result id").to_string(),
                doc.get_score().to_bits(),
            )
        })
        .collect()
}

fn grouped_ids(groups: &HashMap<String, Vec<Doc>>) -> HashMap<String, Vec<String>> {
    groups
        .iter()
        .map(|(key, docs)| {
            (
                key.clone(),
                docs.iter()
                    .map(|doc| doc.get_pk().expect("group result id").to_string())
                    .collect(),
            )
        })
        .collect()
}

#[test]
#[allow(clippy::too_many_lines)]
fn binary_exact_routes_are_typed_deterministic_and_persistent() {
    let temporary = tempdir().expect("temporary directory");
    let path = temporary.path().join("binary");
    let path_string = path.to_str().expect("UTF-8 collection path");
    let collection = Collection::create(path_string, &schema(), None).expect("create collection");
    let docs = fixture_docs();
    let inserted = collection
        .insert(&docs.iter().collect::<Vec<_>>())
        .expect("insert fixture");
    assert_eq!(inserted.success_count, 5);

    let mut direct =
        SearchQuery::binary("bits32", &[0, 0, 0, 0], 5).expect("Binary32 query must be valid");
    direct.set_include_vector(true).expect("include vector");
    direct
        .set_include_doc_id(true)
        .expect("include document id");
    direct
        .set_output_fields(&["category", "bits32"])
        .expect("output projection");
    let direct_hits = collection.query(&direct).expect("Binary32 exact query");
    assert_eq!(
        ids(&direct_hits),
        ["doc-0", "doc-1", "doc-2", "doc-3", "doc-4"]
    );
    assert_eq!(
        direct_hits.iter().map(Doc::get_score).collect::<Vec<_>>(),
        [0.0, -1.0, -2.0, -8.0, -8.0]
    );
    assert!(direct_hits.iter().all(|doc| doc.doc_id().is_some()));
    assert!(direct_hits.iter().all(|doc| doc
        .get_vector_binary32("bits32")
        .expect("Binary32 getter")
        .is_some()));
    assert!(direct_hits
        .iter()
        .all(|doc| doc.vector("bits64").is_none() && doc.vector("dense").is_none()));

    let mut filtered =
        SearchQuery::binary("bits32", &[0, 0, 0, 0], 5).expect("filtered binary query");
    filtered
        .set_filter("category == 'b'")
        .expect("binary scalar filter");
    assert_eq!(
        ids(&collection.query(&filtered).expect("filtered binary query")),
        ["doc-2", "doc-3"]
    );

    let mut radius = SearchQuery::binary("bits32", &[0, 0, 0, 0], 5).expect("radius binary query");
    radius.set_radius(1.0).expect("L2 radius");
    assert_eq!(
        ids(&collection.query(&radius).expect("binary radius query")),
        ["doc-0", "doc-1"]
    );

    let source = SearchQuery::by_id("bits32", "doc-0", 5).expect("source-id query");
    assert_eq!(
        ranking(&collection.query(&source).expect("binary source-id query")),
        ranking(
            &collection
                .query(&SearchQuery::binary("bits32", &[0, 0, 0, 0], 5).unwrap())
                .unwrap()
        )
    );

    let built = SearchQuery::builder()
        .field_name("bits32")
        .binary_vector(&[0, 0, 0, 0])
        .topk(3)
        .build()
        .expect("binary builder route");
    assert_eq!(
        ids(&collection.query(&built).expect("built binary query")),
        ["doc-0", "doc-1", "doc-2"]
    );
    let encoded = serde_json::to_string(&built).expect("serialize binary query");
    let decoded: SearchQuery = serde_json::from_str(&encoded).expect("deserialize binary query");
    assert_eq!(decoded, built);

    let mut branch = SubQuery::new().expect("binary sub-query");
    branch.set_field_name("bits32").expect("sub-query field");
    branch
        .set_binary_vector(&[0, 0, 0, 0])
        .expect("sub-query binary vector");
    branch.set_num_candidates(5).expect("candidate count");
    let encoded = serde_json::to_string(&branch).expect("serialize binary sub-query");
    let decoded: SubQuery = serde_json::from_str(&encoded).expect("deserialize binary sub-query");
    assert_eq!(decoded, branch);
    let mut multi = MultiQuery::new().expect("multi-query");
    multi.add_sub_query(&branch).expect("add binary branch");
    multi.set_topk(3).expect("multi top-k");
    let multi_hits = collection.multi_query(&multi).expect("binary multi-query");
    assert_eq!(ids(&multi_hits), ["doc-0", "doc-1", "doc-2"]);

    let grouped = GroupBySearchQuery::binary("bits32", "category", &[0, 0, 0, 0], 2, 2)
        .expect("binary group-by query");
    let encoded = serde_json::to_string(&grouped).expect("serialize binary group-by query");
    let decoded: GroupBySearchQuery =
        serde_json::from_str(&encoded).expect("deserialize binary group-by query");
    assert_eq!(decoded, grouped);
    let groups = collection.group_by(&grouped).expect("binary group-by");
    assert_eq!(
        grouped_ids(&groups).get("a"),
        Some(&vec!["doc-0".to_string(), "doc-1".to_string()])
    );
    assert_eq!(
        grouped_ids(&groups).get("b"),
        Some(&vec!["doc-2".to_string(), "doc-3".to_string()])
    );

    let binary64 = SearchQuery::binary("bits64", &[0; 8], 5).expect("Binary64 query");
    let binary64_hits = collection.query(&binary64).expect("Binary64 exact query");
    assert_eq!(ids(&binary64_hits), ids(&direct_hits));
    assert_eq!(
        binary64_hits.iter().map(Doc::get_score).collect::<Vec<_>>(),
        [0.0, -1.0, -2.0, -8.0, -8.0]
    );

    let stats = collection.stats().expect("collection stats");
    let flat = stats
        .indexes
        .iter()
        .find(|index| index.name == "bits32")
        .expect("Binary32 Flat stats");
    assert_eq!(flat.index_type, IndexType::Flat);
    assert_eq!(flat.state, "ready");
    assert_eq!(flat.document_count, 5);

    #[cfg(feature = "async")]
    assert_async_parity(&collection, &direct, &multi, &grouped);

    let expected = ranking(&direct_hits);
    collection.flush().expect("flush collection");
    collection.close().expect("close collection");
    let reopened = Collection::open(path_string, None).expect("reopen collection");
    assert_eq!(
        ranking(&reopened.query(&direct).expect("reopened binary query")),
        expected
    );
    reopened.close().expect("close reopened collection");
}

#[cfg(feature = "async")]
fn assert_async_parity(
    collection: &Collection,
    query: &SearchQuery,
    multi: &MultiQuery,
    grouped: &GroupBySearchQuery,
) {
    let sync = ranking(&collection.query(query).expect("sync binary query"));
    let sync_multi = ids(&collection
        .multi_query(multi)
        .expect("sync binary multi-query"))
    .into_iter()
    .map(str::to_string)
    .collect::<Vec<_>>();
    let sync_groups = grouped_ids(&collection.group_by(grouped).expect("sync binary group-by"));
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("Tokio runtime");
    let (actual, actual_multi, actual_groups) = runtime.block_on(async {
        (
            collection
                .query_async(query)
                .await
                .expect("async binary query"),
            collection
                .multi_query_async(multi)
                .await
                .expect("async binary multi-query"),
            collection
                .group_by_async(grouped)
                .await
                .expect("async binary group-by"),
        )
    });
    assert_eq!(ranking(&actual), sync);
    assert_eq!(
        ids(&actual_multi)
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>(),
        sync_multi
    );
    assert_eq!(grouped_ids(&actual_groups), sync_groups);
}

#[test]
#[allow(clippy::too_many_lines)]
fn binary_query_contract_rejects_ambiguous_mismatched_and_ann_payloads() {
    assert_eq!(
        SearchQuery::binary("bits32", &[], 1)
            .expect_err("empty binary query")
            .code,
        ErrorCode::InvalidArgument
    );

    let temporary = tempdir().expect("temporary directory");
    let path = temporary.path().join("contracts");
    let collection = Collection::create(
        path.to_str().expect("UTF-8 collection path"),
        &schema(),
        None,
    )
    .expect("create collection");
    let doc = fixture_docs().remove(0);
    collection.insert(&[&doc]).expect("insert document");

    let wrong_length = SearchQuery::binary("bits32", &[0, 0, 0], 1).expect("typed query");
    let error = collection
        .query(&wrong_length)
        .expect_err("binary byte length mismatch");
    assert_eq!(error.code, ErrorCode::InvalidArgument);
    assert!(error.message.contains("expected 4, got 3"));

    let dense = SearchQuery::new("bits32", &[0.0; 32], 1).expect("dense query");
    assert_eq!(
        collection
            .query(&dense)
            .expect_err("dense payload on binary field")
            .code,
        ErrorCode::InvalidArgument
    );
    let binary_on_dense = SearchQuery::binary("dense", &[0, 0], 1).expect("binary query");
    assert_eq!(
        collection
            .query(&binary_on_dense)
            .expect_err("binary payload on dense field")
            .code,
        ErrorCode::InvalidArgument
    );

    let mut cosine = SearchQuery::binary("bits32", &[0; 4], 1).expect("binary query");
    cosine.params.insert("metric".into(), json!("cosine"));
    let error = collection
        .query(&cosine)
        .expect_err("binary cosine must be rejected");
    assert_eq!(error.code, ErrorCode::NotSupported);

    let mut ambiguous = SearchQuery::binary("bits32", &[0; 4], 1).expect("binary query");
    ambiguous.vector = Some(vec![0.0; 32]);
    assert_eq!(
        collection
            .query(&ambiguous)
            .expect_err("ambiguous query routes")
            .code,
        ErrorCode::InvalidArgument
    );

    let error = SearchQuery::builder()
        .field_name("bits32")
        .vector(&[0.0; 32])
        .binary_vector(&[0; 4])
        .build()
        .expect_err("builder must reject ambiguous routes");
    assert_eq!(error.code, ErrorCode::InvalidArgument);

    let mut ambiguous_branch = SubQuery::new().expect("sub-query");
    ambiguous_branch
        .set_field_name("bits32")
        .expect("sub-query field");
    ambiguous_branch
        .set_binary_vector(&[0; 4])
        .expect("binary sub-query");
    ambiguous_branch.vector = Some(vec![0.0; 32]);
    let mut ambiguous_multi = MultiQuery::new().expect("multi-query");
    ambiguous_multi
        .add_sub_query(&ambiguous_branch)
        .expect("add ambiguous branch payload");
    assert_eq!(
        collection
            .multi_query(&ambiguous_multi)
            .expect_err("ambiguous sub-query routes")
            .code,
        ErrorCode::InvalidArgument
    );

    let mut ambiguous_group = GroupBySearchQuery::binary("bits32", "category", &[0; 4], 1, 1)
        .expect("binary group query");
    ambiguous_group.vector = vec![0.0; 32];
    assert_eq!(
        collection
            .group_by(&ambiguous_group)
            .expect_err("ambiguous group-by routes")
            .code,
        ErrorCode::InvalidArgument
    );

    let mut tuned = SearchQuery::binary("bits32", &[0; 4], 1).expect("binary query");
    tuned.params.insert("type".into(), json!("hnsw"));
    assert_eq!(
        collection
            .query(&tuned)
            .expect_err("binary ANN controls")
            .code,
        ErrorCode::NotSupported
    );
    assert_eq!(
        collection
            .query(&SearchQuery::by_id("bits32", "missing", 1).expect("source-id query"))
            .expect_err("missing binary source")
            .code,
        ErrorCode::NotFound
    );

    let mut binary_field =
        FieldSchema::new("bits", DataType::VectorBinary32, false, 32).expect("binary field");
    let error = binary_field
        .set_index_params(&IndexParams::flat(MetricType::Cosine).expect("Flat descriptor"))
        .expect_err("binary Flat cosine must be rejected");
    assert_eq!(error.code, ErrorCode::NotSupported);
    let error = binary_field
        .set_index_params(&IndexParams::hnsw(MetricType::L2, 8, 16).expect("HNSW descriptor"))
        .expect_err("binary ANN index must remain unsupported");
    assert_eq!(error.code, ErrorCode::NotSupported);
}

fn next_u64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn random_bytes(state: &mut u64, length: usize) -> Vec<u8> {
    (0..length)
        .map(|_| next_u64(state).to_le_bytes()[0])
        .collect()
}

fn hamming(left: &[u8], right: &[u8]) -> u32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left ^ right).count_ones())
        .sum()
}

#[test]
fn binary32_and_binary64_match_an_independent_hamming_oracle() {
    let temporary = tempdir().expect("temporary directory");
    let path = temporary.path().join("differential");
    let collection = Collection::create(
        path.to_str().expect("UTF-8 collection path"),
        &schema(),
        None,
    )
    .expect("create collection");
    let mut state = 0x7ca5_19e3_d42b_608f;
    let mut payloads = Vec::new();
    let mut docs = Vec::new();
    for index in 0..129 {
        let bits32 = random_bytes(&mut state, 4);
        let bits64 = random_bytes(&mut state, 8);
        let mut doc = Doc::with_pk(format!("doc-{index:03}")).expect("document id");
        doc.add_string("category", if index % 2 == 0 { "a" } else { "b" })
            .expect("category");
        doc.add_vector_binary32("bits32", &bits32)
            .expect("Binary32 payload");
        doc.add_vector_binary64("bits64", &bits64)
            .expect("Binary64 payload");
        doc.add_vector_f32("dense", &[0.0, 1.0])
            .expect("dense payload");
        payloads.push((format!("doc-{index:03}"), bits32, bits64));
        docs.push(doc);
    }
    collection
        .insert(&docs.iter().collect::<Vec<_>>())
        .expect("insert differential corpus");

    for query_index in 0..32 {
        for (field, byte_length, payload_index) in [("bits32", 4, 1), ("bits64", 8, 2)] {
            let query_bytes = if query_index % 3 == 0 {
                match payload_index {
                    1 => payloads[query_index].1.clone(),
                    2 => payloads[query_index].2.clone(),
                    _ => unreachable!(),
                }
            } else {
                random_bytes(&mut state, byte_length)
            };
            let mut expected = payloads
                .iter()
                .map(|(id, bits32, bits64)| {
                    let stored = if payload_index == 1 { bits32 } else { bits64 };
                    (id.clone(), hamming(&query_bytes, stored))
                })
                .collect::<Vec<_>>();
            expected.sort_unstable_by(|left, right| {
                left.1.cmp(&right.1).then_with(|| left.0.cmp(&right.0))
            });
            expected.truncate(17);
            let expected = expected
                .into_iter()
                .map(|(id, distance)| {
                    let distance = u8::try_from(distance).expect("Hamming distance fits u8");
                    (id, (-f32::from(distance)).to_bits())
                })
                .collect::<Vec<_>>();
            let actual = collection
                .query(&SearchQuery::binary(field, &query_bytes, 17).expect("binary query"))
                .expect("binary differential query");
            assert_eq!(
                ranking(&actual),
                expected,
                "field={field}, query_index={query_index}"
            );
        }
    }
}
