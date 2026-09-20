#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::too_many_lines
)]

//! SearchQuery / query-control builder coverage.

use a3s_vec::{
    DiskannQueryParams, ErrorCode, FlatQueryParams, Fts, FtsQueryParams, HnswQueryParams,
    IvfQueryParams, IvfRabitqQueryParams, SearchQuery, SearchQueryBuilder,
};

#[test]
fn query_control_structs_validate_setters() {
    let mut hnsw = HnswQueryParams::new(32, 0.0, false, false);
    assert_eq!(hnsw.ef(), 32);
    hnsw.set_ef(64).expect("ef");
    assert!(hnsw.set_ef(0).is_err());

    let mut ivf = IvfQueryParams::new(8, false, 1.0);
    ivf.set_nprobe(16).expect("nprobe");
    assert!(ivf.set_nprobe(0).is_err());

    let mut rabitq = IvfRabitqQueryParams::new(8, 0.0, false, true);
    rabitq.set_nprobe(4).expect("nprobe");
    assert!(rabitq.set_nprobe(0).is_err());

    let flat = FlatQueryParams::new(false, 1.5);
    assert!(!flat.is_using_refiner);

    let diskann = DiskannQueryParams::new(64);
    assert_eq!(diskann.list_size, 64);

    let fts = FtsQueryParams::new(Some("and")).expect("fts");
    assert!(FtsQueryParams::new(Some("xor")).is_err());
    let _ = fts;
}

#[test]
fn search_query_builder_and_route_setters_cover_surface() {
    let mut dense = SearchQuery::new("embedding", &[1.0, 0.0], 5).expect("dense");
    dense.set_filter("bucket == 1").expect("filter");
    dense.set_include_vector(true).expect("include");
    dense.set_radius(0.5).expect("radius");
    dense
        .set_output_fields(&["embedding", "bucket"])
        .expect("fields");
    dense
        .set_hnsw_params(HnswQueryParams::new(16, 0.0, false, false))
        .expect("hnsw");
    dense
        .set_ivf_params(IvfQueryParams::new(4, false, 1.0))
        .expect("ivf");
    dense
        .set_ivf_rabitq_params(IvfRabitqQueryParams::new(4, 0.0, false, false))
        .expect("rabitq");
    dense
        .set_diskann_params(DiskannQueryParams::new(32))
        .expect("diskann");
    assert_eq!(
        dense
            .set_flat_params(FlatQueryParams::new(false, 1.0))
            .unwrap_err()
            .code,
        ErrorCode::NotSupported
    );

    let binary = SearchQuery::binary("bits", &[0; 4], 3).expect("binary");
    assert!(binary.binary_vector.is_some());
    assert!(SearchQuery::binary("bits", &[], 1).is_err());

    let sparse = SearchQuery::sparse("sparse", &[0, 1], &[1.0, 0.5], 3).expect("sparse");
    assert!(sparse.sparse_vector.is_some());
    assert!(SearchQuery::sparse("sparse", &[0], &[1.0, 2.0], 1).is_err());

    let mut fts = Fts::new().expect("fts");
    fts.set_query_string("alpha OR beta").expect("query");
    fts.set_match_string("alpha").expect("match");
    let fts_query = SearchQuery::fts("body", &fts, 5).expect("fts query");
    assert!(fts_query.fts.is_some());

    let by_id = SearchQuery::by_id("embedding", "doc-1", 3).expect("by id");
    assert_eq!(by_id.id.as_deref(), Some("doc-1"));

    let built = SearchQueryBuilder::new()
        .field_name("embedding")
        .vector(&[0.0, 1.0])
        .topk(4)
        .filter("bucket == 1")
        .include_vector(true)
        .include_doc_id(true)
        .output_fields(&["embedding"])
        .build()
        .expect("builder");
    assert_eq!(built.topk, 4);
    assert!(built.include_vector);

    let fts_built = SearchQueryBuilder::new()
        .field_name("body")
        .fts_match_string("alpha")
        .topk(3)
        .build()
        .expect("fts builder");
    assert!(fts_built.fts.is_some());

    let fts_query_built = SearchQueryBuilder::new()
        .field_name("body")
        .fts_query_string("alpha OR beta")
        .topk(3)
        .build()
        .expect("fts query builder");
    assert!(fts_query_built.fts.is_some());

    let binary_built = SearchQueryBuilder::new()
        .field_name("bits")
        .binary_vector(&[0; 4])
        .topk(2)
        .build()
        .expect("binary builder");
    assert!(binary_built.binary_vector.is_some());

    assert!(SearchQueryBuilder::new()
        .field_name("embedding")
        .vector(&[1.0])
        .fts_match_string("x")
        .build()
        .is_err());
    assert!(SearchQueryBuilder::new()
        .field_name("body")
        .fts_match_string("a")
        .fts_query_string("b")
        .build()
        .is_err());
    assert!(SearchQueryBuilder::new().build().is_err());
}

#[test]
fn search_query_mutators_and_getters_cover_remaining_surface() {
    let mut dense = SearchQuery::new("embedding", &[1.0, 0.0], 5).expect("dense");
    dense.set_field_name("vector").expect("rename");
    assert!(dense.set_field_name("").is_err());
    dense
        .set_query_vector(&[0.0, 1.0, f32::NAN])
        .expect_err("nan");
    dense.set_query_vector(&[0.0, 1.0]).expect("vector");
    dense.set_binary_vector(&[1, 2, 3]).expect("binary");
    assert!(dense.set_binary_vector(&[]).is_err());
    dense
        .set_sparse_vector(&[0, 1], &[1.0, 0.5])
        .expect("sparse");
    assert!(dense.set_sparse_vector(&[0], &[1.0, 2.0]).is_err());
    dense.set_filter("").expect("clear filter");
    assert!(dense.get_filter().is_none());
    dense.set_filter("x == 1").expect("filter");
    assert_eq!(dense.get_filter(), Some("x == 1"));
    assert!(dense.set_topk(0).is_err());
    dense.set_topk(7).expect("topk");
    dense.set_include_doc_id(true).expect("doc id");
    assert!(dense.set_output_fields(&[""]).is_err());
    dense.set_include_vector(false).expect("include");
    assert!(dense.has_vector());

    let mut ivf = IvfQueryParams::new(8, false, 1.0);
    assert!(ivf.set_scale_factor(0.0).is_err());
    assert!(ivf.set_scale_factor(f32::NAN).is_err());
    ivf.set_scale_factor(2.5).expect("scale");
    assert_eq!(ivf.scale_factor(), 2.5);
    assert_eq!(ivf.nprobe(), 8);
    ivf.set_nprobe(12).expect("nprobe");
    assert_eq!(ivf.nprobe(), 12);

    let mut rabitq = IvfRabitqQueryParams::new(8, 0.0, false, true);
    assert!(rabitq.set_scale_factor(-1.0).is_err());
    rabitq.set_scale_factor(3.0).expect("scale");
    assert_eq!(rabitq.scale_factor(), 3.0);
    assert_eq!(rabitq.nprobe(), 8);
    rabitq.set_nprobe(5).expect("nprobe");
    assert_eq!(rabitq.nprobe(), 5);
    assert!(rabitq.set_nprobe(0).is_err());

    let mut diskann = DiskannQueryParams::new(64);
    assert!(diskann.set_list_size(0).is_err());
    diskann.set_list_size(128).expect("list");
    assert_eq!(diskann.list_size(), 128);

    dense
        .set_fts_params(FtsQueryParams::new(Some("and")).expect("fts"))
        .expect("fts params");
    let mut fts = Fts::new().expect("fts");
    fts.set_query_string("alpha").expect("q");
    dense.set_fts(&fts).expect("fts");
    assert!(!dense.has_vector());

    assert!(SearchQuery::new("", &[1.0], 1).is_err());
    assert!(SearchQuery::new("embedding", &[], 1).is_err());
    assert!(SearchQuery::new("embedding", &[1.0], 0).is_err());
    assert!(dense.set_radius(f32::INFINITY).is_err());

    let via_builder = SearchQuery::builder()
        .field_name("embedding")
        .vector(&[1.0, 0.0])
        .topk(2)
        .build()
        .expect("builder via SearchQuery::builder");
    assert_eq!(via_builder.topk, 2);

    let mut grouped =
        a3s_vec::GroupBySearchQuery::new("embedding", "tag", &[1.0, 0.0], 2, 2).expect("groupby");
    grouped.set_filter("tag == \"a\"").expect("filter");
    grouped.set_include_vector(true).expect("include");
    grouped.set_output_fields(&["tag"]).expect("fields");
    assert!(grouped.set_output_fields(&[""]).is_err());
    grouped
        .set_hnsw_params(HnswQueryParams::new(16, 0.0, false, false))
        .expect("hnsw");
    grouped
        .set_ivf_params(IvfQueryParams::new(4, false, 1.0))
        .expect("ivf");
    grouped
        .set_ivf_rabitq_params(IvfRabitqQueryParams::new(4, 0.0, false, false))
        .expect("rabitq");
    assert_eq!(
        grouped
            .set_flat_params(FlatQueryParams::new(false, 1.0))
            .unwrap_err()
            .code,
        ErrorCode::NotSupported
    );
    grouped
        .set_diskann_params(DiskannQueryParams::new(32))
        .expect("diskann");

    let mut fts = Fts::new().expect("fts");
    assert!(fts.set_query_string("").is_err());
    assert!(fts.set_match_string("").is_err());
    fts.set_match_string("alpha").expect("match");
    assert!(FtsQueryParams::new(Some("xor")).is_err());
    let mut fts_params = FtsQueryParams::new(Some("and")).expect("op");
    assert!(fts_params.set_default_operator("xor").is_err());
    fts_params.set_default_operator("or").expect("or");
    assert_eq!(fts_params.default_operator().as_deref(), Some("or"));

    assert!(a3s_vec::GroupBySearchQuery::new("embedding", "tag", &[1.0], 0, 1).is_err());
    assert!(a3s_vec::GroupBySearchQuery::new("embedding", "tag", &[1.0], 1, 0).is_err());
    assert!(a3s_vec::GroupBySearchQuery::new("embedding", "tag", &[], 1, 1).is_err());
    assert!(a3s_vec::GroupBySearchQuery::new("embedding", "tag", &[f32::NAN], 1, 1).is_err());
    assert!(a3s_vec::GroupBySearchQuery::binary("bits", "tag", &[], 1, 1).is_err());

    let mut empty_fts = SearchQuery::new("embedding", &[1.0, 0.0], 3).expect("q");
    assert!(empty_fts.set_fts(&Fts::new().expect("empty")).is_err());
}
