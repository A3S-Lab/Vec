use a3s_vec::{
    AddColumnOption, AlterColumnOption, Collection, CollectionHealth, CollectionMaintenanceHealth,
    CollectionMaintenanceOptions, CollectionMaintenanceRuntime, CollectionOptions,
    CollectionResourceLimits, CollectionSchema, CollectionSchemaBuilder, CollectionStats,
    ConfigBuilder, DataType, DiskANNIndexParam, DiskAnnIndexParam, DiskannQueryParams, Doc,
    DocIterator, DocOperator, DocWriteResult, Durability, Error, ErrorCode, FieldSchema,
    FieldValue, FlatIndexParam, FlatQueryParams, Fts, FtsIndexParam, FtsQueryParams,
    GroupBySearchQuery, HnswIndexParam, HnswQueryParams, IVFIndexParam, IndexParams,
    IndexParamsBuilder, IndexStat, IndexType, InvertIndexParam, IoBackend, IvfIndexParam,
    IvfQueryParams, IvfRabitqIndexParam, IvfRabitqQueryParams, MetricType, MultiQuery,
    QuantizeType, RerankMethod, SearchQuery, SearchQueryBuilder, StatsSnapshot, SubQuery,
    VamanaIndexParam, VectorQuery, VectorSchema, VectorValue, WriteResult,
};

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn public_owned_contracts_are_send_and_sync() {
    assert_send_sync::<AddColumnOption>();
    assert_send_sync::<AlterColumnOption>();
    assert_send_sync::<Collection>();
    assert_send_sync::<CollectionHealth>();
    assert_send_sync::<CollectionMaintenanceHealth>();
    assert_send_sync::<CollectionMaintenanceOptions>();
    assert_send_sync::<CollectionMaintenanceRuntime>();
    assert_send_sync::<CollectionOptions>();
    assert_send_sync::<CollectionResourceLimits>();
    assert_send_sync::<CollectionSchema>();
    assert_send_sync::<CollectionSchemaBuilder>();
    assert_send_sync::<CollectionStats>();
    assert_send_sync::<ConfigBuilder>();
    assert_send_sync::<DataType>();
    assert_send_sync::<DiskANNIndexParam>();
    assert_send_sync::<DiskAnnIndexParam>();
    assert_send_sync::<DiskannQueryParams>();
    assert_send_sync::<Doc>();
    assert_send_sync::<DocIterator>();
    assert_send_sync::<DocOperator>();
    assert_send_sync::<DocWriteResult>();
    assert_send_sync::<Durability>();
    assert_send_sync::<Error>();
    assert_send_sync::<ErrorCode>();
    assert_send_sync::<FieldSchema>();
    assert_send_sync::<FieldValue>();
    assert_send_sync::<FlatIndexParam>();
    assert_send_sync::<FlatQueryParams>();
    assert_send_sync::<Fts>();
    assert_send_sync::<FtsIndexParam>();
    assert_send_sync::<FtsQueryParams>();
    assert_send_sync::<GroupBySearchQuery>();
    assert_send_sync::<HnswIndexParam>();
    assert_send_sync::<HnswQueryParams>();
    assert_send_sync::<IVFIndexParam>();
    assert_send_sync::<IndexParams>();
    assert_send_sync::<IndexParamsBuilder>();
    assert_send_sync::<IndexStat>();
    assert_send_sync::<IndexType>();
    assert_send_sync::<InvertIndexParam>();
    assert_send_sync::<IoBackend>();
    assert_send_sync::<IvfIndexParam>();
    assert_send_sync::<IvfQueryParams>();
    assert_send_sync::<IvfRabitqIndexParam>();
    assert_send_sync::<IvfRabitqQueryParams>();
    assert_send_sync::<MetricType>();
    assert_send_sync::<MultiQuery>();
    assert_send_sync::<QuantizeType>();
    assert_send_sync::<RerankMethod>();
    assert_send_sync::<SearchQuery>();
    assert_send_sync::<SearchQueryBuilder>();
    assert_send_sync::<StatsSnapshot>();
    assert_send_sync::<SubQuery>();
    assert_send_sync::<VamanaIndexParam>();
    assert_send_sync::<VectorQuery>();
    assert_send_sync::<VectorSchema>();
    assert_send_sync::<VectorValue>();
    assert_send_sync::<WriteResult>();
}

#[test]
fn public_version_functions_match_the_package_version() {
    let expected = env!("CARGO_PKG_VERSION");
    let components = expected
        .split('.')
        .map(|value| {
            value
                .parse::<i32>()
                .expect("package version must be numeric")
        })
        .collect::<Vec<_>>();
    assert_eq!(components.len(), 3, "package version must be semver-shaped");
    assert_eq!(a3s_vec::version(), expected);
    assert_eq!(
        (
            a3s_vec::version_major(),
            a3s_vec::version_minor(),
            a3s_vec::version_patch(),
        ),
        (components[0], components[1], components[2])
    );
    assert!(a3s_vec::check_version(
        components[0],
        components[1],
        components[2]
    ));
    assert!(!a3s_vec::check_version(
        components[0],
        components[1],
        components[2]
            .checked_add(1)
            .expect("package patch version must be incrementable")
    ));
}
