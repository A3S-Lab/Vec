use a3s_vec::{
    AddColumnOption, AlterColumnOption, Collection, CollectionHealth, CollectionHealthStatus,
    CollectionMaintenanceHealth, CollectionMaintenanceOptions, CollectionMaintenanceRuntime,
    CollectionOptions, CollectionResourceLimits, CollectionSchema, CollectionSchemaBuilder,
    CollectionStats, ConfigBuilder, DataType, DiskANNIndexParam, DiskAnnIndexParam,
    DiskannQueryParams, Doc, DocIterator, DocOperator, DocWriteResult, Durability, Error,
    ErrorCode, FieldSchema, FieldValue, FlatIndexParam, FlatQueryParams, Fts, FtsIndexParam,
    FtsQueryParams, GroupBySearchQuery, HnswIndexParam, HnswQueryParams, IVFIndexParam,
    IndexParams, IndexParamsBuilder, IndexStat, IndexType, InvertIndexParam, IoBackend,
    IvfIndexParam, IvfQueryParams, IvfRabitqIndexParam, IvfRabitqQueryParams, MetricType,
    MultiQuery, QuantizeType, RerankMethod, SearchQuery, SearchQueryBuilder, StatsSnapshot,
    SubQuery, VamanaIndexParam, VectorQuery, VectorSchema, VectorValue, WriteResult,
};
use tempfile::tempdir;

fn assert_send_sync<T: Send + Sync>() {}

fn text_schema(name: &str) -> CollectionSchema {
    CollectionSchema::builder(name)
        .add_field(
            FieldSchema::new("value", DataType::String, false, 0)
                .expect("field schema must be valid"),
        )
        .build()
        .expect("collection schema must be valid")
}

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

#[test]
fn process_configuration_lifecycle_controls_new_collections_only() {
    a3s_vec::shutdown().expect("resetting process configuration must succeed");
    assert!(!a3s_vec::is_initialized());
    let defaults = a3s_vec::default_config();
    let defaults_json = serde_json::to_value(&defaults).expect("defaults must serialize");
    assert_eq!(defaults_json["durability"], "Always");
    assert_eq!(defaults_json["io_backend"], "positioned");

    let mut options = CollectionOptions::new().expect("options must be constructible");
    assert!(!options.read_only());
    assert_eq!(options.durability(), None);
    assert_eq!(options.io_backend(), None);
    assert_eq!(options.resource_limits(), CollectionResourceLimits::new());
    options
        .set_read_only(true)
        .expect("read-only option must be settable");
    options
        .set_durability(Durability::Interval)
        .expect("durability option must be settable");
    options
        .set_io_backend(IoBackend::Mmap)
        .expect("I/O backend option must be settable");
    let limits = CollectionResourceLimits::new()
        .try_with_max_documents(8)
        .expect("resource limit must be valid");
    options
        .set_resource_limits(limits)
        .expect("resource policy must be settable");
    assert!(options.read_only());
    assert_eq!(options.durability(), Some(Durability::Interval));
    assert_eq!(options.io_backend(), Some(IoBackend::Mmap));
    assert_eq!(options.resource_limits(), limits);

    let configured = ConfigBuilder::new()
        .durability(Durability::Manual)
        .wal_max_ops(3)
        .wal_max_bytes(512)
        .io_backend(IoBackend::Mmap)
        .build();
    a3s_vec::initialize(Some(&configured))
        .expect("initializing process configuration must succeed");
    assert!(a3s_vec::is_initialized());

    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary.path().join("configured");
    let schema = text_schema("configuration-contract");
    let collection = Collection::create(
        path.to_str().expect("temporary path must be UTF-8"),
        &schema,
        None,
    )
    .expect("collection must be created");
    assert_eq!(
        collection.stats().expect("stats must succeed").io_backend,
        IoBackend::Mmap
    );
    collection.close().expect("collection must close");

    a3s_vec::shutdown().expect("shutting down process configuration must succeed");
    assert!(!a3s_vec::is_initialized());
    let defaulted_path = temporary.path().join("defaulted");
    let defaulted = Collection::create(
        defaulted_path
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema,
        None,
    )
    .expect("defaulted collection must be created");
    assert_eq!(
        defaulted.stats().expect("stats must succeed").io_backend,
        IoBackend::Positioned
    );
    defaulted.close().expect("defaulted collection must close");
}

#[test]
fn collection_lifecycle_aliases_and_closed_health_are_stable() {
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary.path().join("lifecycle");
    let schema = text_schema("lifecycle-contract");
    let collection = Collection::create_and_open(
        path.to_str().expect("temporary path must be UTF-8"),
        &schema,
        None,
    )
    .expect("create_and_open must create a collection");
    assert!(collection.is_open());
    assert_eq!(collection.path(), path);
    assert_eq!(collection.count().expect("count must succeed"), 0);
    let mut iterator = collection
        .iter_with_options(Some(&["value"]), false)
        .expect("iterator must be created");
    assert_eq!(iterator.revision(), 0);
    assert!(iterator.next().is_none());
    assert_eq!(
        collection.health().expect("health must succeed").status,
        CollectionHealthStatus::Healthy
    );

    let observer = collection.clone();
    collection.close().expect("close must succeed");
    assert!(!observer.is_open());
    let closed = observer
        .health()
        .expect("closed health must remain observable");
    assert_eq!(closed.status, CollectionHealthStatus::Closed);
    assert!(!closed.is_healthy());
    assert_eq!(
        observer
            .schema()
            .expect_err("schema access after close must fail")
            .code,
        ErrorCode::FailedPrecondition
    );
    observer
        .destroy()
        .expect("destroy must release the lock and remove the collection");
    assert!(!path.exists());
}
