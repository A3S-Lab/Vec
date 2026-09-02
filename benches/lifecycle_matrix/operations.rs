/// Runs one isolated sample for each management-plane operation.
#[allow(clippy::too_many_lines)]
fn run_lifecycle_matrix() {
    let config = Config::from_environment();
    println!(
        "operation,documents,dimensions,samples,total_work,p50_us,p95_us,p99_us,work_per_second,work_per_sample"
    );

    measure("create_collection", config, |_| {
        let temporary = tempdir().expect("temporary directory must be available");
        let started = Instant::now();
        let collection = Collection::create(
            &create_path(temporary.path(), "collection"),
            &schema_with_dimension("lifecycle-create", config.dimensions, false, false),
            Some(&manual_options()),
        )
        .expect("collection creation must succeed");
        let elapsed = started.elapsed();
        collection.close().expect("collection close must succeed");
        Sample { elapsed, work: 1 }
    });

    measure("insert_batch", config, |_| {
        let temporary = tempdir().expect("temporary directory must be available");
        let collection = Collection::create(
            &create_path(temporary.path(), "collection"),
            &schema_with_dimension("lifecycle-insert", config.dimensions, false, false),
            Some(&manual_options()),
        )
        .expect("collection creation must succeed");
        let documents = (0..config.documents)
            .map(|index| document(index, config.dimensions))
            .collect::<Vec<_>>();
        let references: Vec<&Doc> = documents.iter().collect();
        let started = Instant::now();
        let result = collection
            .insert(&references)
            .expect("batch insert must succeed");
        let elapsed = started.elapsed();
        assert_eq!(result.success_count, as_u64(config.documents));
        collection.close().expect("collection close must succeed");
        Sample {
            elapsed,
            work: result.success_count,
        }
    });

    measure("update", config, |_| {
        let temporary = tempdir().expect("temporary directory must be available");
        let collection = Collection::create(
            &create_path(temporary.path(), "collection"),
            &schema_with_dimension("lifecycle-update", config.dimensions, false, false),
            Some(&manual_options()),
        )
        .expect("collection creation must succeed");
        let documents = insert_fixture(&collection, config, one_document());
        let mut patch =
            Doc::with_pk(documents[0].get_pk().expect("fixture id")).expect("patch must be valid");
        patch.add_i32("bucket", 15).expect("patch must be valid");
        let started = Instant::now();
        let result = collection.update(&[&patch]).expect("update must succeed");
        let elapsed = started.elapsed();
        assert_eq!(result.success_count, 1);
        collection.close().expect("collection close must succeed");
        Sample { elapsed, work: 1 }
    });

    measure("upsert", config, |sample| {
        let temporary = tempdir().expect("temporary directory must be available");
        let collection = Collection::create(
            &create_path(temporary.path(), "collection"),
            &schema_with_dimension("lifecycle-upsert", config.dimensions, false, false),
            Some(&manual_options()),
        )
        .expect("collection creation must succeed");
        let _ = insert_fixture(&collection, config, one_document());
        let upsert = document(config.documents + slot(sample) + 1, config.dimensions);
        let started = Instant::now();
        let result = collection.upsert(&[&upsert]).expect("upsert must succeed");
        let elapsed = started.elapsed();
        assert_eq!(result.success_count, 1);
        collection.close().expect("collection close must succeed");
        Sample { elapsed, work: 1 }
    });

    measure("delete", config, |_| {
        let temporary = tempdir().expect("temporary directory must be available");
        let collection = Collection::create(
            &create_path(temporary.path(), "collection"),
            &schema_with_dimension("lifecycle-delete", config.dimensions, false, false),
            Some(&manual_options()),
        )
        .expect("collection creation must succeed");
        let documents = insert_fixture(&collection, config, one_document());
        let id = documents[0].get_pk().expect("fixture id");
        let started = Instant::now();
        let result = collection.delete(&[id]).expect("delete must succeed");
        let elapsed = started.elapsed();
        assert_eq!(result.success_count, 1);
        collection.close().expect("collection close must succeed");
        Sample { elapsed, work: 1 }
    });

    measure("delete_by_filter", config, |_| {
        let temporary = tempdir().expect("temporary directory must be available");
        let collection = Collection::create(
            &create_path(temporary.path(), "collection"),
            &schema_with_dimension("lifecycle-filter-delete", config.dimensions, false, true),
            Some(&manual_options()),
        )
        .expect("collection creation must succeed");
        let count = config.documents.max(2);
        let _ = insert_fixture(&collection, config, count);
        let expected = (0..count).filter(|index| index % 2 == 0).count();
        let started = Instant::now();
        collection
            .delete_by_filter("category == 'even'")
            .expect("filtered delete must succeed");
        let elapsed = started.elapsed();
        assert_eq!(
            collection.count().expect("count must succeed"),
            count - expected
        );
        collection.close().expect("collection close must succeed");
        Sample {
            elapsed,
            work: u64::try_from(expected).expect("delete count fits u64"),
        }
    });

    measure("create_index", config, |_| {
        let temporary = tempdir().expect("temporary directory must be available");
        let collection = Collection::create(
            &create_path(temporary.path(), "collection"),
            &schema_with_dimension("lifecycle-create-index", config.dimensions, false, false),
            Some(&manual_options()),
        )
        .expect("collection creation must succeed");
        let _ = insert_fixture(&collection, config, config.documents);
        let params = IndexParams::hnsw(MetricType::Cosine, 16, 96).expect("HNSW params");
        let started = Instant::now();
        collection
            .create_index("embedding", &params)
            .expect("index creation must succeed");
        let elapsed = started.elapsed();
        assert!(!collection
            .stats()
            .expect("stats must succeed")
            .indexes
            .is_empty());
        collection.close().expect("collection close must succeed");
        Sample {
            elapsed,
            work: as_u64(config.documents),
        }
    });

    measure("drop_index", config, |_| {
        let temporary = tempdir().expect("temporary directory must be available");
        let collection = Collection::create(
            &create_path(temporary.path(), "collection"),
            &schema_with_dimension("lifecycle-drop-index", config.dimensions, true, false),
            Some(&manual_options()),
        )
        .expect("collection creation must succeed");
        let _ = insert_fixture(&collection, config, config.documents);
        let started = Instant::now();
        collection
            .drop_index("embedding")
            .expect("index drop must succeed");
        let elapsed = started.elapsed();
        assert!(!collection
            .schema()
            .expect("schema must succeed")
            .has_index("embedding"));
        collection.close().expect("collection close must succeed");
        Sample {
            elapsed,
            work: as_u64(config.documents),
        }
    });

    measure("rebuild_index", config, |_| {
        let temporary = tempdir().expect("temporary directory must be available");
        let collection = Collection::create(
            &create_path(temporary.path(), "collection"),
            &schema_with_dimension("lifecycle-rebuild-index", config.dimensions, true, false),
            Some(&manual_options()),
        )
        .expect("collection creation must succeed");
        let _ = insert_fixture(&collection, config, config.documents);
        let started = Instant::now();
        collection
            .rebuild_index("embedding")
            .expect("index rebuild must succeed");
        let elapsed = started.elapsed();
        assert!(!collection
            .stats()
            .expect("stats must succeed")
            .indexes
            .is_empty());
        collection.close().expect("collection close must succeed");
        Sample {
            elapsed,
            work: as_u64(config.documents),
        }
    });

    measure("optimize", config, |_| {
        let temporary = tempdir().expect("temporary directory must be available");
        let collection = Collection::create(
            &create_path(temporary.path(), "collection"),
            &schema_with_dimension("lifecycle-optimize", config.dimensions, true, true),
            Some(&manual_options()),
        )
        .expect("collection creation must succeed");
        let _ = insert_fixture(&collection, config, config.documents);
        let started = Instant::now();
        collection.optimize().expect("optimize must succeed");
        let elapsed = started.elapsed();
        assert!(collection
            .health()
            .expect("health must succeed")
            .is_healthy());
        collection.close().expect("collection close must succeed");
        Sample {
            elapsed,
            work: as_u64(config.documents),
        }
    });

    measure("schema_evolution", config, |_| {
        let temporary = tempdir().expect("temporary directory must be available");
        let collection = Collection::create(
            &create_path(temporary.path(), "collection"),
            &schema_with_dimension("lifecycle-schema", config.dimensions, false, false),
            Some(&manual_options()),
        )
        .expect("collection creation must succeed");
        let _ = insert_fixture(&collection, config, small_fixture_count(config));
        let added = FieldSchema::new("priority", DataType::Int32, true, 0)
            .expect("added field must be valid");
        let started = Instant::now();
        collection
            .add_column(&added, None)
            .expect("add column must succeed");
        collection
            .rename_column("priority", "rank")
            .expect("rename column must succeed");
        let mut altered =
            FieldSchema::new("rank", DataType::Int32, true, 0).expect("altered field");
        altered
            .set_index_params(&IndexParams::invert(true, false).expect("rank index"))
            .expect("rank index must be valid");
        collection
            .alter_column(&altered, AlterColumnOption::default())
            .expect("alter column must succeed");
        collection
            .drop_column("rank")
            .expect("drop column must succeed");
        let elapsed = started.elapsed();
        assert!(!collection
            .schema()
            .expect("schema must succeed")
            .has_field("rank"));
        collection.close().expect("collection close must succeed");
        Sample { elapsed, work: 4 }
    });

    measure("flush", config, |_| {
        let temporary = tempdir().expect("temporary directory must be available");
        let collection = Collection::create(
            &create_path(temporary.path(), "collection"),
            &schema_with_dimension("lifecycle-flush", config.dimensions, false, false),
            Some(&manual_options()),
        )
        .expect("collection creation must succeed");
        let _ = insert_fixture(&collection, config, config.documents);
        let started = Instant::now();
        collection.flush().expect("flush must succeed");
        let elapsed = started.elapsed();
        collection.close().expect("collection close must succeed");
        Sample { elapsed, work: 1 }
    });

    measure("reopen", config, |_| {
        let temporary = tempdir().expect("temporary directory must be available");
        let path = create_path(temporary.path(), "collection");
        let collection = Collection::create(
            &path,
            &schema_with_dimension("lifecycle-reopen", config.dimensions, true, true),
            Some(&manual_options()),
        )
        .expect("collection creation must succeed");
        let _ = insert_fixture(&collection, config, config.documents);
        collection.flush().expect("fixture flush must succeed");
        collection.close().expect("fixture close must succeed");
        let mut options = CollectionOptions::new().expect("read-only options must be valid");
        options
            .set_read_only(true)
            .expect("read-only option must be valid");
        let started = Instant::now();
        let reopened = Collection::open(&path, Some(&options)).expect("reopen must succeed");
        assert_eq!(
            reopened.count().expect("count must succeed"),
            config.documents
        );
        assert!(
            reopened
                .stats()
                .expect("stats must succeed")
                .index_cache_hit
        );
        let elapsed = started.elapsed();
        reopened.close().expect("read-only close must succeed");
        Sample {
            elapsed,
            work: as_u64(config.documents),
        }
    });

    measure("resource_rejection", config, |_| {
        let temporary = tempdir().expect("temporary directory must be available");
        let mut options = manual_options();
        let limits = CollectionResourceLimits::new()
            .try_with_max_documents(1)
            .expect("document limit must be valid");
        options
            .set_resource_limits(limits)
            .expect("resource limits must be accepted");
        let collection = Collection::create(
            &create_path(temporary.path(), "collection"),
            &schema_with_dimension("lifecycle-resource", config.dimensions, false, false),
            Some(&options),
        )
        .expect("collection creation must succeed");
        let _ = insert_fixture(&collection, config, 1);
        let rejected = document(2, config.dimensions);
        let started = Instant::now();
        let error = collection
            .insert(&[&rejected])
            .expect_err("document limit must reject the second document");
        let elapsed = started.elapsed();
        assert_eq!(error.code, ErrorCode::ResourceExhausted);
        assert_eq!(collection.count().expect("count must succeed"), 1);
        assert_eq!(
            collection
                .stats()
                .expect("stats must succeed")
                .resource_limit_rejections,
            1
        );
        collection.close().expect("collection close must succeed");
        Sample { elapsed, work: 1 }
    });

    measure("stats_health", config, |_| {
        let temporary = tempdir().expect("temporary directory must be available");
        let collection = Collection::create(
            &create_path(temporary.path(), "collection"),
            &schema_with_dimension("lifecycle-stats", config.dimensions, true, true),
            Some(&manual_options()),
        )
        .expect("collection creation must succeed");
        let _ = insert_fixture(&collection, config, config.documents);
        let started = Instant::now();
        let stats = collection.stats().expect("stats must succeed");
        let health = collection.health().expect("health must succeed");
        let elapsed = started.elapsed();
        assert_eq!(stats.doc_count, as_u64(config.documents));
        assert!(health.is_healthy());
        collection.close().expect("collection close must succeed");
        Sample { elapsed, work: 1 }
    });

    measure("maintenance_start_close", config, |_| {
        let temporary = tempdir().expect("temporary directory must be available");
        let collection = Collection::create(
            &create_path(temporary.path(), "collection"),
            &schema_with_dimension("lifecycle-maintenance", config.dimensions, false, false),
            Some(&manual_options()),
        )
        .expect("collection creation must succeed");
        let _ = insert_fixture(&collection, config, small_fixture_count(config));
        let options = CollectionMaintenanceOptions::new()
            .try_with_interval(Duration::from_secs(60))
            .expect("maintenance interval must be valid");
        let started = Instant::now();
        let runtime = collection
            .start_maintenance(options)
            .expect("maintenance must start");
        assert!(runtime.health().worker_alive);
        runtime.close().expect("maintenance close must succeed");
        let elapsed = started.elapsed();
        collection.close().expect("collection close must succeed");
        Sample { elapsed, work: 1 }
    });

    // Keep the compiler from optimizing the final fixture setup away when a
    // platform's filesystem makes a lifecycle operation exceptionally cheap.
    black_box(config);
}
