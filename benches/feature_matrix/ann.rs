#[derive(Clone, Copy)]
enum AnnMode {
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

impl AnnMode {
    fn name(self) -> &'static str {
        match self {
            Self::Hnsw => "ann_hnsw",
            Self::IvfSoar => "ann_ivf_soar",
            Self::HnswRabitq => "ann_hnsw_rabitq",
            Self::IvfRabitq => "ann_ivf_rabitq",
            Self::Vamana => "ann_vamana",
            Self::VamanaIp => "ann_vamana_ip",
            Self::VamanaCosine => "ann_vamana_cosine",
            Self::VamanaMipsL2 => "ann_vamana_mips_l2",
            Self::Diskann => "ann_diskann_pq",
            Self::DiskannIp => "ann_diskann_ip_pq",
            Self::DiskannCosine => "ann_diskann_cosine_pq",
            Self::DiskannMipsL2 => "ann_diskann_mips_l2_pq",
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

    fn params(self, documents: usize) -> IndexParams {
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
                    i32::try_from(documents).expect("document count fits i32"),
                    1.2,
                )
            }
            Self::Diskann | Self::DiskannIp | Self::DiskannCosine | Self::DiskannMipsL2 => {
                IndexParams::diskann(
                    self.metric(),
                    12,
                    i32::try_from(documents).expect("document count fits i32"),
                    2,
                )
            }
        }
        .expect("ANN descriptor must be valid")
    }

    fn query(self, config: Config) -> SearchQuery {
        let mut query = SearchQuery::new("embedding", &vector_for(17, config.dimensions), 10)
            .expect("ANN query must be valid");
        match self {
            Self::Hnsw | Self::HnswRabitq => query
                .set_hnsw_params(HnswQueryParams::new(64, 0.0, false, true))
                .expect("HNSW controls must be valid"),
            Self::IvfSoar => query
                .set_ivf_params(IvfQueryParams::new(8, true, 1.0))
                .expect("IVF controls must be valid"),
            Self::IvfRabitq => {
                let mut controls = IvfRabitqQueryParams::new(8, 0.0, false, true);
                controls
                    .set_scale_factor(4.0)
                    .expect("IVF RaBitQ scale must be valid");
                query
                    .set_ivf_rabitq_params(controls)
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
                query.params.insert(
                    "metric".into(),
                    serde_json::json!(match self.metric() {
                        MetricType::L2 => "l2",
                        MetricType::Ip => "ip",
                        MetricType::Cosine => "cosine",
                        MetricType::MipsL2 => "mips_l2",
                        MetricType::Undefined => "undefined",
                    }),
                );
                query
                    .set_diskann_params(DiskannQueryParams::new(64))
                    .expect("DiskANN controls must be valid");
            }
        }
        query
    }
}

fn measure_ann_modes(config: Config) {
    for mode in [
        AnnMode::Hnsw,
        AnnMode::IvfSoar,
        AnnMode::HnswRabitq,
        AnnMode::IvfRabitq,
        AnnMode::Vamana,
        AnnMode::VamanaIp,
        AnnMode::VamanaCosine,
        AnnMode::VamanaMipsL2,
        AnnMode::Diskann,
        AnnMode::DiskannIp,
        AnnMode::DiskannCosine,
        AnnMode::DiskannMipsL2,
    ] {
        let directory = tempdir().expect("ANN temporary directory must be available");
        let path = directory.path().join(mode.name());
        let path_string = path.to_str().expect("ANN path must be UTF-8");
        let collection = Collection::create(path_string, &ann_schema(config.dimensions), None)
            .expect("ANN collection must be created");
        insert_ann_fixture(&collection, config);
        collection
            .create_index("embedding", &mode.params(config.documents))
            .expect("ANN index must build");
        let query = mode.query(config);
        let samples = config.rounds * config.queries;
        measure(mode.name(), config, samples, |_| {
            let result = collection
                .query(black_box(&query))
                .expect("ANN query must succeed");
            assert!(!result.is_empty());
            u64::try_from(result.len()).expect("result count fits u64")
        });
        collection.close().expect("ANN collection must close");
    }
}

