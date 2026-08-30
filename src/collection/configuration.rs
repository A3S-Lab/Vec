//! Resolution of process defaults and per-collection durability overrides.

use super::CollectionOptions;
use crate::config::{current_config, ConfigBuilder};

pub(super) fn options_config(options: &CollectionOptions) -> ConfigBuilder {
    resolve_options_config(options, current_config())
}

fn resolve_options_config(
    options: &CollectionOptions,
    mut process_config: ConfigBuilder,
) -> ConfigBuilder {
    if let Some(durability) = options.durability {
        process_config.durability = durability;
    }
    process_config
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Durability;

    #[test]
    fn process_durability_is_used_without_a_collection_override() {
        let process = ConfigBuilder::default().durability(Durability::Interval);
        let resolved = resolve_options_config(&CollectionOptions::default(), process);

        assert_eq!(resolved.durability, Durability::Interval);
    }

    #[test]
    fn collection_durability_overrides_the_process_default() {
        let process = ConfigBuilder::default()
            .durability(Durability::Interval)
            .wal_max_ops(3)
            .wal_max_bytes(1024);
        let mut options = CollectionOptions::default();
        options
            .set_durability(Durability::Manual)
            .expect("durability override must be valid");
        let resolved = resolve_options_config(&options, process);

        assert_eq!(resolved.durability, Durability::Manual);
        assert_eq!(resolved.wal_max_ops, Some(3));
        assert_eq!(resolved.wal_max_bytes, Some(1024));
    }
}
