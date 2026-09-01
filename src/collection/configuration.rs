//! Per-collection options and resolution of process defaults.

use super::CollectionResourceLimits;
use crate::config::{current_config, ConfigBuilder, Durability, IoBackend};
use crate::error::Result;

/// Supported options for creating or opening a collection.
///
/// Storage layout, buffer, and segment knobs remain outside the public
/// contract:
///
/// ```compile_fail
/// use a3s_vec::CollectionOptions;
///
/// let mut options = CollectionOptions::new().unwrap();
/// options.set_max_buffer_size(1024).unwrap();
/// options.set_segment_num(2).unwrap();
/// ```
#[derive(Debug, Clone, Default)]
pub struct CollectionOptions {
    pub(super) read_only: bool,
    pub(super) durability: Option<Durability>,
    pub(super) io_backend: Option<IoBackend>,
    pub(super) resource_limits: CollectionResourceLimits,
}

impl CollectionOptions {
    pub fn new() -> Result<Self> {
        Ok(Self::default())
    }

    pub fn set_read_only(&mut self, read_only: bool) -> Result<()> {
        self.read_only = read_only;
        Ok(())
    }

    pub fn read_only(&self) -> bool {
        self.read_only
    }

    pub fn set_durability(&mut self, value: Durability) -> Result<()> {
        self.durability = Some(value);
        Ok(())
    }

    pub fn durability(&self) -> Option<Durability> {
        self.durability
    }

    /// Overrides the process-wide derived-sidecar I/O backend for this handle.
    ///
    /// When absent, the backend configured through [`crate::ConfigBuilder`] is
    /// captured when the collection is created or opened.
    pub fn set_io_backend(&mut self, value: IoBackend) -> Result<()> {
        self.io_backend = Some(value);
        Ok(())
    }

    /// Returns this handle's explicit I/O backend override, if any.
    pub fn io_backend(&self) -> Option<IoBackend> {
        self.io_backend
    }

    /// Applies a typed collection-local resource policy.
    pub fn set_resource_limits(&mut self, value: CollectionResourceLimits) -> Result<()> {
        self.resource_limits = value;
        Ok(())
    }

    /// Returns the resource policy that this handle will capture.
    pub fn resource_limits(&self) -> CollectionResourceLimits {
        self.resource_limits
    }
}

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
    if let Some(io_backend) = options.io_backend {
        process_config.io_backend = io_backend;
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

    #[test]
    fn collection_io_backend_overrides_the_process_default() {
        let process = ConfigBuilder::default().io_backend(crate::IoBackend::Mmap);
        assert_eq!(
            resolve_options_config(&CollectionOptions::default(), process.clone()).io_backend,
            crate::IoBackend::Mmap
        );

        let mut options = CollectionOptions::default();
        options
            .set_io_backend(crate::IoBackend::Positioned)
            .expect("backend override must be valid");
        assert_eq!(
            resolve_options_config(&options, process).io_backend,
            crate::IoBackend::Positioned
        );
    }
}
