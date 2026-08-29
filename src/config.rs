//! Process-wide configuration and lifecycle.

use crate::error::{Error, ErrorCode, Result};
use serde::{Deserialize, Serialize};
use std::sync::{OnceLock, RwLock};

/// WAL acknowledgement policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Durability {
    /// Sync the WAL file before acknowledging each mutation.
    Always,
    /// Sync after the configured operation/byte threshold.
    Interval,
    /// Only sync when [`crate::Collection::flush`] is called.
    Manual,
}

impl Default for Durability {
    fn default() -> Self {
        Self::Always
    }
}

/// Portable I/O policy.  `Portable` is deliberately the default and works on
/// Intel macOS 12 without optional kernel or SIMD APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IoBackend {
    Portable,
    Pread,
    Mmap,
}

impl Default for IoBackend {
    fn default() -> Self {
        Self::Portable
    }
}

/// Configuration collected before library initialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigBuilder {
    pub memory_limit: u64,
    pub num_threads: u32,
    pub enable_console_log: bool,
    pub fts_brute_force_by_keys_ratio: Option<f32>,
    pub durability: Durability,
    pub io_backend: IoBackend,
    pub wal_max_ops: Option<u64>,
    pub wal_max_bytes: Option<u64>,
}

impl ConfigBuilder {
    pub fn new() -> Self {
        Self {
            memory_limit: 0,
            num_threads: 0,
            enable_console_log: false,
            fts_brute_force_by_keys_ratio: None,
            durability: Durability::Always,
            io_backend: IoBackend::Portable,
            wal_max_ops: None,
            wal_max_bytes: None,
        }
    }

    pub fn memory_limit(mut self, bytes: u64) -> Self {
        self.memory_limit = bytes;
        self
    }

    pub fn num_threads(mut self, count: u32) -> Self {
        self.num_threads = count;
        self
    }

    pub fn enable_console_log(mut self, enable: bool) -> Self {
        self.enable_console_log = enable;
        self
    }

    pub fn fts_brute_force_by_keys_ratio(mut self, ratio: f32) -> Self {
        self.fts_brute_force_by_keys_ratio = Some(ratio);
        self
    }

    pub fn durability(mut self, durability: Durability) -> Self {
        self.durability = durability;
        self
    }

    pub fn io_backend(mut self, backend: IoBackend) -> Self {
        self.io_backend = backend;
        self
    }

    pub fn wal_max_ops(mut self, limit: u64) -> Self {
        self.wal_max_ops = (limit > 0).then_some(limit);
        self
    }

    pub fn wal_max_bytes(mut self, limit: u64) -> Self {
        self.wal_max_bytes = (limit > 0).then_some(limit);
        self
    }

    /// Finalizes the plain-data builder.  No resources are allocated here.
    pub fn build(self) -> Self {
        self
    }
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

static CONFIG: OnceLock<RwLock<ConfigBuilder>> = OnceLock::new();

fn config_cell() -> &'static RwLock<ConfigBuilder> {
    CONFIG.get_or_init(|| RwLock::new(ConfigBuilder::default()))
}

/// Returns a fresh builder with portable defaults.
pub fn default_config() -> ConfigBuilder {
    ConfigBuilder::default()
}

/// Initializes the process-wide runtime.  Initialization is idempotent; a
/// later call replaces configuration only when no collection is doing work.
pub fn initialize(config: Option<&ConfigBuilder>) -> Result<()> {
    let chosen = config.cloned().unwrap_or_default();
    if let Some(ratio) = chosen.fts_brute_force_by_keys_ratio {
        if !ratio.is_finite() || !(0.0..=1.0).contains(&ratio) {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "fts brute-force ratio must be finite and between 0 and 1",
            ));
        }
    }
    *config_cell()
        .write()
        .map_err(|_| Error::internal("configuration lock poisoned"))? = chosen;
    INITIALIZED.store(true, std::sync::atomic::Ordering::Release);
    Ok(())
}

/// Returns whether the runtime has been initialized explicitly.
///
/// The embedded engine also works with defaults without an explicit call, but
/// exposing this bit preserves the zvec lifecycle vocabulary.
pub fn is_initialized() -> bool {
    INITIALIZED.load(std::sync::atomic::Ordering::Acquire)
}

static INITIALIZED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Returns a clone of the active configuration for internal consumers.
pub(crate) fn current_config() -> ConfigBuilder {
    config_cell()
        .read()
        .map(|v| v.clone())
        .unwrap_or_else(|_| ConfigBuilder::default())
}

/// Marks the runtime initialized and releases process-level resources.
pub fn shutdown() -> Result<()> {
    INITIALIZED.store(false, std::sync::atomic::Ordering::Release);
    Ok(())
}

/// Version of the native Rust implementation.
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

pub fn check_version(major: i32, minor: i32, patch: i32) -> bool {
    let mut pieces = env!("CARGO_PKG_VERSION").split('.');
    let current = (
        pieces.next().and_then(|v| v.parse::<i32>().ok()).unwrap_or(0),
        pieces.next().and_then(|v| v.parse::<i32>().ok()).unwrap_or(0),
        pieces.next().and_then(|v| v.parse::<i32>().ok()).unwrap_or(0),
    );
    current >= (major, minor, patch)
}

pub fn version_major() -> i32 {
    env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap_or(0)
}

pub fn version_minor() -> i32 {
    env!("CARGO_PKG_VERSION_MINOR").parse().unwrap_or(0)
}

pub fn version_patch() -> i32 {
    env!("CARGO_PKG_VERSION_PATCH").parse().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_portable() {
        let cfg = ConfigBuilder::default();
        assert_eq!(cfg.durability, Durability::Always);
        assert_eq!(cfg.io_backend, IoBackend::Portable);
    }

    #[test]
    fn invalid_ratio_is_rejected() {
        let result = initialize(Some(&ConfigBuilder::default().fts_brute_force_by_keys_ratio(2.0)));
        assert!(result.is_err());
    }
}
