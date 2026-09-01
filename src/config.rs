//! Process-wide configuration and lifecycle.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::sync::{OnceLock, RwLock};

/// WAL acknowledgement policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Durability {
    /// Sync the WAL file before acknowledging each mutation.
    #[default]
    Always,
    /// Sync after the configured operation/byte threshold.
    Interval,
    /// Only sync when [`crate::Collection::flush`] is called.
    Manual,
}

/// Query-time backend for validated derived index sidecars.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum IoBackend {
    /// Read bounded extents directly from the sidecar with portable positioned
    /// file operations.
    #[default]
    Positioned,
    /// Copy the validated sidecar into an immutable anonymous memory map at
    /// open time, then serve bounded query extents from that snapshot.
    Mmap,
}

/// Supported process-wide durability and sidecar-I/O configuration.
///
/// Resource and logging controls are intentionally absent until
/// they have an implemented execution path:
///
/// ```compile_fail
/// use a3s_vec::{ConfigBuilder, LogLevel, LogType};
///
/// let _ = (LogLevel::Info, LogType::Console);
/// let _ = ConfigBuilder::new()
///     .memory_limit(1024)
///     .num_threads(2)
///     .enable_console_log(true)
///     .fts_brute_force_by_keys_ratio(0.5);
/// ```
///
/// The default [`IoBackend::Positioned`] reader can be replaced by the bounded
/// immutable mmap snapshot backend:
///
/// ```
/// use a3s_vec::{ConfigBuilder, IoBackend};
///
/// let _ = ConfigBuilder::new().io_backend(IoBackend::Mmap).build();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigBuilder {
    pub(crate) durability: Durability,
    pub(crate) wal_max_ops: Option<u64>,
    pub(crate) wal_max_bytes: Option<u64>,
    #[serde(default)]
    pub(crate) io_backend: IoBackend,
}

impl ConfigBuilder {
    pub fn new() -> Self {
        Self {
            durability: Durability::Always,
            wal_max_ops: None,
            wal_max_bytes: None,
            io_backend: IoBackend::Positioned,
        }
    }

    pub fn durability(mut self, durability: Durability) -> Self {
        self.durability = durability;
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

    /// Selects the process default for validated derived-sidecar query reads.
    pub fn io_backend(mut self, backend: IoBackend) -> Self {
        self.io_backend = backend;
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

/// Sets the process defaults captured by collections created or opened after
/// this call. Existing collection handles retain their resolved configuration.
pub fn initialize(config: Option<&ConfigBuilder>) -> Result<()> {
    let chosen = config.cloned().unwrap_or_default();
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

static INITIALIZED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Returns a clone of the active configuration for internal consumers.
pub(crate) fn current_config() -> ConfigBuilder {
    config_cell()
        .read()
        .map_or_else(|_| ConfigBuilder::default(), |v| v.clone())
}

/// Resets process defaults and marks the runtime uninitialized.
pub fn shutdown() -> Result<()> {
    *config_cell()
        .write()
        .map_err(|_| Error::internal("configuration lock poisoned"))? = ConfigBuilder::default();
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
        pieces
            .next()
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(0),
        pieces
            .next()
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(0),
        pieces
            .next()
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(0),
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
        assert_eq!(cfg.io_backend, IoBackend::Positioned);
    }

    #[test]
    fn zero_checkpoint_limits_are_disabled() {
        let cfg = ConfigBuilder::default().wal_max_ops(0).wal_max_bytes(0);
        assert_eq!(cfg.wal_max_ops, None);
        assert_eq!(cfg.wal_max_bytes, None);
    }
}
