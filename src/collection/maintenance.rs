//! Explicitly owned background collection maintenance.

use super::{ensure_same_generation, ensure_writable, persist_index_cache, Collection};
use crate::error::{Error, Result};
use crate::index::IndexRegistry;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::Duration;

const DEFAULT_INTERVAL: Duration = Duration::from_secs(60);
const MIN_INTERVAL: Duration = Duration::from_millis(10);
const MAX_INTERVAL: Duration = Duration::from_secs(365 * 24 * 60 * 60);
const MAX_ERROR_BYTES: usize = 1_024;

/// Periodic schedule for one explicitly owned collection maintenance worker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use = "maintenance options do nothing until passed to Collection::start_maintenance"]
pub struct CollectionMaintenanceOptions {
    interval: Duration,
}

impl CollectionMaintenanceOptions {
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the default 60-second interval. Valid schedules range from 10
    /// milliseconds through 365 days.
    pub fn try_with_interval(mut self, interval: Duration) -> Result<Self> {
        validate_interval(interval)?;
        self.interval = interval;
        Ok(self)
    }

    pub fn interval(&self) -> Duration {
        self.interval
    }
}

impl Default for CollectionMaintenanceOptions {
    fn default() -> Self {
        Self {
            interval: DEFAULT_INTERVAL,
        }
    }
}

/// Lifecycle phase of an explicitly owned maintenance worker.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CollectionMaintenancePhase {
    Running,
    Degraded,
    Closing,
    Closed,
}

/// Point-in-time diagnostics for one maintenance schedule.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollectionMaintenanceHealth {
    pub phase: CollectionMaintenancePhase,
    pub interval_ms: u64,
    pub worker_alive: bool,
    pub run_in_progress: bool,
    pub successful_runs: u64,
    pub failed_runs: u64,
    pub skipped_runs: u64,
    pub last_attempted_revision: Option<u64>,
    pub last_successful_revision: Option<u64>,
    pub last_error: Option<String>,
}

impl CollectionMaintenanceHealth {
    /// Returns true for a live non-degraded worker or a cleanly closed worker.
    pub fn is_healthy(&self) -> bool {
        self.last_error.is_none()
            && matches!(
                self.phase,
                CollectionMaintenancePhase::Running | CollectionMaintenancePhase::Closed
            )
    }
}

#[derive(Debug)]
struct MaintenanceState {
    health: CollectionMaintenanceHealth,
    stop_requested: bool,
    run_requested: bool,
}

#[derive(Debug)]
struct MaintenanceShared {
    state: Mutex<MaintenanceState>,
    wake: Condvar,
}

/// Owner of the standard-thread worker that periodically rebuilds derived
/// indexes and checkpoints authoritative state.
///
/// Dropping or closing the runtime requests shutdown and joins the worker.
/// Maintenance serializes with writers while readers retain the previous
/// immutable generation during index construction.
#[must_use = "retain and close the runtime to own its background worker"]
pub struct CollectionMaintenanceRuntime {
    collection: Collection,
    shared: Arc<MaintenanceShared>,
    worker: Mutex<Option<JoinHandle<()>>>,
    close_gate: Mutex<()>,
    claim_released: AtomicBool,
}

impl std::fmt::Debug for CollectionMaintenanceRuntime {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CollectionMaintenanceRuntime")
            .field("health", &self.health())
            .finish_non_exhaustive()
    }
}

impl CollectionMaintenanceRuntime {
    /// Returns a consistent snapshot of worker progress and the last failure.
    pub fn health(&self) -> CollectionMaintenanceHealth {
        lock_state(&self.shared).health.clone()
    }

    /// Coalesces an immediate maintenance request with any already pending run.
    pub fn trigger(&self) -> Result<()> {
        let mut state = lock_state(&self.shared);
        if !state.health.worker_alive
            || matches!(
                state.health.phase,
                CollectionMaintenancePhase::Closing | CollectionMaintenancePhase::Closed
            )
        {
            return Err(Error::failed_precondition(
                "background maintenance is not running",
            ));
        }
        state.run_requested = true;
        drop(state);
        self.shared.wake.notify_one();
        Ok(())
    }

    /// Requests shutdown, joins the worker, and releases the collection's
    /// single scheduler claim. Repeated calls are safe.
    pub fn close(&self) -> Result<()> {
        self.shutdown()
    }

    fn shutdown(&self) -> Result<()> {
        let _close = lock_mutex(&self.close_gate);
        {
            let mut state = lock_state(&self.shared);
            if state.health.phase == CollectionMaintenancePhase::Closed {
                self.release_claim();
                return Ok(());
            }
            state.stop_requested = true;
            state.run_requested = false;
            state.health.phase = CollectionMaintenancePhase::Closing;
        }
        self.shared.wake.notify_all();

        let worker = lock_mutex(&self.worker).take();
        let join_error = worker
            .and_then(|worker| worker.join().err())
            .map(|_| Error::internal("background maintenance worker panicked while shutting down"));
        self.release_claim();

        let mut state = lock_state(&self.shared);
        state.health.worker_alive = false;
        state.health.run_in_progress = false;
        state.health.phase = CollectionMaintenancePhase::Closed;
        if let Some(error) = &join_error {
            state.health.failed_runs = state.health.failed_runs.saturating_add(1);
            state.health.last_error = Some(bounded_error(error.to_string()));
        }
        drop(state);

        if let Some(error) = join_error {
            Err(error)
        } else {
            Ok(())
        }
    }

    fn release_claim(&self) {
        if !self.claim_released.swap(true, Ordering::AcqRel) {
            self.collection
                .inner
                .maintenance_claimed
                .store(false, Ordering::Release);
        }
    }
}

impl Drop for CollectionMaintenanceRuntime {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

impl Collection {
    /// Starts the collection's single explicitly owned background scheduler.
    ///
    /// The worker is opt-in and uses only standard threads. Each due run
    /// rebuilds the complete derived index registry and checkpoints the same
    /// authoritative revision. Read-only and closed collections reject it.
    pub fn start_maintenance(
        &self,
        options: CollectionMaintenanceOptions,
    ) -> Result<CollectionMaintenanceRuntime> {
        validate_interval(options.interval)?;
        self.ensure_open()?;
        {
            let state = self
                .inner
                .state
                .read()
                .map_err(|_| Error::internal("collection state lock poisoned"))?;
            ensure_writable(&state.options)?;
        }
        if self
            .inner
            .maintenance_claimed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(Error::already_exists(
                "background maintenance already has an owner",
            ));
        }
        if !self.is_open() {
            self.inner
                .maintenance_claimed
                .store(false, Ordering::Release);
            return Err(Error::failed_precondition("collection is closed"));
        }

        let shared = Arc::new(MaintenanceShared {
            state: Mutex::new(MaintenanceState {
                health: CollectionMaintenanceHealth {
                    phase: CollectionMaintenancePhase::Running,
                    interval_ms: duration_ms(options.interval),
                    worker_alive: true,
                    run_in_progress: false,
                    successful_runs: 0,
                    failed_runs: 0,
                    skipped_runs: 0,
                    last_attempted_revision: None,
                    last_successful_revision: None,
                    last_error: None,
                },
                stop_requested: false,
                run_requested: false,
            }),
            wake: Condvar::new(),
        });
        let worker_collection = self.clone();
        let worker_shared = Arc::clone(&shared);
        let worker = match thread::Builder::new()
            .name("a3s-vec-maintenance".to_string())
            .spawn(move || {
                maintenance_worker(&worker_collection, &worker_shared, options.interval);
            }) {
            Ok(worker) => worker,
            Err(error) => {
                self.inner
                    .maintenance_claimed
                    .store(false, Ordering::Release);
                return Err(Error::internal(format!(
                    "spawn background maintenance worker: {error}"
                )));
            }
        };

        Ok(CollectionMaintenanceRuntime {
            collection: self.clone(),
            shared,
            worker: Mutex::new(Some(worker)),
            close_gate: Mutex::new(()),
            claim_released: AtomicBool::new(false),
        })
    }

    fn run_maintenance_pass(&self) -> Result<u64> {
        self.ensure_open()?;
        let _writer = self
            .inner
            .writer
            .lock()
            .map_err(|_| Error::internal("writer lock poisoned"))?;
        let current = self
            .inner
            .state
            .read()
            .map_err(|_| Error::internal("collection state lock poisoned"))?
            .clone();
        ensure_writable(&current.options)?;

        let indexes = Arc::new(IndexRegistry::build(
            &current.schema,
            &current.docs,
            current.revision,
        )?);
        let resource_usage = match current.options.resource_limits.enforce_state(
            &current.schema,
            &current.docs,
            &indexes,
        ) {
            Ok(usage) => usage,
            Err(error) => {
                current.stats.record_resource_limit_rejection();
                return Err(error);
            }
        };
        let mut state = self
            .inner
            .state
            .write()
            .map_err(|_| Error::internal("collection state lock poisoned"))?;
        ensure_same_generation(&state, &current)?;
        state.indexes = Arc::clone(&indexes);
        state.resource_usage = resource_usage;
        drop(state);

        let docs = current
            .docs
            .values()
            .map(|doc| doc.as_ref().clone())
            .collect::<Vec<_>>();
        let mut storage = self
            .inner
            .storage
            .lock()
            .map_err(|_| Error::internal("storage lock poisoned"))?;
        storage.checkpoint(&current.schema, &docs, current.revision, true)?;
        persist_index_cache(&storage, &current.schema, &indexes, current.revision, true);
        Ok(current.revision)
    }
}

fn maintenance_worker(
    collection: &Collection,
    shared: &Arc<MaintenanceShared>,
    interval: Duration,
) {
    loop {
        let state = lock_state(shared);
        let waited = shared.wake.wait_timeout_while(state, interval, |state| {
            !state.stop_requested && !state.run_requested
        });
        let (mut state, timeout) = match waited {
            Ok(value) => value,
            Err(poisoned) => poisoned.into_inner(),
        };
        if state.stop_requested {
            break;
        }
        if !state.run_requested && !timeout.timed_out() {
            continue;
        }
        state.run_requested = false;
        drop(state);

        let observed = collection.stats();
        let mut state = lock_state(shared);
        if state.stop_requested {
            break;
        }
        let revision = match observed {
            Ok(collection_stats) => collection_stats.revision,
            Err(error) => {
                record_failure(&mut state.health, None, &error);
                let collection_closed = !collection.is_open();
                drop(state);
                if collection_closed {
                    break;
                }
                continue;
            }
        };
        state.health.last_attempted_revision = Some(revision);
        if state.health.last_successful_revision == Some(revision)
            && state.health.last_error.is_none()
        {
            state.health.skipped_runs = state.health.skipped_runs.saturating_add(1);
            continue;
        }
        state.health.run_in_progress = true;
        drop(state);

        let result = collection.run_maintenance_pass();
        let mut state = lock_state(shared);
        state.health.run_in_progress = false;
        match result {
            Ok(maintained_revision) => {
                state.health.successful_runs = state.health.successful_runs.saturating_add(1);
                state.health.last_successful_revision = Some(maintained_revision);
                state.health.last_error = None;
                if !state.stop_requested {
                    state.health.phase = CollectionMaintenancePhase::Running;
                }
            }
            Err(error) => record_failure(&mut state.health, Some(revision), &error),
        }
        let collection_closed = !collection.is_open();
        drop(state);
        if collection_closed {
            break;
        }
    }

    let mut state = lock_state(shared);
    state.health.worker_alive = false;
    state.health.run_in_progress = false;
    if !state.stop_requested {
        state.health.phase = CollectionMaintenancePhase::Degraded;
        if state.health.last_error.is_none() {
            state.health.last_error = Some("background maintenance worker stopped".to_string());
        }
    }
}

fn record_failure(health: &mut CollectionMaintenanceHealth, revision: Option<u64>, error: &Error) {
    health.failed_runs = health.failed_runs.saturating_add(1);
    health.last_attempted_revision = revision;
    health.last_error = Some(bounded_error(error.to_string()));
    health.phase = CollectionMaintenancePhase::Degraded;
}

fn validate_interval(interval: Duration) -> Result<()> {
    if interval < MIN_INTERVAL {
        return Err(Error::invalid_argument(
            "maintenance interval must be at least 10 milliseconds",
        ));
    }
    if interval > MAX_INTERVAL {
        return Err(Error::invalid_argument(
            "maintenance interval must not exceed 365 days",
        ));
    }
    Ok(())
}

fn duration_ms(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn bounded_error(mut message: String) -> String {
    if message.len() <= MAX_ERROR_BYTES {
        return message;
    }
    let mut end = MAX_ERROR_BYTES;
    while !message.is_char_boundary(end) {
        end -= 1;
    }
    message.truncate(end);
    message
}

fn lock_state(shared: &MaintenanceShared) -> MutexGuard<'_, MaintenanceState> {
    lock_mutex(&shared.state)
}

fn lock_mutex<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_errors_preserve_utf8() {
        let message = "é".repeat(MAX_ERROR_BYTES);
        let bounded = bounded_error(message);
        assert!(bounded.len() <= MAX_ERROR_BYTES);
        assert!(bounded.chars().all(|character| character == 'é'));
    }

    #[test]
    fn runtime_contract_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CollectionMaintenanceRuntime>();
    }
}
