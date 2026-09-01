//! Tokio entry points for running synchronous collection queries off-runtime.

use super::Collection;
use crate::doc::Doc;
use crate::error::{Error, Result};
use crate::multi_query::MultiQuery;
use crate::query::{GroupBySearchQuery, SearchQuery};
use std::collections::HashMap;

impl Collection {
    /// Executes a query on Tokio's blocking pool.
    ///
    /// This keeps positioned `DiskANN` reads and exact refinement off async
    /// runtime worker threads. The query still uses the same synchronous
    /// snapshot, planner, fallback, scoring, and telemetry path as [`Self::query`].
    /// Once scheduled, dropping the returned future does not cancel the
    /// blocking query.
    pub async fn query_async(&self, query: &SearchQuery) -> Result<Vec<Doc>> {
        let collection = self.clone();
        let query = query.clone();
        run_blocking("query", move || collection.query(&query)).await
    }

    /// Executes every multi-query branch on Tokio's blocking pool.
    ///
    /// The full fusion operation runs in one blocking task so all branches use
    /// the same captured collection snapshot as [`Self::multi_query`].
    /// Once scheduled, dropping the returned future does not cancel the task.
    pub async fn multi_query_async(&self, query: &MultiQuery) -> Result<Vec<Doc>> {
        let collection = self.clone();
        let query = query.clone();
        run_blocking("multi-query", move || collection.multi_query(&query)).await
    }

    /// Executes a grouped vector query on Tokio's blocking pool.
    ///
    /// Once scheduled, dropping the returned future does not cancel the task.
    pub async fn group_by_async(
        &self,
        query: &GroupBySearchQuery,
    ) -> Result<HashMap<String, Vec<Doc>>> {
        let collection = self.clone();
        let query = query.clone();
        run_blocking("group-by query", move || collection.group_by(&query)).await
    }
}

async fn run_blocking<T, F>(operation: &'static str, task: F) -> Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T> + Send + 'static,
{
    let runtime = tokio::runtime::Handle::try_current().map_err(|_| {
        Error::failed_precondition(format!(
            "async {operation} requires an active Tokio runtime"
        ))
    })?;
    runtime
        .spawn_blocking(task)
        .await
        .map_err(|error| Error::internal(format!("async {operation} task failed: {error}")))?
}

#[cfg(test)]
mod tests {
    use super::run_blocking;
    use crate::ErrorCode;

    #[test]
    fn missing_tokio_runtime_is_a_typed_error() {
        let future = run_blocking("test operation", || Ok::<_, crate::Error>(()));
        let error = poll_once_without_runtime(future).expect_err("runtime must be required");
        assert_eq!(error.code, ErrorCode::FailedPrecondition);
        assert_eq!(
            error.message,
            "async test operation requires an active Tokio runtime"
        );
    }

    #[test]
    fn query_work_leaves_the_runtime_worker_thread() {
        let caller = std::thread::current().id();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("Tokio runtime must build");
        let worker = runtime
            .block_on(run_blocking("test operation", || {
                Ok::<_, crate::Error>(std::thread::current().id())
            }))
            .expect("blocking task must succeed");
        assert_ne!(worker, caller);
    }

    fn poll_once_without_runtime<F>(future: F) -> F::Output
    where
        F: std::future::Future,
    {
        use std::sync::Arc;
        use std::task::{Context, Poll, Wake, Waker};

        struct NoopWake;

        impl Wake for NoopWake {
            fn wake(self: Arc<Self>) {}
        }

        let waker = Waker::from(Arc::new(NoopWake));
        let mut context = Context::from_waker(&waker);
        let mut future = std::pin::pin!(future);
        match future.as_mut().poll(&mut context) {
            Poll::Ready(output) => output,
            Poll::Pending => panic!("future unexpectedly waited without a Tokio runtime"),
        }
    }
}
