//! Snapshot-isolated document iterator.

use crate::doc::Doc;
use crate::error::Result;

#[derive(Debug, Clone)]
pub struct DocIterator {
    docs: std::vec::IntoIter<Doc>,
    revision: u64,
}

impl DocIterator {
    pub(crate) fn new(docs: Vec<Doc>, revision: u64) -> Self {
        Self {
            docs: docs.into_iter(),
            revision,
        }
    }
    pub fn revision(&self) -> u64 {
        self.revision
    }
}

impl Iterator for DocIterator {
    type Item = Result<Doc>;

    fn next(&mut self) -> Option<Self::Item> {
        self.docs.next().map(Ok)
    }
}
