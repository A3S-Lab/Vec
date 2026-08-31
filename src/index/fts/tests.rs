use super::document_lengths::DocumentLengths;
use super::posting_list::PostingList;
use super::term_dictionary::TermDictionary;
use super::{
    count_to_f64, use_dense_score_scratch, FtsIndex, PostingEntry, DENSE_SCORE_MIN_VISITS,
};
use crate::doc::{Doc, DocumentMap};
use crate::index::ordinals::OrdinalTable;
use crate::query::FtsDefaultOperator;
use crate::schema::IndexParams;
use crate::text::{bm25_term_score, text_value, Tokenizer};
use roaring::RoaringTreemap;
use std::collections::BTreeMap;
use std::sync::Arc;

#[test]
fn dense_score_scratch_is_bounded_by_visits_and_ordinal_span() {
    assert!(!use_dense_score_scratch(DENSE_SCORE_MIN_VISITS - 1, 1));
    assert!(use_dense_score_scratch(DENSE_SCORE_MIN_VISITS, 32_768));
    assert!(!use_dense_score_scratch(DENSE_SCORE_MIN_VISITS, 32_769));
}

#[test]
fn posting_document_length_reuses_the_frequency_entry_padding() {
    assert_eq!(
        std::mem::size_of::<(u64, PostingEntry)>(),
        std::mem::size_of::<(u64, u32)>()
    );
}

#[test]
fn ngram_postings_match_an_independent_bm25_reference() {
    let params = IndexParams::fts(
        Some("ngram"),
        None,
        Some(r#"{"ngram_min":2,"ngram_max":3,"token_chars":["letter"]}"#),
    )
    .expect("FTS params must be valid");
    let docs: DocumentMap = [
        ("alpha", "aaaa中文"),
        ("beta", "aaab workspace"),
        ("empty", ""),
    ]
    .into_iter()
    .map(|(id, body)| {
        let mut doc = Doc::with_pk(id).expect("document must be valid");
        doc.add_string("body", body).expect("body must be valid");
        (id.to_string(), Arc::new(doc))
    })
    .collect();
    let ordinals = OrdinalTable::build(&docs).expect("ordinals must build");
    let index = FtsIndex::build("body", &params, &docs, &ordinals).expect("index must build");
    let terms = index.tokenizer.tokenize("aaaa");
    let actual = index
        .search(&terms, None, FtsDefaultOperator::Or)
        .expect("index search must succeed");

    let corpus: Vec<_> = docs
        .iter()
        .map(|(id, doc)| {
            (
                ordinals.ordinal(id).expect("document ordinal must exist"),
                index
                    .tokenizer
                    .tokenize(text_value(doc, "body").expect("body must exist")),
            )
        })
        .collect();
    let document_count =
        count_to_f64(u64::try_from(corpus.len()).expect("document count must fit u64"));
    let total_tokens = corpus.iter().map(|(_, tokens)| tokens.len()).sum::<usize>();
    let average_length =
        count_to_f64(u64::try_from(total_tokens).expect("token count must fit u64"))
            / document_count;
    let mut expected = BTreeMap::<u64, f64>::new();
    for term in &terms {
        let document_frequency = count_to_f64(
            u64::try_from(
                corpus
                    .iter()
                    .filter(|(_, tokens)| tokens.contains(term))
                    .count(),
            )
            .expect("document frequency must fit u64"),
        );
        for (ordinal, tokens) in &corpus {
            let frequency = tokens.iter().filter(|token| *token == term).count();
            if frequency == 0 {
                continue;
            }
            *expected.entry(*ordinal).or_default() += bm25_term_score(
                count_to_f64(u64::try_from(frequency).expect("frequency must fit u64")),
                document_frequency,
                document_count,
                count_to_f64(u64::try_from(tokens.len()).expect("document length must fit u64")),
                average_length,
            );
        }
    }
    assert_eq!(
        actual
            .iter()
            .map(|(ordinal, score)| (*ordinal, score.to_bits()))
            .collect::<Vec<_>>(),
        expected
            .into_iter()
            .map(|(ordinal, score)| (ordinal, score.to_bits()))
            .collect::<Vec<_>>()
    );
}

#[test]
fn dense_and_sparse_score_accumulators_are_exactly_equivalent() {
    const DOCUMENTS: u64 = 5_000;

    let document_length = |ordinal: u64| u32::try_from(ordinal % 7 + 2).expect("length fits u32");
    let primary = PostingList::from_sorted_entries((0..DOCUMENTS).map(|ordinal| {
        (
            ordinal,
            PostingEntry {
                frequency: u32::try_from(ordinal % 3 + 1).expect("frequency fits u32"),
                document_length: document_length(ordinal),
            },
        )
    }));
    let secondary =
        PostingList::from_sorted_entries((0..DOCUMENTS).filter(|ordinal| ordinal % 2 == 0).map(
            |ordinal| {
                (
                    ordinal,
                    PostingEntry {
                        frequency: 1,
                        document_length: document_length(ordinal),
                    },
                )
            },
        ));
    let document_lengths = DocumentLengths::from_sorted_entries(
        (0..DOCUMENTS).map(|ordinal| (ordinal, document_length(ordinal))),
    )
    .expect("document lengths must build");
    let total_tokens = document_lengths.values().map(u64::from).sum();
    let index = FtsIndex {
        params: IndexParams::fts(Some("standard"), None, None).expect("FTS params must be valid"),
        tokenizer: Tokenizer::from_index_params(Some(
            &IndexParams::fts(Some("standard"), None, None).expect("FTS params must be valid"),
        ))
        .expect("standard tokenizer must be valid"),
        postings: TermDictionary::from_sorted_entries([
            ("rust".to_string(), Arc::new(primary)),
            ("workspace".to_string(), Arc::new(secondary)),
        ]),
        document_lengths,
        total_tokens,
    };
    let terms = ["rust".to_string(), "workspace".to_string()];
    let document_count = count_to_f64(DOCUMENTS);
    let average_length = count_to_f64(total_tokens) / document_count;
    let allowed: RoaringTreemap = (0..DOCUMENTS).filter(|ordinal| ordinal % 2 == 0).collect();
    let dense = index
        .search(&terms, Some(&allowed), FtsDefaultOperator::Or)
        .expect("adaptive search must succeed");
    let sparse = index
        .search_sparse(&terms, Some(&allowed), document_count, average_length)
        .expect("sparse reference must succeed");

    assert_eq!(dense, sparse);
}
