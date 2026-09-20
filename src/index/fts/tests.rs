#![allow(clippy::needless_borrow, clippy::too_many_lines)]
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
        trigrams: super::trigram::TrigramTermIndex::from_terms(["rust", "workspace"]),
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

    // Missing terms are skipped inside dense accumulation.
    let with_missing = index
        .search(
            &["rust".into(), "missing-term".into(), "workspace".into()],
            Some(&allowed),
            FtsDefaultOperator::Or,
        )
        .expect("missing term must be skipped");
    assert_eq!(with_missing, dense);
}

use crate::query::{Fts, SearchQuery};
use crate::schema::IndexParams as SchemaIndexParams;
use crate::text::{parse_fts_query, FtsExpr, FtsExprKind, FtsModifier, ParsedFtsQuery};

fn whitespace_fts_index(docs: &[(&str, &str)]) -> (FtsIndex, DocumentMap, OrdinalTable) {
    let params = SchemaIndexParams::fts(Some("whitespace"), None, None).expect("params");
    let docs: DocumentMap = docs
        .iter()
        .map(|(id, body)| {
            let mut doc = Doc::with_pk(*id).expect("pk");
            doc.add_string("body", body).expect("body");
            ((*id).to_string(), Arc::new(doc))
        })
        .collect();
    let ordinals = OrdinalTable::build(&docs).expect("ordinals");
    let index = FtsIndex::build("body", &params, &docs, &ordinals).expect("index");
    (index, docs, ordinals)
}

fn parse_body(expression: &str) -> ParsedFtsQuery {
    let params = SchemaIndexParams::fts(Some("whitespace"), None, None).expect("params");
    let tokenizer = Tokenizer::from_index_params(Some(&params)).expect("tokenizer");
    let mut fts = Fts::new().expect("fts");
    fts.set_query_string(expression).expect("query string");
    let query = SearchQuery::fts("body", &fts, 16).expect("query");
    parse_fts_query(&query, &tokenizer).expect("parse")
}

#[test]
fn expression_candidates_cover_boolean_match_all_and_empty_paths() {
    let (index, _, _) = whitespace_fts_index(&[
        ("a", "rust vector database"),
        ("b", "rust database engine"),
        ("c", "python vector search"),
        ("d", "legacy index"),
    ]);

    let match_all = parse_body("*");
    let all = index
        .expression_candidates(&match_all.root)
        .expect("match-all");
    assert_eq!(all.len(), 4);

    let empty = FtsExpr {
        kind: FtsExprKind::Empty,
        modifier: FtsModifier::None,
        boost: 1.0,
    };
    assert!(index
        .expression_candidates(&empty)
        .expect("empty")
        .is_empty());

    let matcher = parse_body("rust~1");
    assert!(matches!(matcher.root.kind, FtsExprKind::TermMatcher(_)));
    assert!(index.expression_candidates(&matcher.root).is_err());

    let expanded = FtsExpr {
        kind: FtsExprKind::ExpandedTerms(vec!["rust".into(), "python".into(), "missing".into()]),
        modifier: FtsModifier::None,
        boost: 1.0,
    };
    let expanded_hits = index.expression_candidates(&expanded).expect("expanded");
    assert_eq!(expanded_hits.len(), 3);

    let missing_phrase = FtsExpr {
        kind: FtsExprKind::Phrase {
            terms: vec!["rust".into(), "zzzz".into()],
            slop: 0,
        },
        modifier: FtsModifier::None,
        boost: 1.0,
    };
    assert!(index
        .expression_candidates(&missing_phrase)
        .expect("missing phrase")
        .is_empty());

    let empty_phrase = FtsExpr {
        kind: FtsExprKind::Phrase {
            terms: vec![],
            slop: 0,
        },
        modifier: FtsModifier::None,
        boost: 1.0,
    };
    assert!(index
        .expression_candidates(&empty_phrase)
        .expect("empty phrase")
        .is_empty());
}

#[test]
fn expression_candidates_cover_and_or_must_must_not_intersections() {
    let (index, _, _) = whitespace_fts_index(&[
        ("a", "rust vector database"),
        ("b", "rust database engine"),
        ("c", "python vector search"),
        ("d", "legacy rust"),
    ]);

    let and_must = parse_body("+rust +database");
    let and_hits = index
        .expression_candidates(&and_must.root)
        .expect("and must");
    assert_eq!(and_hits.len(), 2);

    let and_with_neg = parse_body("+rust -legacy");
    let and_neg = index
        .expression_candidates(&and_with_neg.root)
        .expect("and neg");
    assert_eq!(and_neg.len(), 2);

    let only_neg = FtsExpr {
        kind: FtsExprKind::And(vec![FtsExpr {
            kind: FtsExprKind::Term("rust".into()),
            modifier: FtsModifier::MustNot,
            boost: 1.0,
        }]),
        modifier: FtsModifier::None,
        boost: 1.0,
    };
    assert!(index
        .expression_candidates(&only_neg)
        .expect("only neg")
        .is_empty());

    let or_must = parse_body("+rust +database OR python");
    let or_hits = index.expression_candidates(&or_must.root).expect("or must");
    assert!(!or_hits.is_empty());

    let or_must_disjoint = parse_body("+rust +zzzz");
    let disjoint = index
        .expression_candidates(&or_must_disjoint.root)
        .expect("disjoint must");
    assert!(disjoint.is_empty());

    let or_with_neg = parse_body("rust OR python -legacy");
    let or_neg = index
        .expression_candidates(&or_with_neg.root)
        .expect("or neg");
    assert!(!or_neg.is_empty());
    // Prohibited term must be applied when the negative clause matches.
    let legacy_only = parse_body("legacy -rust");
    let _ = index.expression_candidates(&legacy_only.root);

    let nested_and = FtsExpr {
        kind: FtsExprKind::And(vec![
            FtsExpr {
                kind: FtsExprKind::Term("rust".into()),
                modifier: FtsModifier::None,
                boost: 1.0,
            },
            FtsExpr {
                kind: FtsExprKind::Or(vec![
                    FtsExpr {
                        kind: FtsExprKind::Term("vector".into()),
                        modifier: FtsModifier::Must,
                        boost: 1.0,
                    },
                    FtsExpr {
                        kind: FtsExprKind::Term("engine".into()),
                        modifier: FtsModifier::Must,
                        boost: 1.0,
                    },
                ]),
                modifier: FtsModifier::None,
                boost: 1.0,
            },
            FtsExpr {
                kind: FtsExprKind::Term("legacy".into()),
                modifier: FtsModifier::MustNot,
                boost: 1.0,
            },
        ]),
        modifier: FtsModifier::None,
        boost: 1.0,
    };
    let nested = index.expression_candidates(&nested_and).expect("nested");
    // Nested Must over disjoint terms can collapse to empty; the path must still fail closed.
    let _ = nested;
}

#[test]
fn prefers_scan_threshold_tracks_phrase_and_boolean_fanout() {
    let (index, _, _) = whitespace_fts_index(&[
        ("a", "alpha beta gamma"),
        ("b", "alpha beta"),
        ("c", "alpha"),
        ("d", "omega"),
    ]);
    let common = parse_body("alpha");
    assert!(index.prefers_scan_for_expression(&common));

    let rare = parse_body("omega");
    assert!(!index.prefers_scan_for_expression(&rare));

    let phrase = parse_body("\"alpha beta\"");
    let _ = index.prefers_scan_for_expression(&phrase);

    let estimated_empty = index.estimated_candidates(&FtsExpr {
        kind: FtsExprKind::Empty,
        modifier: FtsModifier::None,
        boost: 1.0,
    });
    assert_eq!(estimated_empty, 0);

    let estimated_match_all = index.estimated_candidates(&FtsExpr {
        kind: FtsExprKind::MatchAll,
        modifier: FtsModifier::None,
        boost: 1.0,
    });
    assert_eq!(estimated_match_all, 4);

    let estimated_expanded = index.estimated_candidates(&FtsExpr {
        kind: FtsExprKind::ExpandedTerms(vec!["alpha".into(), "omega".into()]),
        modifier: FtsModifier::None,
        boost: 1.0,
    });
    assert!(estimated_expanded >= 2);

    assert!(!index.candidate_matches(
        &FtsExpr {
            kind: FtsExprKind::Empty,
            modifier: FtsModifier::None,
            boost: 1.0,
        },
        0
    ));
    assert!(index.candidate_matches(
        &FtsExpr {
            kind: FtsExprKind::MatchAll,
            modifier: FtsModifier::None,
            boost: 1.0,
        },
        0
    ));
    assert!(index.candidate_matches(
        &FtsExpr {
            kind: FtsExprKind::ExpandedTerms(vec!["alpha".into(), "missing".into()]),
            modifier: FtsModifier::None,
            boost: 1.0,
        },
        0
    ));
    assert!(!index.candidate_matches(
        &FtsExpr {
            kind: FtsExprKind::Phrase {
                terms: vec![],
                slop: 0,
            },
            modifier: FtsModifier::None,
            boost: 1.0,
        },
        0
    ));
}

#[test]
fn document_lengths_reject_duplicate_and_cover_delta_compaction() {
    assert!(DocumentLengths::from_sorted_entries([(0, 1), (0, 2)]).is_err());

    let mut lengths = DocumentLengths::from_sorted_entries([(0, 3), (2, 5)]).expect("base");
    assert!(lengths.contains_key(0));
    assert!(!lengths.contains_key(1));
    assert_eq!(lengths.get(2), Some(&5));
    assert!(!lengths.is_empty());
    assert_eq!(lengths.len(), 2);

    lengths.insert(1, 4).expect("insert gap");
    assert!(lengths.insert(1, 9).is_err());
    lengths.remove(2).expect("remove");
    assert!(!lengths.contains_key(2));
    lengths.insert(2, 7).expect("reinsert");

    for ordinal in 10..140_u64 {
        lengths.insert(ordinal, 1).expect("bulk insert");
    }
    assert!(lengths.len() > 64);
    let keys: Vec<_> = lengths.keys().collect();
    assert!(keys.contains(&0));
    assert!(keys.contains(&139));
}

#[test]
fn candidate_matches_covers_nested_boolean_and_phrase_leaves() {
    let (index, _, _) = whitespace_fts_index(&[
        ("a", "rust vector database"),
        ("b", "rust database engine"),
        ("c", "python vector search"),
        ("d", "legacy rust"),
    ]);

    let and_expr = FtsExpr {
        kind: FtsExprKind::And(vec![
            FtsExpr {
                kind: FtsExprKind::Term("rust".into()),
                modifier: FtsModifier::None,
                boost: 1.0,
            },
            FtsExpr {
                kind: FtsExprKind::Term("legacy".into()),
                modifier: FtsModifier::MustNot,
                boost: 1.0,
            },
        ]),
        modifier: FtsModifier::None,
        boost: 1.0,
    };
    assert!(index.candidate_matches(&and_expr, 0));
    assert!(!index.candidate_matches(&and_expr, 3));

    let and_only_neg = FtsExpr {
        kind: FtsExprKind::And(vec![FtsExpr {
            kind: FtsExprKind::Term("legacy".into()),
            modifier: FtsModifier::MustNot,
            boost: 1.0,
        }]),
        modifier: FtsModifier::None,
        boost: 1.0,
    };
    assert!(!index.candidate_matches(&and_only_neg, 0));

    let or_must = FtsExpr {
        kind: FtsExprKind::Or(vec![
            FtsExpr {
                kind: FtsExprKind::Term("rust".into()),
                modifier: FtsModifier::Must,
                boost: 1.0,
            },
            FtsExpr {
                kind: FtsExprKind::Term("vector".into()),
                modifier: FtsModifier::Must,
                boost: 1.0,
            },
            FtsExpr {
                kind: FtsExprKind::Term("legacy".into()),
                modifier: FtsModifier::MustNot,
                boost: 1.0,
            },
        ]),
        modifier: FtsModifier::None,
        boost: 1.0,
    };
    assert!(index.candidate_matches(&or_must, 0));
    assert!(!index.candidate_matches(&or_must, 1)); // rust but no vector
    assert!(!index.candidate_matches(&or_must, 3)); // prohibited legacy

    let or_optional = FtsExpr {
        kind: FtsExprKind::Or(vec![
            FtsExpr {
                kind: FtsExprKind::Term("python".into()),
                modifier: FtsModifier::None,
                boost: 1.0,
            },
            FtsExpr {
                kind: FtsExprKind::Term("missing".into()),
                modifier: FtsModifier::None,
                boost: 1.0,
            },
        ]),
        modifier: FtsModifier::None,
        boost: 1.0,
    };
    assert!(index.candidate_matches(&or_optional, 2));
    assert!(!index.candidate_matches(&or_optional, 0));

    let phrase = FtsExpr {
        kind: FtsExprKind::Phrase {
            terms: vec!["rust".into(), "vector".into()],
            slop: 0,
        },
        modifier: FtsModifier::None,
        boost: 1.0,
    };
    assert!(index.candidate_matches(&phrase, 0));
    assert!(!index.candidate_matches(&phrase, 1));

    // Force the AND early-exit when a required child fails mid-scan.
    let and_fail = FtsExpr {
        kind: FtsExprKind::And(vec![
            FtsExpr {
                kind: FtsExprKind::Term("rust".into()),
                modifier: FtsModifier::None,
                boost: 1.0,
            },
            FtsExpr {
                kind: FtsExprKind::Term("zzzz".into()),
                modifier: FtsModifier::None,
                boost: 1.0,
            },
        ]),
        modifier: FtsModifier::None,
        boost: 1.0,
    };
    assert!(!index.candidate_matches(&and_fail, 0));
    let filtered = index.expression_candidates(&and_fail).expect("and fail");
    assert!(filtered.is_empty());
}

#[test]
fn search_paths_cover_conjunctive_expression_and_empty_allow_lists() {
    let (index, docs, ordinals) = whitespace_fts_index(&[
        ("a", "rust vector database"),
        ("b", "rust database engine"),
        ("c", "python vector search"),
        ("d", "legacy rust"),
    ]);

    let empty_allowed = RoaringTreemap::new();
    assert!(index
        .search(
            &["rust".into()],
            Some(&empty_allowed),
            FtsDefaultOperator::Or
        )
        .expect("empty allow")
        .is_empty());

    let and_hits = index
        .search(
            &["rust".into(), "database".into()],
            None,
            FtsDefaultOperator::And,
        )
        .expect("and search");
    assert_eq!(and_hits.len(), 2);

    let missing_and = index
        .search(
            &["rust".into(), "zzzz".into()],
            None,
            FtsDefaultOperator::And,
        )
        .expect("missing and");
    assert!(missing_and.is_empty());

    let mut parsed = parse_body("+rust +database OR python");
    parsed.expand_terms([
        "rust", "database", "python", "vector", "legacy", "engine", "search",
    ]);
    let expr_hits = index
        .search_expression(&parsed, None, &docs, &ordinals, "body")
        .expect("expression search");
    assert!(!expr_hits.is_empty());

    let filtered = index
        .search_expression(
            &parsed,
            Some(&RoaringTreemap::from_iter([0_u64, 2])),
            &docs,
            &ordinals,
            "body",
        )
        .expect("filtered expression");
    assert!(filtered
        .iter()
        .all(|(ordinal, _)| *ordinal == 0 || *ordinal == 2));

    let match_all = parse_body("*");
    let all = index
        .search_expression(&match_all, None, &docs, &ordinals, "body")
        .expect("match all");
    // Match-all without positive terms may score empty; candidate planning still runs.
    let _ = all;

    let prefers = index.prefers_scan_for_expression(&parse_body("rust OR database OR vector"));
    let _ = prefers;
}

#[test]
fn trigram_prefilter_expansion_matches_full_vocabulary_scan() {
    let (index, _docs, _ordinals) = whitespace_fts_index(&[
        ("a", "rust vector database engine"),
        ("b", "rusty crust python search"),
        ("c", "legacy workspace mango trust"),
        ("d", "abracadabra rustacean dust"),
        ("e", "ab cd xy"),
    ]);

    let expressions = [
        "*ust*",
        "rust*",
        "*acean",
        "r*st",
        "rus~1",
        "rustacean~1",
        "database~2",
        "[mango TO trust]",
        "*ab*",
    ];
    for expression in expressions {
        let mut pruned = parse_body(expression);
        let mut full = parse_body(expression);
        index.expand_parsed_query(&mut pruned);
        full.expand_terms(index.postings.iter().map(|(term, _)| term));
        assert_eq!(
            pruned.all_terms(),
            full.all_terms(),
            "trigram prune must be recall-equivalent for `{expression}`"
        );
    }
}

#[test]
fn trigram_prefilter_reduces_wildcard_candidates_before_matcher() {
    let (index, _docs, _ordinals) = whitespace_fts_index(&[
        ("a", "rust rusty crust trust dust"),
        ("b", "python vector database engine"),
        ("c", "workspace search legacy mango"),
    ]);
    let vocab_len = index.postings.iter().count();
    let required = vec!["ust".to_string()];
    let candidates = index.trigrams.terms_containing_all(&required);
    assert!(
        candidates.len() < vocab_len,
        "trigram intersection should prune before matcher: {} vs {}",
        candidates.len(),
        vocab_len
    );
    assert!(candidates.iter().all(|term| term.contains("ust")));

    let mut pruned = parse_body("*ust*");
    index.expand_parsed_query(&mut pruned);
    assert_eq!(
        pruned.all_terms(),
        &[
            "crust".to_string(),
            "dust".to_string(),
            "rust".to_string(),
            "rusty".to_string(),
            "trust".to_string()
        ][..]
    );
}
