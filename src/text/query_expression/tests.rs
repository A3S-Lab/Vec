use super::{contains_ordered_phrase, parse_fts_query, FtsExprKind, FtsModifier};
use crate::query::{Fts, FtsQueryParams, SearchQuery};
use crate::schema::IndexParams;
use crate::text::Tokenizer;

fn parse(expression: &str, operator: Option<&str>) -> super::ParsedFtsQuery {
    let params =
        IndexParams::fts(Some("whitespace"), None, None).expect("FTS params must be valid");
    let tokenizer = Tokenizer::from_index_params(Some(&params)).expect("tokenizer must be valid");
    let mut fts = Fts::new().expect("FTS payload must be valid");
    fts.set_query_string(expression)
        .expect("query string must be valid");
    let mut query = SearchQuery::fts("body", &fts, 10).expect("query must be valid");
    if let Some(operator) = operator {
        query
            .set_fts_params(FtsQueryParams::new(Some(operator)).expect("operator must be valid"))
            .expect("operator must be accepted");
    }
    parse_fts_query(&query, &tokenizer).expect("expression must parse")
}

#[test]
fn boolean_precedence_and_modifiers_are_structural() {
    let parsed = parse("+Rust database OR python AND NOT legacy", None);
    let FtsExprKind::Or(children) = &parsed.root.kind else {
        panic!("root must be OR");
    };
    assert_eq!(children.len(), 2);
    let FtsExprKind::Or(left) = &children[0].kind else {
        panic!("left branch must use the implicit OR shape");
    };
    assert_eq!(left[0].modifier, FtsModifier::Must);
    let FtsExprKind::And(right) = &children[1].kind else {
        panic!("right branch must be AND");
    };
    assert_eq!(right[1].modifier, FtsModifier::MustNot);
}

#[test]
fn explicit_operator_overrides_the_default_operator() {
    let parsed = parse("rust OR python database", Some("AND"));
    assert!(matches!(parsed.root.kind, FtsExprKind::Or(_)));
    assert!(parsed.simple().is_none());
}

#[test]
fn advanced_leaves_expand_against_one_vocabulary() {
    let mut parsed = parse("rust* OR rust~1 OR [mango TO rust]", None);
    parsed.expand_terms(["trust", "rust", "mango", "rustacean", "python", "rust"]);
    assert_eq!(
        parsed.all_terms(),
        ["mango", "python", "rust", "rustacean", "trust"]
    );
    assert!(parsed.simple().is_none());
}

#[test]
fn exact_queries_do_not_consume_a_dynamic_vocabulary() {
    let mut parsed = parse("rust AND database", None);
    parsed.expand_terms(std::iter::once_with(|| {
        panic!("exact queries must keep the specialized path")
    }));
    assert_eq!(parsed.all_terms(), ["database", "rust"]);
    assert!(parsed.simple().is_some());
}

#[test]
fn boosts_and_phrase_slop_are_attached_to_the_target_atom() {
    let parsed = parse("body:\"vector engine\"~2^3", None);
    assert!((parsed.root.boost - 3.0).abs() < f64::EPSILON);
    assert!(matches!(
        parsed.root.kind,
        FtsExprKind::Phrase { slop: 2, .. }
    ));
}

#[test]
fn ordered_phrase_slop_counts_total_intervening_tokens() {
    let tokens = ["vector", "fast", "search", "engine"].map(str::to_string);
    let phrase = ["vector", "search", "engine"].map(str::to_string);
    assert!(!contains_ordered_phrase(&tokens, &phrase, 0));
    assert!(contains_ordered_phrase(&tokens, &phrase, 1));

    let reversed = ["search", "vector"].map(str::to_string);
    assert!(!contains_ordered_phrase(&tokens, &reversed, 4));
}
