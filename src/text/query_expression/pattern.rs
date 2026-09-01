//! Deterministic token-pattern matching for structured FTS leaves.

use std::cmp::Ordering;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum WildcardAtom {
    Literal(char),
    AnyOne,
    AnyMany,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WildcardPattern {
    atoms: Vec<WildcardAtom>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FtsTermMatcher {
    Wildcard(WildcardPattern),
    Fuzzy {
        term: String,
        distance: u8,
    },
    Range {
        lower: Option<(String, bool)>,
        upper: Option<(String, bool)>,
    },
}

impl WildcardPattern {
    pub(super) fn new(atoms: impl IntoIterator<Item = WildcardAtom>) -> Self {
        let mut compact = Vec::new();
        for atom in atoms {
            if atom == WildcardAtom::AnyMany && compact.last() == Some(&WildcardAtom::AnyMany) {
                continue;
            }
            compact.push(atom);
        }
        Self { atoms: compact }
    }

    fn matches(&self, candidate: &str) -> bool {
        let candidate: Vec<char> = candidate.chars().collect();
        let mut previous = vec![false; candidate.len().saturating_add(1)];
        previous[0] = true;
        for atom in &self.atoms {
            let mut current = vec![false; candidate.len().saturating_add(1)];
            match atom {
                WildcardAtom::Literal(expected) => {
                    for (index, character) in candidate.iter().enumerate() {
                        current[index + 1] = previous[index] && character == expected;
                    }
                }
                WildcardAtom::AnyOne => {
                    current[1..].copy_from_slice(&previous[..candidate.len()]);
                }
                WildcardAtom::AnyMany => {
                    current[0] = previous[0];
                    for index in 0..candidate.len() {
                        current[index + 1] = previous[index + 1] || current[index];
                    }
                }
            }
            previous = current;
        }
        previous.last().copied().unwrap_or(false)
    }
}

impl FtsTermMatcher {
    pub(crate) fn matches(&self, candidate: &str) -> bool {
        match self {
            Self::Wildcard(pattern) => pattern.matches(candidate),
            Self::Fuzzy { term, distance } => fuzzy_matches(term, candidate, *distance),
            Self::Range { lower, upper } => {
                lower
                    .as_ref()
                    .map_or(true, |(bound, inclusive)| match candidate.cmp(bound) {
                        Ordering::Greater => true,
                        Ordering::Equal => *inclusive,
                        Ordering::Less => false,
                    })
                    && upper.as_ref().map_or(true, |(bound, inclusive)| {
                        match candidate.cmp(bound) {
                            Ordering::Less => true,
                            Ordering::Equal => *inclusive,
                            Ordering::Greater => false,
                        }
                    })
            }
        }
    }
}

fn fuzzy_matches(expected: &str, candidate: &str, maximum: u8) -> bool {
    let expected: Vec<char> = expected.chars().collect();
    let candidate: Vec<char> = candidate.chars().collect();
    let maximum = usize::from(maximum);
    if expected.len().abs_diff(candidate.len()) > maximum {
        return false;
    }
    let mut before_previous = vec![0_usize; candidate.len().saturating_add(1)];
    let mut previous: Vec<usize> = (0..=candidate.len()).collect();
    for (expected_index, expected_character) in expected.iter().enumerate() {
        let mut current = vec![0_usize; candidate.len().saturating_add(1)];
        current[0] = expected_index + 1;
        let mut row_minimum = current[0];
        for (candidate_index, candidate_character) in candidate.iter().enumerate() {
            let substitution =
                previous[candidate_index] + usize::from(expected_character != candidate_character);
            current[candidate_index + 1] = substitution
                .min(previous[candidate_index + 1].saturating_add(1))
                .min(current[candidate_index].saturating_add(1));
            if expected_index > 0
                && candidate_index > 0
                && expected_character == &candidate[candidate_index - 1]
                && expected[expected_index - 1] == *candidate_character
            {
                current[candidate_index + 1] =
                    current[candidate_index + 1].min(before_previous[candidate_index - 1] + 1);
            }
            row_minimum = row_minimum.min(current[candidate_index + 1]);
        }
        if row_minimum > maximum {
            return false;
        }
        before_previous = previous;
        previous = current;
    }
    previous.last().copied().unwrap_or(usize::MAX) <= maximum
}

#[cfg(test)]
mod tests {
    use super::{FtsTermMatcher, WildcardAtom, WildcardPattern};

    #[test]
    fn wildcard_matching_handles_unicode_and_collapsed_stars() {
        let pattern = WildcardPattern::new([
            WildcardAtom::Literal('\u{4e2d}'),
            WildcardAtom::AnyMany,
            WildcardAtom::AnyMany,
            WildcardAtom::Literal('\u{6587}'),
            WildcardAtom::AnyOne,
        ]);
        assert!(
            FtsTermMatcher::Wildcard(pattern.clone()).matches("\u{4e2d}\u{56fd}\u{6587}\u{4ef6}")
        );
        assert!(!FtsTermMatcher::Wildcard(pattern).matches("\u{4e2d}\u{56fd}\u{6587}"));
    }

    #[test]
    fn fuzzy_matching_uses_bounded_transposition_aware_distance() {
        let matcher = FtsTermMatcher::Fuzzy {
            term: "rust".into(),
            distance: 1,
        };
        assert!(matcher.matches("rust"));
        assert!(matcher.matches("trust"));
        assert!(matcher.matches("rsut"));
        assert!(!matcher.matches("rustacean"));
    }

    #[test]
    fn range_matching_honors_independent_bound_inclusivity() {
        let matcher = FtsTermMatcher::Range {
            lower: Some(("alpha".into(), true)),
            upper: Some(("gamma".into(), false)),
        };
        assert!(matcher.matches("alpha"));
        assert!(matcher.matches("beta"));
        assert!(!matcher.matches("gamma"));
    }
}
