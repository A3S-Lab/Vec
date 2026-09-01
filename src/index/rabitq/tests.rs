use super::{pack_values, packed_bytes, unpack_value, RabitqQuantizer};
use crate::index::ordinal_map::OrdinalMap;
use crate::index::quantization::{score_dense, QuantizedVector};
use crate::types::{MetricType, QuantizeType};

fn vectors() -> OrdinalMap<QuantizedVector> {
    (0_u16..48)
        .map(|ordinal| {
            let vector = (0_u16..7)
                .map(|dimension| {
                    let raw = (ordinal * 31 + dimension * 17 + ordinal * dimension * 3) % 101;
                    (f32::from(raw) - 50.0) / 50.0
                })
                .collect();
            (
                u64::from(ordinal),
                QuantizedVector::encode(vector, QuantizeType::Undefined).unwrap(),
            )
        })
        .collect()
}

#[test]
fn encoding_is_deterministic_compact_and_metric_aware() {
    let vectors = vectors();
    for metric in [MetricType::L2, MetricType::Ip, MetricType::Cosine] {
        let first = RabitqQuantizer::build(&vectors, 7, 7, 8, 24, metric).unwrap();
        let second = RabitqQuantizer::build(&vectors, 7, 7, 8, 24, metric).unwrap();
        assert_eq!(first, second);
        assert!(first.validates(&vectors, 7, 7, 8, 24, metric));
        assert!(first
            .codes
            .values()
            .all(|code| code.packed.len() == packed_bytes(8, 7)));
    }
}

#[test]
fn seven_bit_estimates_are_no_worse_than_one_bit_on_a_fixed_fixture() {
    let vectors = vectors();
    let query = vectors.get(17).unwrap().decode();
    for metric in [MetricType::L2, MetricType::Ip, MetricType::Cosine] {
        let one_bit = RabitqQuantizer::build(&vectors, 7, 1, 8, 24, metric).unwrap();
        let seven_bit = RabitqQuantizer::build(&vectors, 7, 7, 8, 24, metric).unwrap();
        let one_query = one_bit.prepare_query(&query).unwrap();
        let seven_query = seven_bit.prepare_query(&query).unwrap();
        let mut one_error = 0.0;
        let mut seven_error = 0.0;
        for (ordinal, vector) in vectors.iter() {
            let exact = score_dense(&query, &vector.decode(), metric);
            let one_score = one_bit.score(&one_query, ordinal).unwrap();
            let seven_score = seven_bit.score(&seven_query, ordinal).unwrap();
            assert!(one_score.is_finite());
            assert!(seven_score.is_finite());
            one_error += (one_score - exact).abs();
            seven_error += (seven_score - exact).abs();
        }
        assert!(
            seven_error <= one_error,
            "{metric:?}: {seven_error} > {one_error}"
        );
    }
}

#[test]
fn compact_codes_round_trip_every_supported_bit_width() {
    for total_bits in 1..=9 {
        let mask = (1_u16 << total_bits) - 1;
        let values: Vec<u16> = (0_u16..37)
            .map(|value| value.wrapping_mul(29) & mask)
            .collect();
        let packed = pack_values(&values, total_bits);
        assert_eq!(packed.len(), packed_bytes(values.len(), total_bits));
        assert_eq!(
            values,
            (0..values.len())
                .map(|index| unpack_value(&packed, index, total_bits))
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn l2_factors_remain_finite_for_extreme_f32_coordinates() {
    let vectors: OrdinalMap<QuantizedVector> = [
        (0, vec![f32::MAX, f32::MAX]),
        (1, vec![-f32::MAX, -f32::MAX]),
    ]
    .into_iter()
    .map(|(ordinal, vector)| {
        (
            ordinal,
            QuantizedVector::encode(vector, QuantizeType::Undefined).unwrap(),
        )
    })
    .collect();
    let quantizer = RabitqQuantizer::build(&vectors, 2, 7, 1, 0, MetricType::L2).unwrap();
    let query = quantizer.prepare_query(&[f32::MAX, f32::MAX]).unwrap();
    let near = quantizer.score(&query, 0).unwrap();
    let far = quantizer.score(&query, 1).unwrap();
    assert!(near.is_finite());
    assert!(far.is_finite());
    assert!(near > far);
}
