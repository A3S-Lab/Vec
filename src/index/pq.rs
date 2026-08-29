//! Product-quantization primitives.

use crate::error::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct ProductQuantizer {
    pub dimension: usize,
    pub subspaces: usize,
    pub centroids: Vec<Vec<Vec<f32>>>,
}

impl ProductQuantizer {
    pub fn train(vectors: &[Vec<f32>], subspaces: usize, _iterations: usize) -> Result<Self> {
        if vectors.is_empty() || subspaces == 0 {
            return Err(Error::invalid_argument(
                "PQ training requires vectors and positive subspaces",
            ));
        }
        let dimension = vectors[0].len();
        if dimension == 0
            || dimension % subspaces != 0
            || vectors.iter().any(|v| v.len() != dimension)
        {
            return Err(Error::invalid_argument(
                "PQ dimension must be divisible by subspaces",
            ));
        }
        let width = dimension / subspaces;
        let mut centroids = Vec::with_capacity(subspaces);
        for sub in 0..subspaces {
            let centroid = vectors[0][sub * width..(sub + 1) * width].to_vec();
            centroids.push(vec![centroid]);
        }
        Ok(Self {
            dimension,
            subspaces,
            centroids,
        })
    }

    pub fn encode(&self, vector: &[f32]) -> Result<Vec<u8>> {
        if vector.len() != self.dimension {
            return Err(Error::invalid_argument("PQ vector dimension mismatch"));
        }
        let width = self.dimension / self.subspaces;
        Ok((0..self.subspaces)
            .map(|sub| {
                let target = &vector[sub * width..(sub + 1) * width];
                self.centroids[sub]
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        squared(target, a)
                            .partial_cmp(&squared(target, b))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map_or(0, |(i, _)| i as u8)
            })
            .collect())
    }
}

fn squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = *x - *y;
            d * d
        })
        .sum()
}
