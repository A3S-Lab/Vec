#!/usr/bin/env python3
"""Run the scale-configurable companion benchmark against zvec 0.7.0.

The Rust side is ``cargo bench --bench scale_compare``.  This script uses the
same corpus generator, query schedule, batch size, HNSW controls, and CSV
columns so that a comparison can be repeated without relying on a manually
copied table.  It is intentionally not part of the Rust release CI: zvec is an
external Python/C++ package and its native wheel is platform-specific.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence, TextIO

import numpy as np
import zvec

TOPK = 10
SPLITMIX_INCREMENT = 0x9E3779B97F4A7C15
SPLITMIX_MULTIPLIER_1 = 0xBF58476D1CE4E5B9
SPLITMIX_MULTIPLIER_2 = 0x94D049BB133111EB
SEED_INDEX_MULTIPLIER = 0xD6E8FEB86659FD93
SEED_DIMENSION_MULTIPLIER = 0xA5A3564D9C4F1B27


@dataclass(frozen=True)
class Config:
    documents: int
    dimensions: int
    queries: int
    rounds: int
    batch_size: int
    ef_search: int
    hnsw_m: int
    ef_construction: int
    mode: str
    post_optimize: bool
    output: Optional[Path]


@dataclass
class Measurement:
    elapsed_seconds: float
    samples_seconds: list[float]
    rankings: list[list[str]]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def env_positive(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def parse_args() -> Config:
    smoke = os.environ.get("A3S_VEC_BENCH_SCALE") == "smoke"
    defaults = {
        "documents": 96 if smoke else env_positive("A3S_VEC_SCALE_DOCUMENTS", 10_000),
        "dimensions": 8 if smoke else env_positive("A3S_VEC_SCALE_DIMENSIONS", 128),
        "queries": 8 if smoke else env_positive("A3S_VEC_SCALE_QUERIES", 32),
        "rounds": 2 if smoke else env_positive("A3S_VEC_SCALE_ROUNDS", 3),
        "batch_size": 32 if smoke else env_positive("A3S_VEC_SCALE_BATCH_SIZE", 512),
        "ef_search": 64 if smoke else env_positive("A3S_VEC_SCALE_EF_SEARCH", 64),
        "hnsw_m": 16 if smoke else env_positive("A3S_VEC_SCALE_HNSW_M", 16),
        "ef_construction": 96
        if smoke
        else env_positive("A3S_VEC_SCALE_EF_CONSTRUCTION", 96),
    }
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("documents", "dimensions", "queries", "rounds", "batch_size"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=positive_int, default=defaults[name])
    parser.add_argument("--ef-search", type=positive_int, default=defaults["ef_search"])
    parser.add_argument("--hnsw-m", type=positive_int, default=defaults["hnsw_m"])
    parser.add_argument(
        "--ef-construction", type=positive_int, default=defaults["ef_construction"]
    )
    parser.add_argument(
        "--mode",
        choices=("flat", "hnsw", "both"),
        default=os.environ.get("A3S_VEC_SCALE_MODE", "both").lower(),
    )
    parser.add_argument(
        "--post-optimize",
        action="store_true",
        help="time an explicit zvec optimize after HNSW creation and include it in total build",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write CSV to this path instead of stdout",
    )
    args = parser.parse_args()
    if args.documents < TOPK:
        parser.error("documents must be at least 10")
    return Config(
        documents=args.documents,
        dimensions=args.dimensions,
        queries=args.queries,
        rounds=args.rounds,
        batch_size=args.batch_size,
        ef_search=args.ef_search,
        hnsw_m=args.hnsw_m,
        ef_construction=args.ef_construction,
        mode=args.mode,
        post_optimize=args.post_optimize,
        output=args.output,
    )


def splitmix64(value: np.ndarray) -> np.ndarray:
    """Vectorized uint64 SplitMix matching the Rust benchmark."""

    state = (value + np.uint64(SPLITMIX_INCREMENT)).astype(np.uint64)
    state = ((state ^ (state >> np.uint64(30))) * np.uint64(SPLITMIX_MULTIPLIER_1)).astype(
        np.uint64
    )
    state = ((state ^ (state >> np.uint64(27))) * np.uint64(SPLITMIX_MULTIPLIER_2)).astype(
        np.uint64
    )
    return state ^ (state >> np.uint64(31))


def vector_batch(start: int, end: int, dimensions: int) -> np.ndarray:
    indices = np.arange(start, end, dtype=np.uint64)[:, None]
    dims = np.arange(dimensions, dtype=np.uint64)[None, :]
    with np.errstate(over="ignore"):
        seeds = (
            indices * np.uint64(SEED_INDEX_MULTIPLIER)
            + dims * np.uint64(SEED_DIMENSION_MULTIPLIER)
        ).astype(np.uint64)
        random_bits = splitmix64(seeds)
    unit = (random_bits >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    return (unit * 2.0 - 1.0).astype(np.float32)


def query_vectors(config: Config) -> np.ndarray:
    indices = [((query * 7_919) + 17) % config.documents for query in range(config.queries)]
    return vector_batch_indices(indices, config.dimensions)


def vector_batch_indices(indices: Sequence[int], dimensions: int) -> np.ndarray:
    values = np.asarray(indices, dtype=np.uint64)[:, None]
    dims = np.arange(dimensions, dtype=np.uint64)[None, :]
    with np.errstate(over="ignore"):
        seeds = (
            values * np.uint64(SEED_INDEX_MULTIPLIER)
            + dims * np.uint64(SEED_DIMENSION_MULTIPLIER)
        ).astype(np.uint64)
        random_bits = splitmix64(seeds)
    unit = (random_bits >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    return (unit * 2.0 - 1.0).astype(np.float32)


def document_batches(config: Config) -> Iterator[list[zvec.Doc]]:
    for start in range(0, config.documents, config.batch_size):
        end = min(start + config.batch_size, config.documents)
        vectors = vector_batch(start, end, config.dimensions)
        yield [
            zvec.Doc(
                f"doc-{index:08d}",
                vectors={"embedding": vectors[offset]},
            )
            for offset, index in enumerate(range(start, end))
        ]


def create_collection(path: str, config: Config) -> zvec.Collection:
    schema = zvec.CollectionSchema(
        "scale_compare",
        vectors=zvec.VectorSchema(
            "embedding",
            zvec.DataType.VECTOR_FP32,
            config.dimensions,
            zvec.FlatIndexParam(metric_type=zvec.MetricType.COSINE),
        ),
    )
    return zvec.create_and_open(path, schema)


def insert_fixture(collection: zvec.Collection, config: Config) -> float:
    started = time.perf_counter()
    for documents in document_batches(config):
        result = collection.insert(documents)
        statuses = result if isinstance(result, list) else [result]
        if not all(status.ok() for status in statuses):
            raise RuntimeError(f"zvec insert failed: {statuses}")
    collection.flush()
    return time.perf_counter() - started


def result_ids(results: Sequence[zvec.Doc]) -> list[str]:
    return [document.id for document in results]


def run_queries(
    collection: zvec.Collection,
    vectors: np.ndarray,
    config: Config,
    mode: str,
) -> Measurement:
    parameter = (
        zvec.HnswQueryParam(ef=config.ef_search, is_using_refiner=False)
        if mode == "hnsw"
        else None
    )
    warmup = zvec.Query(
        field_name="embedding",
        vector=vectors[0],
        param=parameter,
    )
    collection.query(queries=warmup, topk=TOPK, include_vector=False)
    started = time.perf_counter()
    samples: list[float] = []
    rankings: list[list[str]] = []
    for _ in range(config.rounds):
        current: list[list[str]] = []
        for vector in vectors:
            query = zvec.Query(field_name="embedding", vector=vector, param=parameter)
            query_started = time.perf_counter()
            results = collection.query(queries=query, topk=TOPK, include_vector=False)
            samples.append(time.perf_counter() - query_started)
            current.append(result_ids(results))
        rankings = current
    return Measurement(time.perf_counter() - started, samples, rankings)


def percentile(samples: Sequence[float], percentage: int) -> float:
    ordered = sorted(samples)
    rank = max(1, math.ceil(len(ordered) * percentage / 100))
    return ordered[min(rank - 1, len(ordered) - 1)]


def recall_at_topk(expected: Sequence[Sequence[str]], actual: Sequence[Sequence[str]]) -> float:
    hits = sum(
        sum(document_id in expected_ids for document_id in actual_ids)
        for expected_ids, actual_ids in zip(expected, actual)
    )
    return hits / (len(expected) * TOPK)


def row(
    config: Config,
    mode: str,
    insert_seconds: float,
    index_seconds: float,
    optimize_seconds: float,
    measurement: Measurement,
    recall: float,
) -> list[str]:
    total_queries = config.queries * config.rounds
    qps = total_queries / measurement.elapsed_seconds
    return [
        "zvec",
        zvec.__version__,
        mode,
        str(config.documents),
        str(config.dimensions),
        str(config.queries),
        str(config.rounds),
        str(config.batch_size),
        str(config.ef_search),
        str(config.hnsw_m),
        str(config.ef_construction),
        f"{insert_seconds * 1_000:.3f}",
        f"{index_seconds * 1_000:.3f}",
        f"{optimize_seconds * 1_000:.3f}",
        f"{(insert_seconds + index_seconds + optimize_seconds) * 1_000:.3f}",
        f"{recall:.4f}",
        f"{percentile(measurement.samples_seconds, 50) * 1_000_000:.3f}",
        f"{percentile(measurement.samples_seconds, 95) * 1_000_000:.3f}",
        f"{percentile(measurement.samples_seconds, 99) * 1_000_000:.3f}",
        f"{qps:.2f}",
    ]


def run(config: Config, output: TextIO) -> None:
    try:
        zvec.init(query_threads=1, optimize_threads=1)
    except RuntimeError as error:
        # Some embedding applications initialize zvec before importing this
        # helper. The benchmark still remains deterministic if that setup is
        # already complete.
        if "already" not in str(error).lower():
            raise

    root = Path(tempfile.mkdtemp(prefix="a3s-vec-zvec-scale-"))
    collection_path = root / "collection"
    collection: Optional[zvec.Collection] = None
    try:
        collection = create_collection(str(collection_path), config)
        insert_seconds = insert_fixture(collection, config)
        vectors = query_vectors(config)
        exact = run_queries(collection, vectors, config, "flat")
        expected = exact.rankings
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(
            [
                "engine",
                "version",
                "mode",
                "documents",
                "dimensions",
                "queries",
                "rounds",
                "batch_size",
                "ef_search",
                "hnsw_m",
                "ef_construction",
                "insert_ms",
                "index_build_ms",
                "optimize_ms",
                "total_build_ms",
                "recall_at_10",
                "p50_us",
                "p95_us",
                "p99_us",
                "qps",
            ]
        )
        if config.mode in ("flat", "both"):
            writer.writerow(row(config, "flat", insert_seconds, 0.0, 0.0, exact, 1.0))
        if config.mode in ("hnsw", "both"):
            started = time.perf_counter()
            collection.create_index(
                "embedding",
                zvec.HnswIndexParam(
                    metric_type=zvec.MetricType.COSINE,
                    m=config.hnsw_m,
                    ef_construction=config.ef_construction,
                ),
            )
            index_seconds = time.perf_counter() - started
            optimize_seconds = 0.0
            if config.post_optimize:
                started = time.perf_counter()
                collection.optimize()
                optimize_seconds = time.perf_counter() - started
            measurement = run_queries(collection, vectors, config, "hnsw")
            writer.writerow(
                row(
                    config,
                    "hnsw",
                    insert_seconds,
                    index_seconds,
                    optimize_seconds,
                    measurement,
                    recall_at_topk(expected, measurement.rankings),
                )
            )
    finally:
        if collection is not None:
            collection.close()
        shutil.rmtree(root, ignore_errors=True)


def main() -> int:
    config = parse_args()
    if config.mode not in {"flat", "hnsw", "both"}:
        raise ValueError("mode must be flat, hnsw, or both")
    if config.output is None:
        run(config, sys.stdout)
    else:
        config.output.parent.mkdir(parents=True, exist_ok=True)
        with config.output.open("w", encoding="utf-8", newline="") as stream:
            run(config, stream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
