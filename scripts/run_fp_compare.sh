#!/usr/bin/env bash
# First-principles same-host a3s-vec ↔ zvec compare runner.
# See docs/scale-compare-protocol.md.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

STAMP="${COMPARE_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${COMPARE_OUT:-$ROOT/target/fp-compare-$STAMP}"
PYTHON="${COMPARE_PYTHON:-$ROOT/.venv-zvec/bin/python}"
PROCESSES="${COMPARE_PROCESSES:-3}"
DOCUMENTS="${A3S_VEC_SCALE_DOCUMENTS:-100000}"
DIMENSIONS="${A3S_VEC_SCALE_DIMENSIONS:-128}"
MODE="${A3S_VEC_SCALE_MODE:-both}"
RAYON="${RAYON_NUM_THREADS:-1}"

if [[ ! -x "$PYTHON" ]]; then
  echo "missing zvec venv python at $PYTHON" >&2
  echo "create with: python3.13 -m venv .venv-zvec && .venv-zvec/bin/pip install zvec==0.7.0 numpy" >&2
  exit 1
fi

mkdir -p "$OUT"
{
  echo "stamp=$STAMP"
  echo "host=$(uname -m) $(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
  echo "os=$(sw_vers -productVersion 2>/dev/null || uname -s)"
  echo "rustc=$(rustc --version)"
  echo "python=$("$PYTHON" --version 2>&1)"
  echo "zvec=$("$PYTHON" -c 'import zvec; print(zvec.__version__)')"
  echo "git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "documents=$DOCUMENTS dimensions=$DIMENSIONS mode=$MODE processes=$PROCESSES rayon=$RAYON"
} | tee "$OUT/meta.txt"

HEADER='engine,version,mode,documents,dimensions,queries,rounds,batch_size,ef_search,hnsw_m,ef_construction,insert_ms,index_build_ms,optimize_ms,total_build_ms,recall_at_10,p50_us,p95_us,p99_us,qps'
echo "$HEADER" >"$OUT/a3s-all.csv"
echo "$HEADER" >"$OUT/zvec-all.csv"

for i in $(seq 1 "$PROCESSES"); do
  echo "a3s process $i/$PROCESSES" >&2
  RAYON_NUM_THREADS="$RAYON" \
    A3S_VEC_SCALE_DOCUMENTS="$DOCUMENTS" \
    A3S_VEC_SCALE_DIMENSIONS="$DIMENSIONS" \
    A3S_VEC_SCALE_MODE="$MODE" \
    cargo bench --bench scale_compare --quiet \
    | tee "$OUT/a3s-p$i.csv" \
    | awk -F, 'NR>1 {print}' >>"$OUT/a3s-all.csv"
done

for i in $(seq 1 "$PROCESSES"); do
  echo "zvec process $i/$PROCESSES" >&2
  "$PYTHON" scripts/scale_compare_zvec.py \
    --documents "$DOCUMENTS" \
    --dimensions "$DIMENSIONS" \
    --mode "$MODE" \
    --output "$OUT/zvec-p$i.csv"
  awk -F, 'NR>1 {print}' "$OUT/zvec-p$i.csv" >>"$OUT/zvec-all.csv"
done

"$PYTHON" - "$OUT" <<'PY'
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

out = Path(sys.argv[1])
numeric = [
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

def load(path: Path):
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    return rows

def median_table(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (row["engine"], row["mode"], row["documents"], row["dimensions"])
        groups[key].append(row)
    medians = []
    for key, items in sorted(groups.items()):
        engine, mode, documents, dimensions = key
        entry = {
            "engine": engine,
            "mode": mode,
            "documents": documents,
            "dimensions": dimensions,
            "n": str(len(items)),
        }
        for col in numeric:
            values = [float(item[col]) for item in items]
            entry[col] = f"{statistics.median(values):.4f}" if col == "recall_at_10" else f"{statistics.median(values):.3f}"
            if col == "recall_at_10":
                entry["recall_min"] = f"{min(values):.4f}"
                entry["recall_max"] = f"{max(values):.4f}"
        medians.append(entry)
    return medians

combined = load(out / "a3s-all.csv") + load(out / "zvec-all.csv")
medians = median_table(combined)
fields = [
    "engine",
    "mode",
    "documents",
    "dimensions",
    "n",
    "insert_ms",
    "index_build_ms",
    "optimize_ms",
    "total_build_ms",
    "recall_at_10",
    "recall_min",
    "recall_max",
    "p50_us",
    "p95_us",
    "p99_us",
    "qps",
]
with (out / "medians.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(medians)
print(f"wrote {out / 'medians.csv'}")
for row in medians:
    print(
        f"{row['engine']:8} {row['mode']:4} n={row['n']} "
        f"build={row['index_build_ms']}ms p50={row['p50_us']}us "
        f"recall={row['recall_at_10']} ({row['recall_min']}-{row['recall_max']})"
    )
PY

echo "artifacts: $OUT" >&2
