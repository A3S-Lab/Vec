# Validate the CSV emitted by benches/concurrent_queries.rs.
NR == 1 { sub(/^\357\273\277/, "") }
/^#/ { next }
$1 == "mode" { next }
NF != 12 { bad = 1; next }
{
  sub(/\r$/, "", $12)
  number = "^[0-9]+([.][0-9]+)?$"
  if ($1 != "hnsw" || $2 !~ number || $3 !~ number || $4 !~ number ||
      $5 !~ number || $6 !~ number || $7 !~ number || $8 !~ number ||
      $9 !~ number || $10 !~ number || $11 !~ number || $12 !~ number ||
      $2 <= 0 || $3 <= 0 || $4 <= 0 || $5 <= 0 || $6 <= 0 || $7 <= 0 ||
      $8 < 0.80 || $8 > 1 || $9 <= 0 || $10 <= 0 || $11 <= 0 ||
      $9 > $10 || $10 > $11 || $12 <= 0) bad = 1
  seen[$2]++
  rows++
}
END {
  if (rows < 1 || bad) exit 1
  for (workers in seen) if (seen[workers] != 1) exit 1
  print "concurrent-query gate: valid unique worker rows with monotonic latency metrics"
}
