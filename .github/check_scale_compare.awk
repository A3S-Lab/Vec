# Validate the 20-column CSV emitted by either scale comparison harness.
# Build-stage values are printed to three decimal milliseconds.  The total is
# computed before formatting while the component columns are rounded
# independently, so the largest legitimate drift is just under 0.002 ms.
NR == 1 { sub(/^\357\273\277/, "") }
/^#/ { next }
$1 == "engine" { next }
NF != 20 { bad = 1; next }
{
  sub(/\r$/, "", $20)
  number = "^[0-9]+([.][0-9]+)?$"
  if (($1 != "a3s-vec" && $1 != "zvec") || ($3 != "flat" && $3 != "hnsw") ||
      $4 !~ number || $5 !~ number || $6 !~ number || $7 !~ number ||
      $8 !~ number || $9 !~ number || $10 !~ number || $11 !~ number ||
      $12 !~ number || $13 !~ number || $14 !~ number || $15 !~ number ||
      $16 !~ number || $17 !~ number || $18 !~ number || $19 !~ number ||
      $20 !~ number || $4 < 10 || $5 <= 0 || $6 <= 0 || $7 <= 0 ||
      $8 <= 0 || $9 <= 0 || $10 <= 0 || $11 <= 0 || $12 <= 0 ||
      $15 <= 0 || $16 < 0 || $16 > 1 || $17 <= 0 || $18 <= 0 ||
      $19 <= 0 || $20 <= 0 || $17 > $18 || $18 > $19 ||
      $15 + 0.002 < $12 + $13 + $14) bad = 1
  seen[$3]++
  rows++
}
END {
  if (rows < 1 || bad) exit 1
  for (mode in seen) if (seen[mode] != 1) exit 1
  print "scale-compare gate: valid unique mode rows with monotonic latency metrics"
}
