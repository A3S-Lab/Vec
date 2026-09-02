# Validate the CSV emitted by benches/mixed_workload.rs.
NR == 1 { sub(/^\357\273\277/, "") }
/^#/ { next }
$1 == "mode" { next }
NF != 19 { bad = 1; next }
{
  sub(/\r$/, "", $19)
  number = "^[0-9]+([.][0-9]+)?$"
  if ($1 != "mixed" || $2 !~ number || $3 !~ number || $4 !~ number ||
      $5 !~ number || $6 !~ number || $7 !~ number || $8 !~ number ||
      $9 !~ number || $10 !~ number || $11 !~ number || $12 !~ number ||
      $13 !~ number || $14 !~ number || $15 !~ number || $16 !~ number ||
      $17 !~ number || $18 !~ number || $19 !~ number || $2 <= 0 ||
      $3 <= 0 || $4 <= 0 || $5 <= 0 || $6 <= 0 || $7 <= 0 || $8 <= 0 ||
      $9 < 0 || $9 > 1 || $10 <= 0 || $11 <= 0 || $12 <= 0 ||
      $13 <= 0 || $14 <= 0 || $15 <= 0 || $16 <= 0 || $17 <= 0 ||
      $18 <= 0 || $19 <= 0 || $10 > $11 || $11 > $12 ||
      $13 > $14 || $14 > $15) bad = 1
  seen[$2]++
  rows++
}
END {
  if (rows < 1 || bad) exit 1
  for (readers in seen) if (seen[readers] != 1) exit 1
  print "mixed-workload gate: valid unique reader rows with monotonic read/write latency metrics"
}
