# Validate the lifecycle/resource/maintenance CSV emitted by
# benches/lifecycle_matrix.rs.
NR == 1 { sub(/^\357\273\277/, "") }
/^#/ { next }
$1 == "operation" { next }
NF != 10 { bad = 1; next }
{
  sub(/\r$/, "", $10)
  number = "^[0-9]+([.][0-9]+)?$"
  if ($1 !~ /^[A-Za-z0-9_]+$/ ||
      $2 !~ number || $3 !~ number || $4 !~ number || $5 !~ number ||
      $6 !~ number || $7 !~ number || $8 !~ number || $9 !~ number ||
      $10 !~ number || $2 <= 0 || $3 <= 0 || $4 <= 0 || $5 <= 0 ||
      $6 <= 0 || $6 > $7 || $7 > $8 || $9 <= 0 || $10 <= 0 ||
      $5 < $4) bad = 1
  seen[$1]++
  rows++
}
END {
  if (rows != 16 || bad) exit 1
  for (name in seen) if (seen[name] != 1) exit 1
  print "lifecycle-matrix gate: 16 unique rows with valid metrics"
}
