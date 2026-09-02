# Validate the CSV emitted by benches/feature_matrix.rs.
# Keep this independent of the runner shell so the same gate is easy to run
# locally and in GitHub Actions.
NR == 1 { sub(/^\357\273\277/, "") }
/^#/ { next }
$1 == "operation" { next }
NF != 10 { bad = 1; next }
{
  # PowerShell-generated local captures can retain CRLF on the final field.
  sub(/\r$/, "", $10)
  # Reject empty/non-numeric fields as well as non-positive values.  The
  # benchmark prints fixed-point decimals, so this also rejects NaN/Infinity.
  number = "^[0-9]+([.][0-9]+)?$"
  if ($1 !~ /^[A-Za-z0-9_]+$/ ||
      $2 !~ number || $3 !~ number || $4 !~ number || $5 !~ number ||
      $6 !~ number || $7 !~ number || $8 !~ number || $9 !~ number ||
      $10 !~ number || $2 <= 0 || $3 <= 0 || $4 <= 0 || $5 <= 0 ||
      $6 <= 0 || $6 > $7 || $7 > $8 || $9 <= 0 || $10 <= 0) bad = 1
  seen[$1]++
  rows++
}
END {
  if (rows != 53 || bad) exit 1
  for (name in seen) if (seen[name] != 1) exit 1
  print "feature-matrix gate: 53 unique rows with valid metrics"
}
