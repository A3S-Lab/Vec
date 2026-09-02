#!/bin/sh

set -eu

fail() {
    printf '%s\n' "macOS 12 Intel qualification failed: $1" >&2
    exit 1
}

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repository_root=$(CDPATH= cd -- "${script_dir}/.." && pwd)
cd "${repository_root}"

[ "$(uname -s)" = "Darwin" ] || fail "host is not macOS"
[ "$(uname -m)" = "x86_64" ] || fail "host architecture is not Intel x86_64"

os_version=$(sw_vers -productVersion)
os_major=${os_version%%.*}
[ "${os_major}" = "12" ] || fail "host is macOS ${os_version}, not macOS 12"
[ "${MACOSX_DEPLOYMENT_TARGET:-}" = "12.0" ] || fail "MACOSX_DEPLOYMENT_TARGET must be 12.0"

expected_revision=${EXPECTED_REVISION:-}
[ "${#expected_revision}" -eq 40 ] || fail "EXPECTED_REVISION must be a full commit hash"
case "${expected_revision}" in
    *[!0-9a-f]*) fail "EXPECTED_REVISION must contain lowercase hexadecimal characters only" ;;
esac

revision=$(git rev-parse HEAD)
[ "${revision}" = "${expected_revision}" ] || fail "checked-out revision does not match EXPECTED_REVISION"
[ -z "$(git status --porcelain --untracked-files=no)" ] || fail "tracked worktree is dirty"

rustc_version=$(rustc +stable --version)
cargo_version=$(cargo +stable --version)
export CARGO_NET_OFFLINE=true

cargo +stable fmt --all -- --check
cargo +stable clippy --locked --all-targets --all-features -- -D warnings
cargo +stable test --locked
cargo +stable test --locked --all-features
cargo +stable run --locked --example crud_operations
cargo +stable run --locked --example retrieval_workflows
cargo +stable run --locked --example group_by
cargo +stable run --locked --example schema_iteration
cargo +stable run --locked --example maintenance_health
cargo +stable rustdoc --locked --all-features -- -D warnings
cargo +stable package --locked

package_version=$(sed -n 's/^version = "\([^"]*\)"$/\1/p' Cargo.toml | sed -n '1p')
[ -n "${package_version}" ] || fail "package version could not be resolved"

crate_path="target/package/a3s-vec-${package_version}.crate"
[ -f "${crate_path}" ] || fail "versioned crate artifact is missing"

crate_sha256=$(shasum -a 256 "${crate_path}" | awk '{print $1}')
lock_sha256=$(shasum -a 256 Cargo.lock | awk '{print $1}')
qualified_at=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
evidence_dir=target/macos12-intel-evidence
evidence_path="${evidence_dir}/a3s-vec-${package_version}-macos12-intel.json"
mkdir -p "${evidence_dir}"

printf '%s\n' \
    '{' \
    '  "schema": "a3s-vec-macos12-intel-v1",' \
    "  \"version\": \"${package_version}\"," \
    "  \"revision\": \"${revision}\"," \
    "  \"qualified_at\": \"${qualified_at}\"," \
    '  "passed": true,' \
    '  "host": {' \
    "    \"os\": \"macOS ${os_version}\"," \
    '    "architecture": "x86_64",' \
    '    "deployment_target": "12.0"' \
    '  },' \
    '  "features": ["default", "all"],' \
    '  "gates": ["format", "clippy", "unit", "integration", "recovery", "async", "diskann", "examples", "rustdoc", "package"],' \
    "  \"rustc\": \"${rustc_version}\"," \
    "  \"cargo\": \"${cargo_version}\"," \
    "  \"cargo_lock_sha256\": \"${lock_sha256}\"," \
    "  \"crate_sha256\": \"${crate_sha256}\"" \
    '}' > "${evidence_path}"

cp "${crate_path}" "${evidence_dir}/a3s-vec-${package_version}.crate"
printf '%s  %s\n' "${crate_sha256}" "a3s-vec-${package_version}.crate" \
    > "${evidence_dir}/a3s-vec-${package_version}.crate.sha256"

printf '%s\n' "macOS 12 Intel qualification passed for ${revision}"
printf '%s\n' "Evidence: ${evidence_path}"
