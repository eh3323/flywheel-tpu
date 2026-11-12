#!/usr/bin/env bash
# Copyright 2026 heavyball-research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# One TPU job on the self-hosted v6e runner (.github/workflows/tpu.yml).
#
#   tpu_job.sh pr SRC BASE_SHA OUT     # a pull request's tests
#   tpu_job.sh full SRC DEADLINE OUT   # the whole suite, until DEADLINE
#
# SRC is the checkout under test and OUT a directory for the results; BASE_SHA
# is the commit the pull request targets, DEADLINE a Unix time. This script
# comes from main whatever the pull request changes, and the pull request's
# code runs only in containers that see its tree read-only, OUT, the chip's
# device nodes and the vbar relay's socket, with no network and no other
# path into the VM. Every TPU job takes /tmp/tpu.lock, as the benchmarks and
# evals do, and holds it to the end.
#
# pr: snapshot the IR of the merge base and of the pull request with
# scripts/mlir_snapshot.py, pick the tests with .github/ci/tpu_tests.py, run
# them. full: run every test; at DEADLINE the run is interrupted, and the
# report lists what it did not reach.
set -euo pipefail

mode=$1
src=$(realpath "$2")
arg=$3
out=$(realpath -m "$4")
ci_dir=$(cd "$(dirname "$0")" && pwd)
: "${JAX_VERSION:?JAX_VERSION must be set}"
# Note: setup_tpu_runner.sh writes both into the runner's .env.
vbar_socket=${VBAR_SOCKET:-/run/flywheel-tpu-ci/vbar.sock}
accelerator=${TPU_ACCELERATOR_TYPE:-v6e-1}
tpu_lock=/tmp/tpu.lock
label=flywheel-tpu-ci
container=""
num_containers=0

# Note: a self-hosted runner keeps its workspace, and with it the last job's
# results.
rm -rf "$out"
mkdir -p "$out"
work=$(mktemp -d)

summarize() {
    if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
        cat "$out/selection.md" "$out/report.md" 2>/dev/null \
            >> "$GITHUB_STEP_SUMMARY" || true
    fi
}

cleanup() {
    status=$?
    if [[ -n "$container" ]]; then
        docker rm -f "$container" >/dev/null 2>&1 || true
    fi
    summarize
    rm -rf "$work"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# Remove whatever an earlier job left running: there is one runner, so any
# container of ours is stale.
remove_stale_containers() {
    docker ps -aq --filter "label=$label" | xargs -r docker rm -f >/dev/null
}

# The image, built from this script's copy of tpu.Dockerfile and tagged by
# its hash, so a pull request that edits the file still runs main's image.
build_image() {
    local tag
    tag=$(cat "$ci_dir/tpu.Dockerfile" <(echo "$JAX_VERSION") \
        | sha256sum | cut -c1-16)
    image=$label:$tag
    if ! docker image inspect "$image" >/dev/null 2>&1; then
        docker build -q -t "$image" --build-arg "JAX_VERSION=$JAX_VERSION" \
            -f "$ci_dir/tpu.Dockerfile" "$ci_dir" >/dev/null
    fi
    docker images "$label" --format '{{.Repository}}:{{.Tag}}' \
        | grep -vxF "$image" | xargs -r docker rmi >/dev/null || true
}

# Note: opened read-only and created only when missing, as in the eval
# scripts: with fs.protected_regular, an O_CREAT open of a file in /tmp that
# another user owns fails even when it exists.
take_lock() {
    [[ -e "$tpu_lock" ]] || (umask 022 && touch "$tpu_lock") 2>/dev/null \
        || true
    exec 9< "$tpu_lock"
    if ! flock -n 9; then
        echo "Waiting for $tpu_lock: another TPU job holds it."
        flock 9
    fi
    echo "Holding $tpu_lock."
}

# run_in TREE COMMAND...: COMMAND in a fresh container over TREE.
run_in() {
    local tree=$1 pid status=0
    shift
    num_containers=$((num_containers + 1))
    container=$label-$$-$num_containers
    # Note: python as PID 1 would ignore SIGTERM and SIGINT, hence --init.
    # memlock: libtpu pins the chip's DMA buffers. The TPU_* variables are
    # what libtpu otherwise reads from the metadata server, which the
    # container cannot reach.
    docker run --rm --init --name "$container" --label "$label=1" \
        --network none --read-only --tmpfs /tmp:size=32g \
        --user "$(id -u):$(id -g)" \
        --device /dev/vfio/vfio --device /dev/vfio/0 \
        --ulimit memlock=-1:-1 \
        -v "$vbar_socket:/run/vbar.sock" \
        -e VBAR_CONTROL_SERVICE_URL=unix:/run/vbar.sock \
        -e TPU_SKIP_MDS_QUERY=1 \
        -e "TPU_ACCELERATOR_TYPE=$accelerator" \
        -e TPU_WORKER_ID=0 \
        -e TPU_WORKER_HOSTNAMES=localhost \
        -e HOME=/tmp \
        -v "$tree:/src:ro" -v "$out:/out" \
        "$image" "$@" &
    pid=$!
    # Note: waiting on a background job lets the INT/TERM trap run at once.
    wait "$pid" || status=$?
    docker rm -f "$container" >/dev/null 2>&1 || true
    container=""
    return "$status"
}

run_pr() {
    local base_sha=$1 base status=0
    base=$(git -C "$src" merge-base "$base_sha" HEAD)
    echo "Merge base: $base"
    git -C "$src" diff --name-only "$base" HEAD > "$out/changed_files.txt"
    # Note: a clone, not a worktree, so the tree's .git holds no host path.
    git clone -q --no-checkout "$src" "$work/base"
    git -C "$work/base" checkout -q --detach "$base"
    take_lock
    run_in "$work/base" python scripts/mlir_snapshot.py dump \
        --out /out/snapshot-base
    # Note: compare exits 1 when the IR differs; only a missing report means
    # it failed.
    run_in "$src" python scripts/mlir_snapshot.py compare \
        --baseline /out/snapshot-base --out /out/snapshot-pr \
        --changed /out/changed_configs.json || true
    if [[ ! -f "$out/changed_configs.json" ]]; then
        echo "::error::mlir_snapshot.py compare failed."
        return 1
    fi
    run_in "$src" python .github/ci/tpu_tests.py select \
        --changed-configs /out/changed_configs.json \
        --changed-files /out/changed_files.txt \
        --out /out/selected.txt --summary /out/selection.md
    if [[ ! -s "$out/selected.txt" ]]; then
        return 0
    fi
    mapfile -t tests < "$out/selected.txt"
    run_in "$src" python -m pytest -p no:cacheprovider -rfE --durations=30 \
        --junitxml=/out/junit.xml "${tests[@]}" || status=$?
    run_in "$src" python .github/ci/tpu_tests.py report \
        --junit /out/junit.xml --summary /out/report.md || status=1
    return "$status"
}

# Note: the deadline is a timeout inside the container, so pytest gets SIGINT
# and still writes its report, then SIGKILL 10 minutes later. A timer on the
# host would inherit the lock and could hold it past the job.
run_full() {
    local deadline=$1 status=0
    take_lock
    run_in "$src" python -m pytest -p no:cacheprovider --collect-only -q \
        > "$out/collected.txt"
    if (( $(date +%s) >= deadline )); then
        echo "::error::The window ended before the suite could start."
        return 1
    fi
    run_in "$src" timeout -s INT -k 600 $((deadline - $(date +%s))) \
        python -m pytest -p no:cacheprovider -rfE --durations=50 \
        --junitxml=/out/junit.xml || status=$?
    if (( status == 124 )); then
        echo "::warning::The window ended before the suite did."
    fi
    run_in "$src" python .github/ci/tpu_tests.py report \
        --junit /out/junit.xml --collected /out/collected.txt \
        --summary /out/report.md || status=1
    return "$status"
}

remove_stale_containers
build_image
case "$mode" in
    pr) run_pr "$arg" ;;
    full) run_full "$arg" ;;
    *) echo "unknown mode $mode" >&2; exit 2 ;;
esac
