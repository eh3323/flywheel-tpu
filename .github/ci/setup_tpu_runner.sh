#!/usr/bin/env bash
# One-time install, on the TPU VM, of the GitHub Actions runner tpu.yml runs on. Run it as
# root with a registration token from the repository's Settings -> Actions -> Runners ->
# New self-hosted runner (the token expires after an hour):
#
#   sudo RUNNER_TOKEN=<token> bash .github/ci/setup_tpu_runner.sh
#
# The runner gets an account of its own, with no sudo and a private home, and runs as a
# systemd service. Then, in the repository settings:
#   - Actions -> General: "Require approval for all external contributors". A pull request
#     can rewrite the workflows, so this approval is what keeps a fork's code off the VM.
#   - Secrets and variables -> Actions -> Variables: TPU_RUNNER = true, which turns tpu.yml on.
#   - Issues -> Labels: tpu-test, the label that runs a pull request on the chip.
set -euo pipefail

repo_url="${REPO_URL:-https://github.com/heavyball-research/flywheel-tpu}"
runner_user="${RUNNER_USER:-gha-runner}"
runner_labels="${RUNNER_LABELS:-tpu,v6e}"
runner_dir="/home/${runner_user}/actions-runner"

if [ "$(id -u)" -ne 0 ]; then
  echo "run as root: sudo RUNNER_TOKEN=<token> bash $0" >&2
  exit 1
fi
: "${RUNNER_TOKEN:?set RUNNER_TOKEN to the registration token}"

create_user() {
  if ! id "$runner_user" >/dev/null 2>&1; then
    useradd --create-home --shell /bin/bash "$runner_user"
  fi
  chmod 750 "/home/${runner_user}"
}

# The latest runner release, checked against the SHA-256 its release notes publish.
install_runner() {
  local tmp version sha tarball
  tmp="$(mktemp -d)"
  curl -fsSL -o "${tmp}/release.json" https://api.github.com/repos/actions/runner/releases/latest
  read -r version sha < <(python3 - "${tmp}/release.json" <<'EOF'
import json, re, sys
release = json.load(open(sys.argv[1]))
sha = re.search(r"<!-- BEGIN SHA linux-x64 -->([0-9a-f]{64})", release["body"]).group(1)
print(release["tag_name"].lstrip("v"), sha)
EOF
  )
  tarball="actions-runner-linux-x64-${version}.tar.gz"
  curl -fsSL -o "${tmp}/${tarball}" \
    "https://github.com/actions/runner/releases/download/v${version}/${tarball}"
  echo "${sha}  ${tmp}/${tarball}" | sha256sum -c -
  sudo -u "$runner_user" mkdir -p "$runner_dir"
  sudo -u "$runner_user" tar -xzf "${tmp}/${tarball}" -C "$runner_dir"
  rm -r "$tmp"
  "${runner_dir}/bin/installdependencies.sh"
}

configure() {
  (cd "$runner_dir" && sudo -u "$runner_user" ./config.sh --unattended --replace \
    --url "$repo_url" --token "$RUNNER_TOKEN" --name "$(hostname)" --labels "$runner_labels")
}

install_service() {
  (cd "$runner_dir" && ./svc.sh install "$runner_user" && ./svc.sh start)
}

create_user
install_runner
configure
install_service
echo "runner installed as ${runner_user} with labels ${runner_labels}; now set TPU_RUNNER=true"
