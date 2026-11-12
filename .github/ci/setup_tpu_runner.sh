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

# Set up the TPU VM as the self-hosted runner of .github/workflows/tpu.yml.
# Run it once, as root, from a checkout of main:
#
#   sudo .github/ci/setup_tpu_runner.sh TOKEN
#
# TOKEN is a runner registration token, from the repository's Settings >
# Actions > Runners > New self-hosted runner; it expires after an hour. The
# script
# - creates the user tpu-ci, in the docker group, for the runner to run as;
# - runs vbar_relay.py as the service flywheel-tpu-ci-vbar-relay, so the
#   test containers reach the TPU's vbar control agent with no network;
# - installs the GitHub Actions runner, its version pinned and its tarball
#   checked against the release's SHA-256, registers it with the label
#   tpu-v6e, and starts it as a service.
#
# The docker group can do anything root can on the VM, so on the host the
# runner only runs main's workflow and main's tpu_job.sh; a pull request's
# code runs only in tpu_job.sh's containers. To take the runner down:
#
#   cd /home/tpu-ci/runner && sudo ./svc.sh uninstall
#   sudo -u tpu-ci ./config.sh remove --token TOKEN
#   sudo systemctl disable --now flywheel-tpu-ci-vbar-relay
set -euo pipefail

token=${1:?usage: setup_tpu_runner.sh TOKEN}
repo=${REPO:-heavyball-research/flywheel-tpu}
user=tpu-ci
runner_version=2.338.2
runner_sha256=baebf204307bac6193480852732f0aba2ae97b6c5597c244f4d5558390147c40
relay_dir=/usr/local/lib/flywheel-tpu-ci
socket=/run/flywheel-tpu-ci/vbar.sock
ci_dir=$(cd "$(dirname "$0")" && pwd)

if (( EUID != 0 )); then
    echo "Run as root." >&2
    exit 1
fi

if ! id -u "$user" >/dev/null 2>&1; then
    useradd --system --create-home --shell /bin/bash "$user"
fi
usermod -aG docker "$user"
home=$(getent passwd "$user" | cut -d: -f6)

install -D -m 0755 "$ci_dir/vbar_relay.py" "$relay_dir/vbar_relay.py"
cat > /etc/systemd/system/flywheel-tpu-ci-vbar-relay.service <<EOF
[Unit]
Description=flywheel-tpu CI: a unix socket to the TPU vbar control agent
After=docker.service

[Service]
User=$user
RuntimeDirectory=flywheel-tpu-ci
RuntimeDirectoryMode=0700
ExecStart=/usr/bin/python3 $relay_dir/vbar_relay.py $socket
Restart=always

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now flywheel-tpu-ci-vbar-relay

runner_dir=$home/runner
if [[ -f "$runner_dir/.runner" ]]; then
    echo "The runner in $runner_dir is registered already."
    exit 0
fi
tarball=actions-runner-linux-x64-$runner_version.tar.gz
download=$(mktemp -d)
trap 'rm -rf "$download"' EXIT
curl -fsSL -o "$download/$tarball" \
    "https://github.com/actions/runner/releases/download/v$runner_version/$tarball"
echo "$runner_sha256  $download/$tarball" | sha256sum -c -
install -d -o "$user" -g "$user" "$runner_dir"
tar -xzf "$download/$tarball" -C "$runner_dir"
chown -R "$user:$user" "$runner_dir"
# Note: tpu_job.sh passes the accelerator type to libtpu, which cannot read
# it from the metadata server inside a container.
accelerator=$(curl -fsS -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type")
printf 'TPU_ACCELERATOR_TYPE=%s\nVBAR_SOCKET=%s\n' "$accelerator" "$socket" \
    > "$runner_dir/.env"
chown "$user:$user" "$runner_dir/.env"
sudo -u "$user" "$runner_dir/config.sh" --unattended \
    --url "https://github.com/$repo" --token "$token" \
    --name "$(hostname)" --labels tpu-v6e --replace
cd "$runner_dir"
./svc.sh install "$user"
./svc.sh start
