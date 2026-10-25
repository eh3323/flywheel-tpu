"""Wait for this host's TPU to be free, then check it is the chip the job expects.

The runner's chip is shared with people benchmarking on it and only one process can open
it, so the TPU job opens it from a child process, which lets go on exit, until one gets
it or --timeout passes, and only then starts pytest.

    python .github/ci/tpu_wait.py --timeout 3600 --expect-kind v6
"""

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass

PROBE = """
import importlib.metadata as metadata
import jax
devices = jax.devices()
print(devices[0].platform, len(devices), devices[0].device_kind, sep="\\t")
print(*(f"{name} {metadata.version(name)}" for name in ("jax", "jaxlib", "libtpu")), sep=", ")
"""
# Note: a held chip fails the kernel's exclusive open of /dev/vfio/0 ("Device or resource
# busy"). Without ALLOW_MULTIPLE_LIBTPU_LOAD=1, libtpu's /tmp/libtpu_lockfile check comes
# first and says "already in use", also when the lockfile is merely another user's.
BUSY = ("Device or resource busy", "already in use")
PROBE_TIMEOUT = 300


@dataclass
class Probe:
    status: str  # "ok", "busy" or "error"
    detail: str  # the child's stdout when ok, else the end of its stderr


def probe() -> Probe:
    """Open the TPU in a child process and report whether it could."""
    env = {**os.environ, "JAX_PLATFORMS": "tpu"}
    try:
        child = subprocess.run(
            [sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, timeout=PROBE_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        return Probe("error", f"opening the TPU took over {PROBE_TIMEOUT}s")
    if child.returncode == 0:
        return Probe("ok", child.stdout)
    tail = "\n".join(child.stderr.strip().splitlines()[-3:])
    return Probe("busy" if any(marker in child.stderr for marker in BUSY) else "error", tail)


def wait_for_chip(timeout: float, interval: float) -> Probe:
    """Probe until the chip opens, fails for another reason than being held, or time is up."""
    deadline = time.monotonic() + timeout
    while True:
        result = probe()
        if result.status != "busy" or time.monotonic() + interval > deadline:
            return result
        print(f"TPU held by another process, retrying in {interval:.0f}s", flush=True)
        time.sleep(interval)


def check_device(result: Probe, expect_kind: str) -> None:
    """Exit unless the probe saw a TPU whose device_kind contains expect_kind."""
    devices, versions = result.detail.splitlines()[:2]
    platform, count, kind = devices.split("\t")
    if platform != "tpu" or expect_kind.lower() not in kind.lower():
        raise SystemExit(f"expected a {expect_kind} TPU, found {count} x {kind!r} on {platform}")
    print(f"{count} x {kind}; {versions}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--timeout", type=float, default=3600, help="seconds to wait for the chip")
    parser.add_argument("--interval", type=float, default=60, help="seconds between probes")
    parser.add_argument("--expect-kind", default="v6", help="a substring of jax's device_kind")
    args = parser.parse_args(argv)
    result = wait_for_chip(args.timeout, args.interval)
    if result.status == "busy":
        print(f"TPU still held after {args.timeout:.0f}s:\n{result.detail}", file=sys.stderr)
        return 1
    if result.status == "error":
        print(f"could not open the TPU:\n{result.detail}", file=sys.stderr)
        return 1
    check_device(result, args.expect_kind)
    return 0


if __name__ == "__main__":
    sys.exit(main())
