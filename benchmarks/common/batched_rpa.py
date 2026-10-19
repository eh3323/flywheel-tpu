"""SHA-pinned loader for vLLM TorchTPU's batched RPA kernel, the kernel its
`--attention-backend CUSTOM` serves with.

--batched-rpa-source defaults to the third_party/vllm-torchtpu submodule
(git submodule update --init third_party/vllm-torchtpu). Every
src/vllm_torchtpu file the kernel imports is verified against its SHA-256 at
BATCHED_RPA_GIT_SHA, the submodule pin, and the kernel is imported through the
real vllm_torchtpu package, so the import-time JAX patches and libtpu flag
defaults vLLM TorchTPU serves with apply here too. Those files need only JAX;
no vLLM or torch install.

    PYTHONPATH=. uv run --no-sync python benchmarks/common/batched_rpa.py   # host-only self-test
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path

DEFAULT_BATCHED_RPA_SOURCE = Path(__file__).resolve().parents[2] / (
    "third_party") / "vllm-torchtpu"
BATCHED_RPA_GIT_SHA = "fe86920a4c851ed80c82ad78ca9ea7cc88add3a6"
BATCHED_RPA_PACKAGE = "vllm_torchtpu.kernels.experimental.batched_rpa"
# The kernel's whole vllm_torchtpu import closure, relative to
# src/vllm_torchtpu.
BATCHED_RPA_SOURCE_SHA256 = {
    "__init__.py":
        "a1fc9f8de9942c0e27dfa639e4acfa59b48aa0443189eb33ff0609f1f14cba3d",
    "env_override.py":
        "564eff39afb19621938f51ca67590ca6eaac3d9a1973e08cf537dcece08a3061",
    "envs.py":
        "51fe1c89a983ba03bd4b7b049c8a7676b6afcb155a22fc0f2b0f0db8a81444fa",
    "kernels/__init__.py":
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "kernels/experimental/__init__.py":
        "43955acccf4b719dbab4dceb4c9437274dc864bf3d15b22c252cd9d010255c7d",
    "kernels/experimental/batched_rpa/__init__.py":
        "dae715a4f8e5c127d40a795ddf630d294563ac44234e333077f4aa9cbeae3e05",
    "kernels/experimental/batched_rpa/bref_override.py":
        "447be6f37e5b2bf5632a0bd9f8e9724e4b9373c2f6bf452fa9165e8dccb5b380",
    "kernels/experimental/batched_rpa/configs.py":
        "dfa7df3e22eda2ae12d93b56a02facef4d7c0c0bfbeff03af4c9dc6253b408fd",
    "kernels/experimental/batched_rpa/flash_attention.py":
        "b490517cfc503ccc298d5d511469370730fbe50c82288048337810f54dedb804",
    "kernels/experimental/batched_rpa/kernel.py":
        "522246e55347638f05aa060722612294ca54bc63631c78d5aa74a53cc93718e7",
    "kernels/experimental/batched_rpa/pipeline.py":
        "ee702226a72f5db5eab4bdbe90fefef1ec43a4cb027105b8923276880a53ef34",
    "kernels/experimental/batched_rpa/schedule.py":
        "3e77d06c7669ea83869f3fa19cacc98608c891357517b91a736551335c2a429b",
    "kernels/experimental/batched_rpa/schedule_cp.py":
        "fab697e2e4af64f58704f8056431140a527c418bd99e43b40c0f2e843e35b577",
    "kernels/experimental/batched_rpa/stitch_utils.py":
        "221b32d923a22dd12dd5d2a268defceb62936d237cc035bd0f3962e988d3e9ce",
    "kernels/experimental/batched_rpa/utils.py":
        "1c917940957aef3c04f24e6125b78a31e1229c2221dbf6ad6dd2c519947858d5",
    "kernels/experimental/batched_rpa/wrapper.py":
        "67380581df5ac1bdc67e403e36b264c9248fcc7946814653a2c099eafdb782ff",
    "logger.py":
        "2505160410a746207d8a406b64ac26e2b8fb6090d58dc1c726ba25ccb5650136",
    "patch_registry.py":
        "21e631bb1f98477c89e567c1bd0e425da81e42be1844213f9d4257b2004e0fa0",
}
# The page size vLLM TorchTPU allocates under --attention-backend CUSTOM when
# --block-size is not given (PallasBatchedRPAAttentionBackend.
# get_preferred_block_size at BATCHED_RPA_GIT_SHA).
CUSTOM_PAGE_SIZE = 256
# JAX device_kind -> pltpu.ChipVersion value, for get_kv_cache_shape, which
# vLLM TorchTPU resolves from the accelerator type instead of a live device.
CHIP_VERSIONS = {"TPU v6 lite": "v6e", "TPU7x": "7x"}


def validate_batched_rpa_source(source_root: Path) -> Path:
    """The src/vllm_torchtpu directory inside a vLLM TorchTPU checkout."""
    source_root = Path(source_root).expanduser().resolve()
    package_root = source_root / "src" / "vllm_torchtpu"
    missing = [name for name in BATCHED_RPA_SOURCE_SHA256
               if not (package_root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{package_root} is missing {missing}; --batched-rpa-source must "
            f"be a vLLM TorchTPU checkout at {BATCHED_RPA_GIT_SHA} (for the "
            f"default, run git submodule update --init "
            f"third_party/vllm-torchtpu).")
    for name, expected in BATCHED_RPA_SOURCE_SHA256.items():
        digest = hashlib.sha256((package_root / name).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(
                f"{package_root / name} does not match vLLM TorchTPU "
                f"{BATCHED_RPA_GIT_SHA}; got sha256 {digest}.")
    return package_root


def load_batched_rpa(source_root: Path):
    """The pinned batched RPA (wrapper, configs) modules.

    Call it before JAX starts its backend: importing vllm_torchtpu sets the
    LIBTPU_INIT_ARGS defaults vLLM TorchTPU serves with (among them the
    dynamic SMEM negotiation batched RPA's scheduling metadata needs), and
    libtpu reads them only once.
    """
    from jax._src import xla_bridge

    package_root = validate_batched_rpa_source(source_root)
    if "vllm_torchtpu" in sys.modules:
        raise RuntimeError(
            "vllm_torchtpu was imported before the pinned batched RPA source; "
            "refusing to benchmark an ambiguous baseline.")
    if xla_bridge.backends_are_initialized():
        raise RuntimeError(
            "JAX started its backend before batched RPA was loaded, so vLLM "
            "TorchTPU's LIBTPU_INIT_ARGS defaults would not apply.")
    sys.path.insert(0, str(package_root.parent))
    wrapper = importlib.import_module(f"{BATCHED_RPA_PACKAGE}.wrapper")
    configs = importlib.import_module(f"{BATCHED_RPA_PACKAGE}.configs")
    loaded = Path(sys.modules["vllm_torchtpu"].__file__).resolve()
    if loaded.parent != package_root:
        raise RuntimeError(
            f"vllm_torchtpu resolved to {loaded}, not {package_root}.")
    return wrapper, configs


def chip_version(device_kind: str):
    """The pltpu.ChipVersion of the chip JAX calls device_kind."""
    from jax.experimental.pallas import tpu as pltpu

    if device_kind not in CHIP_VERSIONS:
        raise ValueError(
            f"no chip version for {device_kind!r}; add it to CHIP_VERSIONS.")
    return pltpu.ChipVersion(CHIP_VERSIONS[device_kind])


def self_test():
    validate_batched_rpa_source(DEFAULT_BATCHED_RPA_SOURCE)
    print("self-test ok")


if __name__ == "__main__":
    self_test()
