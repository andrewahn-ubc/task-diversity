"""Fail fast if the installed environment cannot run GPU JAX on Narval."""

from __future__ import annotations

import importlib.metadata as metadata
import platform
import subprocess
import sys
import tomllib
from pathlib import Path


REQUIRED = (
    "jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt", "flax", "chex",
    "optax", "numpy", "scipy", "nvidia-cuda-runtime-cu12", "nvidia-cudnn-cu12",
    "matplotlib", "pyyaml", "wandb",
)


def check_environment() -> None:
    assert sys.version_info[:2] == (3, 13), f"Expected Python 3.13, got {sys.version}"
    assert platform.system() == "Linux" and platform.machine() == "x86_64", (
        f"Expected Linux x86_64, got {platform.platform()}"
    )
    libc, version = platform.libc_ver()
    assert libc == "glibc" and tuple(map(int, version.split("."))) >= (2, 28), (
        f"Expected glibc >= 2.28 for the pinned wheels, got {libc} {version}"
    )

    lock_path = Path(__file__).resolve().parents[1] / "uv.lock"
    with lock_path.open("rb") as lock_file:
        locked = {p["name"]: p["version"] for p in tomllib.load(lock_file)["package"]
                  if "version" in p}
    for name in REQUIRED:
        actual = metadata.version(name)
        assert actual == locked[name], f"{name}: installed {actual}, locked {locked[name]}"
        print(f"{name}=={actual}")

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    )
    print(result.stdout.strip())
    driver = result.stdout.splitlines()[0].split(",")[0].strip()
    assert tuple(map(int, driver.split("."))) >= (575, 51, 3), (
        f"CUDA 12.9 wheels require NVIDIA driver >= 575.51.03; got {driver}"
    )

    import jax
    import jax.numpy as jnp

    cuda_devices = jax.devices("cuda")
    assert cuda_devices, "JAX did not detect a CUDA GPU"
    result = jax.jit(lambda x: x @ x)(jnp.eye(16, dtype=jnp.float32))
    result.block_until_ready()
    assert result.device in cuda_devices, f"JAX calculation ran on {result.device}"
    print(f"CUDA JAX smoke test passed on {result.device}")


if __name__ == "__main__":
    check_environment()
