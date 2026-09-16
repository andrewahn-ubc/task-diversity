# banyan_grid/tasks/ruleset_dataset_compact.py
import bz2
import json
import os

import jax.numpy as jnp
import numpy as np


def load_meta(dir_path: str, dataset_file: str = "rulesets.uint32.npy.bz2"):
    """Load meta file corresponding to the dataset file.

    Derives meta filename from dataset filename:
    - dataset_file: 'd1_n20_bs0_rs12345.uint32.npy.bz2'
    - meta_file:    'd1_n20_bs0_rs12345.uint32_meta.json'

    Also handles case where extension is omitted.
    """
    # Extract base name and derive meta filename
    if dataset_file.endswith(".uint32.npy.bz2"):
        base_name = dataset_file[: -len(".uint32.npy.bz2")]
    elif dataset_file.endswith(".npy.bz2"):
        base_name = dataset_file[: -len(".npy.bz2")]
    else:
        # Extension omitted - use as-is
        base_name = dataset_file

    meta_file = f"{base_name}.uint32_meta.json"
    meta_path = os.path.join(dir_path, meta_file)
    if not os.path.exists(meta_path):
        alt_meta_path = os.path.join(dir_path, f"{base_name}_meta.json")
        if os.path.exists(alt_meta_path):
            meta_path = alt_meta_path
    with open(meta_path, "r") as f:
        return json.load(f)


def load_packed_u32_bz2(
    dir_path: str, file_name: str = "rulesets.uint32.npy.bz2"
) -> np.ndarray:
    # Handle case where extension is omitted
    if not file_name.endswith(".uint32.npy.bz2"):
        file_name = f"{file_name}.uint32.npy.bz2"
    path = os.path.join(dir_path, file_name)
    with bz2.BZ2File(path, "rb") as f:
        arr = np.load(f, allow_pickle=False)  # reads the .npy from the bz2 stream
    assert arr.dtype == np.uint32 and arr.ndim == 2
    return arr  # shape (N, R), uint32


def device_put_packed(arr_u32: np.ndarray) -> jnp.ndarray:
    return jnp.asarray(arr_u32, dtype=jnp.uint32)


def load_distractor_table(
    dir_path: str, dataset_file: str = "rulesets.uint32.npy.bz2"
) -> jnp.ndarray | None:
    """Load the distractor lookup table if it exists alongside the dataset."""
    if dataset_file.endswith(".uint32.npy.bz2"):
        base_name = dataset_file[: -len(".uint32.npy.bz2")]
    elif dataset_file.endswith(".npy.bz2"):
        base_name = dataset_file[: -len(".npy.bz2")]
    else:
        base_name = dataset_file

    table_path = os.path.join(dir_path, f"{base_name}.uint32_distractor_table.npy")
    if os.path.exists(table_path):
        arr = np.load(table_path, allow_pickle=False)
        return jnp.asarray(arr, dtype=jnp.int32)
    return None
