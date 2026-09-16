import numpy as np

from banyan_grid.utils.banyan import build_depth_lookup_from_meta


def _to_numpy_depth_lookup(meta):
    out = build_depth_lookup_from_meta(meta)
    if out is None:
        return None
    counts, depths = out
    return np.array(counts), np.array(depths)


def test_build_depth_lookup_mixed_topology_disabled():
    meta = {
        "tree_topology": "mixed_u1_b2",
        "rules_per_depth": {"1": 1, "2": 1, "3": 3, "4": 7, "5": 15, "6": 31},
    }
    assert build_depth_lookup_from_meta(meta) is None


def test_build_depth_lookup_ambiguous_distribution_disabled():
    meta = {
        "tree_topology": "custom",
        "topology_rule_count_distribution_per_depth": {
            "4": {"7": 10},
            "5": {"7": 12},
        },
    }
    assert build_depth_lookup_from_meta(meta) is None


def test_build_depth_lookup_unambiguous_distribution_used():
    meta = {
        "tree_topology": "custom",
        "topology_rule_count_distribution_per_depth": {
            "2": {"1": 50},
            "3": {"2": 40, "3": 10},
        },
    }
    out = _to_numpy_depth_lookup(meta)
    assert out is not None
    counts, depths = out
    np.testing.assert_array_equal(counts, np.array([0, 1, 2, 3], dtype=np.int32))
    np.testing.assert_array_equal(depths, np.array([1, 2, 3, 3], dtype=np.int32))
