import jax.numpy as jnp
import numpy as np

from banyan_grid.environment.constants import (
    Colors,
    RULE_TYPE_COLLECT,
    RULE_TYPE_COMBINE,
    RULE_TYPE_DISTRACTOR_COMBINE,
    RULE_TYPE_TERNARY_COMBINE,
    RULE_TYPE_TRANSFORM,
)
from banyan_grid.tasks.ruleset_factory import (
    _pair_to_output_uid_global,
    _pair_to_output_uid_global_np,
    _triple_to_output_uid_global,
    _triple_to_output_uid_global_np,
    _uid_to_tuple,
    _unary_to_output_uid_global,
    _unary_to_output_uid_global_np,
)

_LEGACY_ITEM_BITS = 3
_LEGACY_ITEM_MASK = (1 << _LEGACY_ITEM_BITS) - 1
_COMPACT_ITEM_BITS = 4
_COMPACT_ITEM_MASK = (1 << _COMPACT_ITEM_BITS) - 1
_COMPACT_FLAG = np.uint32(1 << 31)
_PACKED_RULE_TYPE_TERNARY_COMPACT = np.uint32(0x7)
_FIXED_OUTPUT_BASE_SEED = 0
_NUM_REAL_COLORS = int(Colors.BLACK)


def pack_ternary_rule_meta_np(
    input_item_c: np.ndarray, required_adjacent: np.ndarray
) -> np.ndarray:
    return ((input_item_c.astype(np.uint32) & np.uint32(_COMPACT_ITEM_MASK)) << np.uint32(1)) | (
        required_adjacent.astype(np.uint32) & np.uint32(0x1)
    )


def unpack_ternary_rule_meta(meta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    meta = meta.astype(jnp.int32)
    return meta >> jnp.int32(1), meta & jnp.int32(0x1)


def _uid_from_item_color_np(item: np.ndarray, color: np.ndarray) -> np.ndarray:
    return item.astype(np.int32) * np.int32(_NUM_REAL_COLORS) + color.astype(np.int32)


def _uid_from_item_color(item: jnp.ndarray, color: jnp.ndarray) -> jnp.ndarray:
    return item.astype(jnp.int32) * jnp.int32(_NUM_REAL_COLORS) + color.astype(jnp.int32)


def _pair_uid_vec_np(uid_a: np.ndarray, uid_b: np.ndarray) -> np.ndarray:
    vec = np.vectorize(
        lambda a, b: _pair_to_output_uid_global_np(
            int(a),
            int(b),
            base_seed=_FIXED_OUTPUT_BASE_SEED,
        ),
        otypes=[np.int32],
    )
    return vec(uid_a, uid_b)


def _unary_uid_vec_np(uid_in: np.ndarray) -> np.ndarray:
    vec = np.vectorize(
        lambda a: _unary_to_output_uid_global_np(
            int(a),
            base_seed=_FIXED_OUTPUT_BASE_SEED,
        ),
        otypes=[np.int32],
    )
    return vec(uid_in)


def _triple_uid_vec_np(uid_a: np.ndarray, uid_b: np.ndarray, uid_c: np.ndarray) -> np.ndarray:
    vec = np.vectorize(
        lambda a, b, c: _triple_to_output_uid_global_np(
            int(a),
            int(b),
            int(c),
            base_seed=_FIXED_OUTPUT_BASE_SEED,
        ),
        otypes=[np.int32],
    )
    return vec(uid_a, uid_b, uid_c)


def pack_rules_uint32_np(rulesets_int32: np.ndarray) -> np.ndarray:
    """
    rulesets_int32: (..., R, 6) int32 in the canonical row encoding.
    Returns: (..., R) uint32 packed.

    Compatibility:
    - Legacy rows with item ids <= 7 keep the original explicit-output packing.
    - New rows that require item ids 8/9 use a compact packing that reconstructs
      outputs deterministically at unpack time.
    """
    rs = rulesets_int32.astype(np.int32, copy=False)
    rt = rs[..., 0].astype(np.uint32)
    out = rt & np.uint32(0x7)

    # Common views.
    packed_cols = rs[..., 5].astype(np.uint32)
    c1 = packed_cols & np.uint32(0xF)
    c2 = (packed_cols >> np.uint32(4)) & np.uint32(0xF)
    c3 = (packed_cols >> np.uint32(8)) & np.uint32(0xF)
    cout3 = (packed_cols >> np.uint32(12)) & np.uint32(0xF)
    cout_bin = (packed_cols >> np.uint32(8)) & np.uint32(0xF)

    # Collect.
    is_collect = rt == RULE_TYPE_COLLECT
    tile = rs[..., 1].astype(np.uint32) & np.uint32(0x1F)
    item_collect = rs[..., 2].astype(np.uint32)
    color_collect = rs[..., 3].astype(np.uint32) & np.uint32(0xF)
    use_compact_collect = is_collect & (item_collect > np.uint32(_LEGACY_ITEM_MASK))

    p_collect_legacy = (
        (rt & np.uint32(0x7))
        | (tile << np.uint32(3))
        | ((item_collect & np.uint32(_LEGACY_ITEM_MASK)) << np.uint32(8))
        | (color_collect << np.uint32(11))
    )
    p_collect_compact = (
        _COMPACT_FLAG
        | (rt & np.uint32(0x7))
        | (tile << np.uint32(3))
        | ((item_collect & np.uint32(_COMPACT_ITEM_MASK)) << np.uint32(8))
        | (color_collect << np.uint32(12))
    )
    out = np.where(use_compact_collect, p_collect_compact, out)
    out = np.where(is_collect & (~use_compact_collect), p_collect_legacy, out)

    # Binary-like rows: combine, distractor combine, transform.
    is_binary_like = (
        (rt == RULE_TYPE_COMBINE)
        | (rt == RULE_TYPE_DISTRACTOR_COMBINE)
        | (rt == RULE_TYPE_TRANSFORM)
    )
    in1 = rs[..., 1].astype(np.uint32)
    in2 = rs[..., 2].astype(np.uint32)
    out_item = rs[..., 3].astype(np.uint32)
    adj = rs[..., 4].astype(np.uint32) & np.uint32(0x1)

    use_compact_binary = is_binary_like & (
        (in1 > np.uint32(_LEGACY_ITEM_MASK))
        | (in2 > np.uint32(_LEGACY_ITEM_MASK))
        | (out_item > np.uint32(_LEGACY_ITEM_MASK))
    )

    p_binary_legacy = (
        (rt & np.uint32(0x7))
        | ((in1 & np.uint32(_LEGACY_ITEM_MASK)) << np.uint32(3))
        | ((in2 & np.uint32(_LEGACY_ITEM_MASK)) << np.uint32(6))
        | ((out_item & np.uint32(_LEGACY_ITEM_MASK)) << np.uint32(9))
        | (adj << np.uint32(12))
        | (c1 << np.uint32(13))
        | (c2 << np.uint32(17))
        | (cout_bin << np.uint32(21))
    )

    p_binary_compact = (
        _COMPACT_FLAG
        | (rt & np.uint32(0x7))
        | ((in1 & np.uint32(_COMPACT_ITEM_MASK)) << np.uint32(3))
        | ((in2 & np.uint32(_COMPACT_ITEM_MASK)) << np.uint32(7))
        | (c1 << np.uint32(11))
        | (c2 << np.uint32(15))
        | (adj << np.uint32(19))
    )

    out = np.where(use_compact_binary, p_binary_compact, out)
    out = np.where(is_binary_like & (~use_compact_binary), p_binary_legacy, out)

    # Validate compact binary outputs against the fixed mapping used at unpack time.
    if np.any(use_compact_binary):
        item1 = rs[..., 1].astype(np.int32)
        item2 = rs[..., 2].astype(np.int32)
        outi = rs[..., 3].astype(np.int32)
        uid1 = _uid_from_item_color_np(item1, c1.astype(np.int32))
        uid2 = _uid_from_item_color_np(item2, c2.astype(np.int32))
        out_uid = _uid_from_item_color_np(outi, cout_bin.astype(np.int32))
        derived_uid = np.where(
            rt == RULE_TYPE_TRANSFORM,
            _unary_uid_vec_np(uid1),
            _pair_uid_vec_np(uid1, uid2),
        )
        if np.any(use_compact_binary & (derived_uid != out_uid)):
            raise ValueError(
                "Compact producer rows must use the fixed output mapping used by the packed codec."
            )

    # Ternary rows.
    is_ternary = rt == RULE_TYPE_TERNARY_COMBINE
    meta = rs[..., 4].astype(np.uint32)
    in3 = (meta >> np.uint32(1)).astype(np.uint32)
    adj3 = meta & np.uint32(0x1)
    use_compact_ternary = is_ternary & (
        (in1 > np.uint32(_LEGACY_ITEM_MASK))
        | (in2 > np.uint32(_LEGACY_ITEM_MASK))
        | (in3 > np.uint32(_LEGACY_ITEM_MASK))
        | (out_item > np.uint32(_LEGACY_ITEM_MASK))
    )

    p_ternary_legacy = (
        (rt & np.uint32(0x7))
        | ((in1 & np.uint32(_LEGACY_ITEM_MASK)) << np.uint32(3))
        | ((in2 & np.uint32(_LEGACY_ITEM_MASK)) << np.uint32(6))
        | ((in3 & np.uint32(_LEGACY_ITEM_MASK)) << np.uint32(9))
        | ((out_item & np.uint32(_LEGACY_ITEM_MASK)) << np.uint32(12))
        | (adj3 << np.uint32(15))
        | (c1 << np.uint32(16))
        | (c2 << np.uint32(20))
        | (c3 << np.uint32(24))
        | (cout3 << np.uint32(28))
    )

    p_ternary_compact = (
        _PACKED_RULE_TYPE_TERNARY_COMPACT
        | ((in1 & np.uint32(_COMPACT_ITEM_MASK)) << np.uint32(3))
        | ((in2 & np.uint32(_COMPACT_ITEM_MASK)) << np.uint32(7))
        | ((in3 & np.uint32(_COMPACT_ITEM_MASK)) << np.uint32(11))
        | (c1 << np.uint32(15))
        | (c2 << np.uint32(19))
        | (c3 << np.uint32(23))
        | (adj3 << np.uint32(27))
    )

    out = np.where(use_compact_ternary, p_ternary_compact, out)
    out = np.where(is_ternary & (~use_compact_ternary), p_ternary_legacy, out)

    if np.any(use_compact_ternary):
        uid1 = _uid_from_item_color_np(rs[..., 1].astype(np.int32), c1.astype(np.int32))
        uid2 = _uid_from_item_color_np(rs[..., 2].astype(np.int32), c2.astype(np.int32))
        uid3 = _uid_from_item_color_np((meta >> np.uint32(1)).astype(np.int32), c3.astype(np.int32))
        out_uid = _uid_from_item_color_np(rs[..., 3].astype(np.int32), cout3.astype(np.int32))
        derived_uid = _triple_uid_vec_np(uid1, uid2, uid3)
        if np.any(use_compact_ternary & (derived_uid != out_uid)):
            raise ValueError(
                "Compact ternary rows must use the fixed output mapping used by the packed codec."
            )

    # Bounds validation.
    if np.any(is_collect & ((item_collect < 0) | (item_collect > np.uint32(_COMPACT_ITEM_MASK)))):
        raise ValueError("Collect item ids must fit in 4 bits.")
    if np.any(
        is_binary_like
        & (
            (in1 > np.uint32(_COMPACT_ITEM_MASK))
            | (in2 > np.uint32(_COMPACT_ITEM_MASK))
            | (out_item > np.uint32(_COMPACT_ITEM_MASK))
        )
    ):
        raise ValueError("Binary-like item ids must fit in 4 bits.")
    if np.any(
        is_ternary
        & (
            (in1 > np.uint32(_COMPACT_ITEM_MASK))
            | (in2 > np.uint32(_COMPACT_ITEM_MASK))
            | (in3 > np.uint32(_COMPACT_ITEM_MASK))
            | (out_item > np.uint32(_COMPACT_ITEM_MASK))
        )
    ):
        raise ValueError("Ternary item ids must fit in 4 bits.")

    return out.astype(np.uint32, copy=False)


def unpack_rules_uint32(packed: jnp.ndarray) -> jnp.ndarray:
    """
    packed: (..., R) uint32
    Returns: (..., R, 6) int32 in the canonical 6-field encoding.
    """
    p = packed.astype(jnp.uint32)
    rt_raw = p & jnp.uint32(0x7)
    is_compact_collect = (rt_raw == jnp.uint32(RULE_TYPE_COLLECT)) & (
        (p & jnp.uint32(_COMPACT_FLAG)) != 0
    )
    is_compact_binary = (
        ((rt_raw == jnp.uint32(RULE_TYPE_COMBINE))
        | (rt_raw == jnp.uint32(RULE_TYPE_DISTRACTOR_COMBINE))
        | (rt_raw == jnp.uint32(RULE_TYPE_TRANSFORM)))
        & ((p & jnp.uint32(_COMPACT_FLAG)) != 0)
    )
    is_compact_ternary = rt_raw == jnp.uint32(_PACKED_RULE_TYPE_TERNARY_COMPACT)

    rt = jnp.where(
        is_compact_ternary,
        jnp.int32(RULE_TYPE_TERNARY_COMBINE),
        rt_raw.astype(jnp.int32),
    )

    shape_out = p.shape + (6,)
    enc = jnp.zeros(shape_out, dtype=jnp.int32)
    enc = enc.at[..., 0].set(rt)

    is_collect = (rt == RULE_TYPE_COLLECT) & (~is_compact_collect)
    is_binary_like = (
        ((rt == RULE_TYPE_COMBINE) | (rt == RULE_TYPE_DISTRACTOR_COMBINE) | (rt == RULE_TYPE_TRANSFORM))
        & (~is_compact_binary)
    )
    is_ternary = (rt == RULE_TYPE_TERNARY_COMBINE) & (~is_compact_ternary)

    # Legacy collect decode.
    tile_c = ((p >> jnp.uint32(3)) & jnp.uint32(0x1F)).astype(jnp.int32)
    item_c = ((p >> jnp.uint32(8)) & jnp.uint32(_LEGACY_ITEM_MASK)).astype(jnp.int32)
    color_c = ((p >> jnp.uint32(11)) & jnp.uint32(0x0F)).astype(jnp.int32)
    enc = enc.at[..., 1].set(jnp.where(is_collect, tile_c, enc[..., 1]))
    enc = enc.at[..., 2].set(jnp.where(is_collect, item_c, enc[..., 2]))
    enc = enc.at[..., 3].set(jnp.where(is_collect, color_c, enc[..., 3]))

    # Compact collect decode.
    compact_tile = ((p >> jnp.uint32(3)) & jnp.uint32(0x1F)).astype(jnp.int32)
    compact_item = ((p >> jnp.uint32(8)) & jnp.uint32(_COMPACT_ITEM_MASK)).astype(jnp.int32)
    compact_color = ((p >> jnp.uint32(12)) & jnp.uint32(0x0F)).astype(jnp.int32)
    enc = enc.at[..., 1].set(jnp.where(is_compact_collect, compact_tile, enc[..., 1]))
    enc = enc.at[..., 2].set(jnp.where(is_compact_collect, compact_item, enc[..., 2]))
    enc = enc.at[..., 3].set(jnp.where(is_compact_collect, compact_color, enc[..., 3]))

    # Legacy binary-like decode.
    in1 = ((p >> jnp.uint32(3)) & jnp.uint32(_LEGACY_ITEM_MASK)).astype(jnp.int32)
    in2 = ((p >> jnp.uint32(6)) & jnp.uint32(_LEGACY_ITEM_MASK)).astype(jnp.int32)
    outi = ((p >> jnp.uint32(9)) & jnp.uint32(_LEGACY_ITEM_MASK)).astype(jnp.int32)
    adj = ((p >> jnp.uint32(12)) & jnp.uint32(0x01)).astype(jnp.int32)
    c1 = ((p >> jnp.uint32(13)) & jnp.uint32(0x0F)).astype(jnp.int32)
    c2 = ((p >> jnp.uint32(17)) & jnp.uint32(0x0F)).astype(jnp.int32)
    cout = ((p >> jnp.uint32(21)) & jnp.uint32(0x0F)).astype(jnp.int32)
    packed_cols = (c1 | (c2 << 4) | (cout << 8)).astype(jnp.int32)
    enc = enc.at[..., 1].set(jnp.where(is_binary_like, in1, enc[..., 1]))
    enc = enc.at[..., 2].set(jnp.where(is_binary_like, in2, enc[..., 2]))
    enc = enc.at[..., 3].set(jnp.where(is_binary_like, outi, enc[..., 3]))
    enc = enc.at[..., 4].set(jnp.where(is_binary_like, adj, enc[..., 4]))
    enc = enc.at[..., 5].set(jnp.where(is_binary_like, packed_cols, enc[..., 5]))

    # Compact binary-like decode.
    cin1 = ((p >> jnp.uint32(3)) & jnp.uint32(_COMPACT_ITEM_MASK)).astype(jnp.int32)
    cin2 = ((p >> jnp.uint32(7)) & jnp.uint32(_COMPACT_ITEM_MASK)).astype(jnp.int32)
    cc1 = ((p >> jnp.uint32(11)) & jnp.uint32(0x0F)).astype(jnp.int32)
    cc2 = ((p >> jnp.uint32(15)) & jnp.uint32(0x0F)).astype(jnp.int32)
    cadj = ((p >> jnp.uint32(19)) & jnp.uint32(0x01)).astype(jnp.int32)
    cuid1 = _uid_from_item_color(cin1, cc1)
    cuid2 = _uid_from_item_color(cin2, cc2)
    compact_out_uid = jnp.where(
        rt == RULE_TYPE_TRANSFORM,
        _unary_to_output_uid_global(cuid1, base_seed=_FIXED_OUTPUT_BASE_SEED),
        _pair_to_output_uid_global(
            cuid1,
            cuid2,
            base_seed=_FIXED_OUTPUT_BASE_SEED,
        ),
    )
    compact_out_item, compact_out_color = _uid_to_tuple(compact_out_uid)
    compact_packed_cols = (
        cc1
        | (jnp.where(rt == RULE_TYPE_TRANSFORM, 0, cc2) << 4)
        | (compact_out_color.astype(jnp.int32) << 8)
    ).astype(jnp.int32)
    enc = enc.at[..., 1].set(jnp.where(is_compact_binary, cin1, enc[..., 1]))
    enc = enc.at[..., 2].set(
        jnp.where(
            is_compact_binary,
            jnp.where(rt == RULE_TYPE_TRANSFORM, 0, cin2),
            enc[..., 2],
        )
    )
    enc = enc.at[..., 3].set(jnp.where(is_compact_binary, compact_out_item.astype(jnp.int32), enc[..., 3]))
    enc = enc.at[..., 4].set(jnp.where(is_compact_binary, cadj, enc[..., 4]))
    enc = enc.at[..., 5].set(jnp.where(is_compact_binary, compact_packed_cols, enc[..., 5]))

    # Legacy ternary decode.
    tin1 = ((p >> jnp.uint32(3)) & jnp.uint32(_LEGACY_ITEM_MASK)).astype(jnp.int32)
    tin2 = ((p >> jnp.uint32(6)) & jnp.uint32(_LEGACY_ITEM_MASK)).astype(jnp.int32)
    tin3 = ((p >> jnp.uint32(9)) & jnp.uint32(_LEGACY_ITEM_MASK)).astype(jnp.int32)
    tout = ((p >> jnp.uint32(12)) & jnp.uint32(_LEGACY_ITEM_MASK)).astype(jnp.int32)
    tadj = ((p >> jnp.uint32(15)) & jnp.uint32(0x01)).astype(jnp.int32)
    tc1 = ((p >> jnp.uint32(16)) & jnp.uint32(0x0F)).astype(jnp.int32)
    tc2 = ((p >> jnp.uint32(20)) & jnp.uint32(0x0F)).astype(jnp.int32)
    tc3 = ((p >> jnp.uint32(24)) & jnp.uint32(0x0F)).astype(jnp.int32)
    tcout = ((p >> jnp.uint32(28)) & jnp.uint32(0x0F)).astype(jnp.int32)
    tmeta = ((tin3 << 1) | tadj).astype(jnp.int32)
    tpacked_cols = (tc1 | (tc2 << 4) | (tc3 << 8) | (tcout << 12)).astype(jnp.int32)
    enc = enc.at[..., 1].set(jnp.where(is_ternary, tin1, enc[..., 1]))
    enc = enc.at[..., 2].set(jnp.where(is_ternary, tin2, enc[..., 2]))
    enc = enc.at[..., 3].set(jnp.where(is_ternary, tout, enc[..., 3]))
    enc = enc.at[..., 4].set(jnp.where(is_ternary, tmeta, enc[..., 4]))
    enc = enc.at[..., 5].set(jnp.where(is_ternary, tpacked_cols, enc[..., 5]))

    # Compact ternary decode.
    ct1 = ((p >> jnp.uint32(3)) & jnp.uint32(_COMPACT_ITEM_MASK)).astype(jnp.int32)
    ct2 = ((p >> jnp.uint32(7)) & jnp.uint32(_COMPACT_ITEM_MASK)).astype(jnp.int32)
    ct3 = ((p >> jnp.uint32(11)) & jnp.uint32(_COMPACT_ITEM_MASK)).astype(jnp.int32)
    cc1_t = ((p >> jnp.uint32(15)) & jnp.uint32(0x0F)).astype(jnp.int32)
    cc2_t = ((p >> jnp.uint32(19)) & jnp.uint32(0x0F)).astype(jnp.int32)
    cc3_t = ((p >> jnp.uint32(23)) & jnp.uint32(0x0F)).astype(jnp.int32)
    cadj_t = ((p >> jnp.uint32(27)) & jnp.uint32(0x01)).astype(jnp.int32)
    cuid1_t = _uid_from_item_color(ct1, cc1_t)
    cuid2_t = _uid_from_item_color(ct2, cc2_t)
    cuid3_t = _uid_from_item_color(ct3, cc3_t)
    compact_tout_uid = _triple_to_output_uid_global(
        cuid1_t,
        cuid2_t,
        cuid3_t,
        base_seed=_FIXED_OUTPUT_BASE_SEED,
    )
    compact_tout_item, compact_tout_color = _uid_to_tuple(compact_tout_uid)
    compact_tmeta = ((ct3 << 1) | cadj_t).astype(jnp.int32)
    compact_tpacked_cols = (
        cc1_t | (cc2_t << 4) | (cc3_t << 8) | (compact_tout_color.astype(jnp.int32) << 12)
    ).astype(jnp.int32)
    enc = enc.at[..., 1].set(jnp.where(is_compact_ternary, ct1, enc[..., 1]))
    enc = enc.at[..., 2].set(jnp.where(is_compact_ternary, ct2, enc[..., 2]))
    enc = enc.at[..., 3].set(
        jnp.where(is_compact_ternary, compact_tout_item.astype(jnp.int32), enc[..., 3])
    )
    enc = enc.at[..., 4].set(jnp.where(is_compact_ternary, compact_tmeta, enc[..., 4]))
    enc = enc.at[..., 5].set(jnp.where(is_compact_ternary, compact_tpacked_cols, enc[..., 5]))

    return enc


def unpack_rules_uint32_np(packed: np.ndarray) -> np.ndarray:
    """Decode packed uint32 rulesets to int32 ``[..., R, 6]`` (NumPy)."""
    packed_jax = jnp.asarray(packed, dtype=jnp.uint32)
    return np.asarray(unpack_rules_uint32(packed_jax), dtype=np.int32)


unpack_rules_uint32_jit = unpack_rules_uint32
