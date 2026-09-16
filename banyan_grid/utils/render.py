from __future__ import annotations

import functools
from typing import Callable, cast

import jax
import jax.numpy as jnp

# Stay in sync with your env constants
from banyan_grid.environment.constants import (
    NUM_ITEMS,
    Colors,
)

# Palette (RGB in 0..255)
_PALETTE = jnp.array(
    [
        [255, 0, 0],  # 0  RED
        [27, 174, 96],  # 1  GREEN
        [0, 0, 255],  # 2  BLUE
        [112, 39, 195],  # 3  PURPLE
        [241, 196, 15],  # 4  YELLOW
        [100, 100, 100],  # 5  GREY
        [255, 255, 255],  # 6  WHITE
        [165, 42, 42],  # 7  BROWN
        [255, 20, 147],  # 8  PINK
        [255, 165, 0],  # 9  ORANGE
        [0, 206, 209],  # 10 CYAN
        [50, 205, 50],  # 11 LIME
        [0, 0, 0],  # 12 BLACK (sentinel)
    ],
    dtype=jnp.float32,
)


def _col(idx):
    return _PALETTE[idx]


COL = _col
COL_BLACK = COL(Colors.BLACK)
COL_WHITE = COL(Colors.WHITE)
COL_GREEN = COL(Colors.GREEN)
COL_YELLOW = COL(Colors.YELLOW)
COL_GREY = COL(Colors.GREY)
COL_PURPLE = COL(Colors.PURPLE)
COL_BLUE = COL(Colors.BLUE)
COL_RED = COL(Colors.RED)
COL_PINK = COL(Colors.PINK)
COL_BROWN = COL(Colors.BROWN)
COL_ORANGE = COL(Colors.ORANGE)
COL_CYAN = COL(Colors.CYAN)
COL_LIME = COL(Colors.LIME)
COL_DARK = jnp.array([44, 62, 80], dtype=jnp.float32)

# Agent overlay colors
AGENT_COLORS = jnp.stack(
    [
        jnp.array([52, 152, 219], dtype=jnp.float32),
        jnp.array([230, 126, 34], dtype=jnp.float32),
    ],
    axis=0,
)

# Layout
# Agent icon overlays (on tiles)
_ICON_SIZE = 10
_ICON_GAP = 2

# Footer layout
_FOOTER_PADDING = 10
_FOOTER_IDENT_HEIGHT = 20
_FOOTER_ITEM_SIZE = 28
_FOOTER_ITEM_GAP = 6

# Header height
_HEADER_HEIGHT = 24

# Default item colors when inventory color is BLACK
_DEFAULT_ITEM_COLORS = jnp.stack(
    [
        COL_YELLOW,  # KEY
        COL_RED,  # BALL
        COL_WHITE,  # MAP
        COL_ORANGE,  # TRIANGLE
        COL_YELLOW,  # STAR
        COL_PURPLE,  # HEX
        COL_BLUE,  # SQUARE
        COL_BROWN,  # PYRAMID
        COL_CYAN,  # DIAMOND
        COL_LIME,  # CRESCENT
    ],
    axis=0,
)


# Geometry
def _coords(tile_px: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    xs = jnp.linspace(0.0, 1.0, tile_px, endpoint=False) + 0.5 / tile_px
    X, Y = jnp.meshgrid(xs, xs, indexing="xy")
    return X, Y


def _coords_hw(h: int, w: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    xs = jnp.linspace(0.0, 1.0, w, endpoint=False) + 0.5 / w
    ys = jnp.linspace(0.0, 1.0, h, endpoint=False) + 0.5 / h
    X, Y = jnp.meshgrid(xs, ys, indexing="xy")
    return X, Y


def _rect_mask(X, Y, xmin, xmax, ymin, ymax):
    return (X >= xmin) & (X <= xmax) & (Y >= ymin) & (Y <= ymax)


def _circle_mask(X, Y, cx, cy, r):
    return (X - cx) ** 2 + (Y - cy) ** 2 <= (r**2)


def _triangle_mask(X, Y, a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c
    v0x, v0y = bx - ax, by - ay
    v1x, v1y = cx - ax, cy - ay
    v2x, v2y = X - ax, Y - ay
    d00 = v0x * v0x + v0y * v0y
    d01 = v0x * v1x + v0y * v1y
    d11 = v1x * v1x + v1y * v1y
    d20 = v2x * v0x + v2y * v0y
    d21 = v2x * v1x + v2y * v1y
    denom = d00 * d11 - d01 * d01
    u = (d11 * d20 - d01 * d21) / (denom + 1e-8)
    v = (d00 * d21 - d01 * d20) / (denom + 1e-8)
    return (u >= 0.0) & (v >= 0.0) & ((u + v) <= 1.0)


def _rotate_coords(X, Y, theta: jax.Array | float, cx: float = 0.5, cy: float = 0.5):
    Xc, Yc = X - cx, Y - cy
    ct, st = jnp.cos(theta), jnp.sin(theta)
    Xr = Xc * ct - Yc * st + cx
    Yr = Xc * st + Yc * ct + cy
    return Xr, Yr


def _paint(img, mask, color):
    c = (color / 255.0)[None, None, :]
    m = mask[..., None]
    return jnp.where(m, c, img)


# Floor
def _floor_with_border(X, Y, color=COL_BLACK):
    img = _paint(
        jnp.ones((X.shape[0], X.shape[1], 3), dtype=jnp.float32),
        jnp.ones_like(X, dtype=bool),
        color,
    )
    border = (X < 0.03) | (Y < 0.03)
    img = _paint(img, border, COL_GREY)
    return img


def _open_fast(X, Y):
    return _floor_with_border(X, Y, COL_BLACK)


def _open_medium(X, Y):
    img = _floor_with_border(X, Y, COL_BLACK)
    stripes = jnp.mod(X + Y, 0.20) < 0.06
    img = _paint(img, stripes, COL_WHITE * 0.25 + COL_BLACK * 0.75)
    return img


def _open_slow(X, Y):
    img = _floor_with_border(X, Y, COL_BLACK)
    stripes1 = jnp.mod(X + Y, 0.15) < 0.05
    stripes2 = jnp.mod(X - Y, 0.15) < 0.05
    img = _paint(img, stripes1 | stripes2, COL_WHITE * 0.20 + COL_BLACK * 0.80)
    return img


def _block(X, Y):
    return _paint(
        jnp.ones((X.shape[0], X.shape[1], 3), dtype=jnp.float32),
        jnp.ones_like(X, dtype=bool),
        COL_GREY,
    )


def _goal(X, Y):
    return _floor_with_border(X, Y, COL_GREEN)


# Collectibles
def _key(X, Y, color):
    img = _floor_with_border(X, Y)
    img = _paint(img, _rect_mask(X, Y, 0.50, 0.63, 0.31, 0.88), color)
    img = _paint(img, _rect_mask(X, Y, 0.38, 0.50, 0.59, 0.66), color)
    img = _paint(img, _rect_mask(X, Y, 0.38, 0.50, 0.81, 0.88), color)
    img = _paint(img, _circle_mask(X, Y, 0.56, 0.28, 0.19), color)
    img = _paint(img, _circle_mask(X, Y, 0.56, 0.28, 0.06), COL_BLACK)
    return img


def _ball(X, Y, color):
    img = _floor_with_border(X, Y)
    img = _paint(img, _circle_mask(X, Y, 0.5, 0.5, 0.31), color)
    return img


def _map_tile(X, Y, color):
    img = _floor_with_border(X, Y)
    img = _paint(img, _rect_mask(X, Y, 0.1, 0.9, 0.1, 0.9), color)
    return img


def _shovel(X, Y, color):
    img = _floor_with_border(X, Y)
    img = _paint(img, _rect_mask(X, Y, 0.45, 0.55, 0.40, 0.80), COL_GREY)  # handle
    dx = (X - 0.5) / 0.25
    dy = (Y - 0.35) / 0.30
    half_ellipse = (dx * dx + dy * dy <= 1.0) & (Y <= 0.35)
    img = _paint(img, half_ellipse, color)
    img = _paint(img, _rect_mask(X, Y, 0.45, 0.55, 0.35, 0.40), COL_GREY)
    return img


def _star(X, Y, color):
    img = _floor_with_border(X, Y)
    bar_v = _rect_mask(X, Y, 0.47, 0.53, 0.25, 0.75)
    bar_h = _rect_mask(X, Y, 0.25, 0.75, 0.47, 0.53)
    d1 = (jnp.abs((Y - 0.5) - (X - 0.5)) < 0.03) & _rect_mask(X, Y, 0.3, 0.7, 0.3, 0.7)
    d2 = (jnp.abs((Y - 0.5) + (X - 0.5)) < 0.03) & _rect_mask(X, Y, 0.3, 0.7, 0.3, 0.7)
    img = _paint(img, bar_v | bar_h | d1 | d2, color)
    return img


def _hexagon(X, Y, color):
    img = _floor_with_border(X, Y)
    hx = jnp.abs(X - 0.5)
    hy = jnp.abs(Y - 0.5)
    mask = (hx <= 0.30) & (hy * 1.732 + hx <= 0.60)
    return _paint(img, mask, color)


def _square(X, Y, color):
    img = _floor_with_border(X, Y)
    return _paint(img, _rect_mask(X, Y, 0.25, 0.75, 0.25, 0.75), color)


def _pyramid(X, Y, color):
    img = _floor_with_border(X, Y)
    mask = _triangle_mask(
        X, Y, jnp.array([0.20, 0.75]), jnp.array([0.50, 0.25]), jnp.array([0.80, 0.75])
    )
    return _paint(img, mask, color)


def _triangle(X, Y, color):
    img = _floor_with_border(X, Y)
    mask = _triangle_mask(
        X, Y, jnp.array([0.20, 0.25]), jnp.array([0.80, 0.25]), jnp.array([0.50, 0.80])
    )
    return _paint(img, mask, color)


def _diamond(X, Y, color):
    img = _floor_with_border(X, Y)
    mask = (jnp.abs(X - 0.5) + jnp.abs(Y - 0.5)) <= 0.34
    return _paint(img, mask, color)


def _crescent(X, Y, color):
    img = _floor_with_border(X, Y)
    outer = _circle_mask(X, Y, 0.48, 0.5, 0.28)
    inner = _circle_mask(X, Y, 0.60, 0.46, 0.24)
    return _paint(img, outer & (~inner), color)


# Interactive tiles
def _pick_color(idx: int, default_color: jnp.ndarray) -> jnp.ndarray:
    return jax.lax.cond(idx != Colors.BLACK, lambda: COL(idx), lambda: default_color)


# Tile rendering
def _render_tile_case(X, Y, tile_type: int, color_idx: int) -> jnp.ndarray:
    return jax.lax.switch(
        tile_type,
        [
            lambda: _open_fast(X, Y),  # 0
            lambda: _open_medium(X, Y),  # 1
            lambda: _block(X, Y),  # 2
            lambda: _goal(X, Y),  # 3
            lambda: _key(X, Y, _pick_color(color_idx, COL_YELLOW)),  # 4 KEY
            lambda: _ball(X, Y, _pick_color(color_idx, COL_WHITE)),  # 5 BALL
            lambda: _ball(X, Y, _pick_color(color_idx, COL_BLUE)),  # 6 BALL_2
            lambda: _ball(X, Y, _pick_color(color_idx, COL_PURPLE)),  # 7 BALL_3
            lambda: _ball(X, Y, _pick_color(color_idx, COL_GREY)),  # 8 BALL_4
            lambda: _map_tile(X, Y, _pick_color(color_idx, COL_WHITE)),  # 9 MAP
            lambda: _map_tile(X, Y, _pick_color(color_idx, COL_BLUE)),  # 10 MAP_2
            lambda: _map_tile(X, Y, _pick_color(color_idx, COL_PURPLE)),  # 11 MAP_3
            lambda: _map_tile(X, Y, _pick_color(color_idx, COL_PINK)),  # 12 MAP_4
            lambda: _triangle(X, Y, _pick_color(color_idx, COL_ORANGE)),  # 13 TRIANGLE
            lambda: _star(X, Y, _pick_color(color_idx, COL_YELLOW)),  # 14 STAR
            lambda: _hexagon(X, Y, _pick_color(color_idx, COL_PURPLE)),  # 15 HEX
            lambda: _square(X, Y, _pick_color(color_idx, COL_BLUE)),  # 16 SQUARE
            lambda: _pyramid(X, Y, _pick_color(color_idx, COL_BROWN)),  # 17 PYRAMID
            lambda: _diamond(X, Y, _pick_color(color_idx, COL_CYAN)),  # 18 DIAMOND
            lambda: _crescent(X, Y, _pick_color(color_idx, COL_LIME)),  # 19 CRESCENT
        ],
    )


def _render_tiles_grid(
    map_array: jnp.ndarray, color_map: jnp.ndarray, tile_px: int
) -> jnp.ndarray:
    H, W = map_array.shape
    X, Y = _coords(tile_px)

    def one(tt, cc):
        return _render_tile_case(X, Y, tt, cc)

    tiles = jax.vmap(one)(map_array.reshape(-1), color_map.reshape(-1))
    tiles = tiles.reshape(H, W, tile_px, tile_px, 3)
    tiles = jnp.transpose(tiles, (0, 2, 1, 3, 4))
    board = tiles.reshape(H * tile_px, W * tile_px, 3)
    return board


# Agents + icons
def _agent_triangle_mask(tile_px: int, direction: jax.Array | int) -> jnp.ndarray:
    X, Y = _coords(tile_px)
    a = jnp.array([0.12, 0.20])
    b = jnp.array([0.87, 0.50])
    c = jnp.array([0.12, 0.80])
    theta = jnp.array([jnp.pi / 2, 0.0, -jnp.pi / 2, jnp.pi])[direction % 4]
    Xr, Yr = _rotate_coords(X, Y, theta)
    return _triangle_mask(Xr, Yr, a, b, c)


def _overlay_patch(img, patch, y0, x0):
    return jax.lax.dynamic_update_slice(img, patch, (y0, x0, 0))


def _slice_patch(img, y0, x0, tile_px):
    return jax.lax.dynamic_slice(img, (y0, x0, 0), (tile_px, tile_px, 3))


def _draw_agents_and_icons(
    board: jnp.ndarray,
    positions: jnp.ndarray,
    directions: jnp.ndarray,
    inventories: jnp.ndarray,
    inventory_colors: jnp.ndarray,
    tile_px: int,
) -> jnp.ndarray:
    img = board
    icon_size = _ICON_SIZE
    gap = _ICON_GAP

    X, Y = _coords(tile_px)

    pos = positions
    dir_i = directions % 4
    y0 = pos[0] * tile_px
    x0 = pos[1] * tile_px

    tile = _slice_patch(img, y0, x0, tile_px)
    tri = _agent_triangle_mask(tile_px, dir_i)
    col = AGENT_COLORS[0]
    tile = _paint(tile, tri, col * 0.9 + COL_BLACK * 0.1)

    def one_item(j, timg):
        has_item = inventories[j] > 0
        col_idx = inventory_colors[j]
        col_item = jax.lax.cond(
            col_idx != Colors.BLACK,
            lambda: COL(col_idx),
            lambda: _DEFAULT_ITEM_COLORS[j],
        )

        x1 = tile_px - (j + 1) * (icon_size + gap)
        y1 = 0
        x2 = x1 + icon_size
        y2 = y1 + icon_size

        x1f, x2f = jnp.clip(x1 / tile_px, 0, 1), jnp.clip(x2 / tile_px, 0, 1)
        y1f, y2f = jnp.clip(y1 / tile_px, 0, 1), jnp.clip(y2 / tile_px, 0, 1)

        mask = _rect_mask(X, Y, x1f, x2f, y1f, y2f) & has_item
        return _paint(timg, mask, col_item)

    tile = jax.lax.fori_loop(0, NUM_ITEMS, one_item, tile)
    return _overlay_patch(img, tile, y0, x0)


# Footer
def _draw_footer(
    board: jnp.ndarray,
    inventories: jnp.ndarray,
    inventory_colors: jnp.ndarray,
    tile_px: int,
) -> jnp.ndarray:
    H_px, W_px, _ = board.shape
    footer_h = tile_px
    header_h = _HEADER_HEIGHT

    full = jnp.pad(
        board,
        ((0, header_h + footer_h), (0, 0), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    header = _paint(
        jnp.ones((header_h, W_px, 3), dtype=jnp.float32),
        jnp.ones((header_h, W_px), dtype=bool),
        COL_DARK,
    )
    full = _overlay_patch(full, header, H_px, 0)

    footer = _paint(
        jnp.ones((footer_h, W_px, 3), dtype=jnp.float32),
        jnp.ones((footer_h, W_px), dtype=bool),
        COL_DARK,
    )

    padding = _FOOTER_PADDING
    slot_w = tile_px
    ident_h = _FOOTER_IDENT_HEIGHT
    item_sz = _FOOTER_ITEM_SIZE

    Xf, Yf = _coords_hw(footer_h, W_px)

    x_left = 0

    # ID bar
    mask_id = _rect_mask(Xf, Yf, x_left / W_px, (x_left + slot_w) / W_px, 0.0, ident_h / footer_h)
    footer = _paint(footer, mask_id, AGENT_COLORS[0])

    def one_item(j, fim):
        has = inventories[j] > 0
        col_idx = inventory_colors[j]
        col = jax.lax.cond(
            col_idx != Colors.BLACK,
            lambda: COL(col_idx),
            lambda: _DEFAULT_ITEM_COLORS[j],
        )
        xi = x_left + j * (item_sz + _FOOTER_ITEM_GAP)
        mask = _rect_mask(
            Xf,
            Yf,
            xi / W_px,
            (xi + item_sz) / W_px,
            (ident_h + padding) / footer_h,
            (ident_h + padding + item_sz) / footer_h,
        )
        return _paint(fim, mask & has, col)

    footer = jax.lax.fori_loop(0, NUM_ITEMS, one_item, footer)
    return _overlay_patch(full, footer, H_px + header_h, 0)


# JIT entrypoint
@functools.partial(jax.jit, static_argnames=("tile_size",))
def _render_grid_jitted(
    map_array: jnp.ndarray,
    color_map: jnp.ndarray,
    agent_positions: jnp.ndarray,
    directions: jnp.ndarray,
    inventories: jnp.ndarray,
    inventory_colors: jnp.ndarray,
    tile_size: int = 128,
) -> jnp.ndarray:
    board = _render_tiles_grid(map_array, color_map, tile_size)
    board = _draw_agents_and_icons(
        board, agent_positions, directions, inventories, inventory_colors, tile_size
    )
    board = _draw_footer(board, inventories, inventory_colors, tile_size)
    return jnp.clip(board * 255.0, 0.0, 255.0).round().astype(jnp.uint8)


@functools.partial(jax.jit, static_argnames=("tile_size",), backend="cpu")
def _render_grid_jitted_cpu(
    map_array: jnp.ndarray,
    color_map: jnp.ndarray,
    agent_positions: jnp.ndarray,
    directions: jnp.ndarray,
    inventories: jnp.ndarray,
    inventory_colors: jnp.ndarray,
    tile_size: int = 128,
) -> jnp.ndarray:
    board = _render_tiles_grid(map_array, color_map, tile_size)
    board = _draw_agents_and_icons(
        board, agent_positions, directions, inventories, inventory_colors, tile_size
    )
    board = _draw_footer(board, inventories, inventory_colors, tile_size)
    return jnp.clip(board * 255.0, 0.0, 255.0).round().astype(jnp.uint8)


RenderFn = Callable[
    [jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, int], jax.Array
]
_render_grid_jitted_fn = cast(RenderFn, _render_grid_jitted)
_render_grid_jitted_cpu_fn = cast(RenderFn, _render_grid_jitted_cpu)


# Public API
def render_grid(
    grid_map,
    color_map,
    agent_positions,
    directions,
    inventories,
    inventory_colors,
    step: int = 0,  # Unused, kept for backward compatibility
    *,
    tile_size: int = 128,
):
    """Render a grid world state to a uint8 image."""
    m = jnp.asarray(grid_map, dtype=jnp.int32)
    cm = jnp.asarray(color_map, dtype=jnp.int32)
    pos = jnp.asarray(agent_positions, dtype=jnp.int32)
    dirs = jnp.asarray(directions, dtype=jnp.int32)
    inv = jnp.asarray(inventories, dtype=jnp.int32)
    inv_colors = jnp.asarray(inventory_colors, dtype=jnp.int32)

    img_u8 = _render_grid_jitted_fn(m, cm, pos, dirs, inv, inv_colors, tile_size)
    return jax.device_get(img_u8)


def render_grid_cpu(
    grid_map,
    color_map,
    agent_positions,
    directions,
    inventories,
    inventory_colors,
    step: int = 0,
    *,
    tile_size: int = 128,
):
    m = jnp.asarray(grid_map, dtype=jnp.int32)
    cm = jnp.asarray(color_map, dtype=jnp.int32)
    pos = jnp.asarray(agent_positions, dtype=jnp.int32)
    dirs = jnp.asarray(directions, dtype=jnp.int32)
    inv = jnp.asarray(inventories, dtype=jnp.int32)
    inv_colors = jnp.asarray(inventory_colors, dtype=jnp.int32)

    img_u8 = _render_grid_jitted_cpu_fn(m, cm, pos, dirs, inv, inv_colors, tile_size)
    return jax.device_get(img_u8)
