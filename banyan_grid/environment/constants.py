import jax.numpy as jnp
from flax import struct

NUM_ACTIONS = 9
NUM_ITEMS = 10
MAX_RULE_ENCODING_LEN = 6
MAX_INVENTORY_SIZE = 1  # Maximum items an agent can hold at once
# Rule types for ruleset encodings
RULE_TYPE_COLLECT = 1
RULE_TYPE_COMBINE = 2
RULE_TYPE_DISTRACTOR_COMBINE = 3
RULE_TYPE_TRANSFORM = 4
# Value 5 is reserved: packed datasets encode inactive (padding) rule rows as
# uint32 value 5, and loaders count active rules via `packed != 5`.
RULE_TYPE_PAD = 5
RULE_TYPE_TERNARY_COMBINE = 6
NUM_RULE_TYPES = RULE_TYPE_TERNARY_COMBINE + 1


# NOTE: always keep BLACK at the end (used elsewhere in code to check value of last color)
class Colors(struct.PyTreeNode):
    RED: int = struct.field(pytree_node=False, default=0)
    GREEN: int = struct.field(pytree_node=False, default=1)
    BLUE: int = struct.field(pytree_node=False, default=2)
    PURPLE: int = struct.field(pytree_node=False, default=3)
    YELLOW: int = struct.field(pytree_node=False, default=4)
    GREY: int = struct.field(pytree_node=False, default=5)
    WHITE: int = struct.field(pytree_node=False, default=6)
    BROWN: int = struct.field(pytree_node=False, default=7)
    PINK: int = struct.field(pytree_node=False, default=8)
    ORANGE: int = struct.field(pytree_node=False, default=9)
    CYAN: int = struct.field(pytree_node=False, default=10)
    LIME: int = struct.field(pytree_node=False, default=11)
    BLACK: int = struct.field(pytree_node=False, default=12)


class TileType(struct.PyTreeNode):
    OPEN_FAST: int = struct.field(pytree_node=False, default=0)
    OPEN_MEDIUM: int = struct.field(pytree_node=False, default=1)
    BLOCK: int = struct.field(pytree_node=False, default=2)
    GOAL: int = struct.field(pytree_node=False, default=3)
    KEY: int = struct.field(pytree_node=False, default=4)
    BALL: int = struct.field(pytree_node=False, default=5)
    BALL_2: int = struct.field(pytree_node=False, default=6)
    BALL_3: int = struct.field(pytree_node=False, default=7)
    BALL_4: int = struct.field(pytree_node=False, default=8)
    MAP: int = struct.field(pytree_node=False, default=9)
    MAP_2: int = struct.field(pytree_node=False, default=10)
    MAP_3: int = struct.field(pytree_node=False, default=11)
    MAP_4: int = struct.field(pytree_node=False, default=12)
    TRIANGLE: int = struct.field(pytree_node=False, default=13)
    STAR: int = struct.field(pytree_node=False, default=14)
    HEX: int = struct.field(pytree_node=False, default=15)
    SQUARE: int = struct.field(pytree_node=False, default=16)
    PYRAMID: int = struct.field(pytree_node=False, default=17)
    DIAMOND: int = struct.field(pytree_node=False, default=18)
    CRESCENT: int = struct.field(pytree_node=False, default=19)


class ItemType(struct.PyTreeNode):
    KEY: int = struct.field(pytree_node=False, default=0)
    BALL: int = struct.field(pytree_node=False, default=1)
    MAP: int = struct.field(pytree_node=False, default=2)
    TRIANGLE: int = struct.field(pytree_node=False, default=3)
    STAR: int = struct.field(pytree_node=False, default=4)
    HEX: int = struct.field(pytree_node=False, default=5)
    SQUARE: int = struct.field(pytree_node=False, default=6)
    PYRAMID: int = struct.field(pytree_node=False, default=7)
    DIAMOND: int = struct.field(pytree_node=False, default=8)
    CRESCENT: int = struct.field(pytree_node=False, default=9)


RULESET_ITEM_TYPES = [
    ItemType.KEY,
    ItemType.BALL,
    ItemType.MAP,
    ItemType.TRIANGLE,
    ItemType.STAR,
    ItemType.HEX,
    ItemType.SQUARE,
    ItemType.PYRAMID,
    ItemType.DIAMOND,
    ItemType.CRESCENT,
]
NUM_RULESET_ITEMS = len(RULESET_ITEM_TYPES)


# NOTE: MERGE is a retained no-op kept for action-space compatibility; transform
# tasks still rely on TOGGLE.
class Action(struct.PyTreeNode):
    RIGHT: int = struct.field(pytree_node=False, default=0)
    LEFT: int = struct.field(pytree_node=False, default=1)
    UP: int = struct.field(pytree_node=False, default=2)
    DOWN: int = struct.field(pytree_node=False, default=3)
    STAY: int = struct.field(pytree_node=False, default=4)
    DROP: int = struct.field(pytree_node=False, default=5)
    PICKUP: int = struct.field(pytree_node=False, default=6)
    TOGGLE: int = struct.field(pytree_node=False, default=7)
    MERGE: int = struct.field(pytree_node=False, default=8)


MOVEMENT = jnp.array(
    [
        [0, 1],  # 0: Right
        [0, -1],  # 1: Left
        [-1, 0],  # 2: Up
        [1, 0],  # 3: Down
        [0, 0],  # 4: Stay
        [0, 0],  # 5: Drop
        [0, 0],  # 6: Pickup
        [0, 0],  # 7: Toggle
        [0, 0],  # 8: Merge
    ],
    dtype=jnp.int32,
)

# Adjacent positions (including diagonals)
ADJACENT_POSITIONS = jnp.array(
    [
        [-1, -1],  # Top-left
        [-1, 0],  # Top
        [-1, 1],  # Top-right
        [0, -1],  # Left
        [0, 1],  # Right
        [1, -1],  # Bottom-left
        [1, 0],  # Bottom
        [1, 1],  # Bottom-right
    ],
    dtype=jnp.int32,
)

COLLECTIBLE_ITEMS = [
    TileType.KEY,
    TileType.BALL,
    TileType.BALL_2,
    TileType.BALL_3,
    TileType.BALL_4,
    TileType.MAP,
    TileType.MAP_2,
    TileType.MAP_3,
    TileType.MAP_4,
    TileType.TRIANGLE,
    TileType.STAR,
    TileType.HEX,
    TileType.SQUARE,
    TileType.PYRAMID,
    TileType.DIAMOND,
    TileType.CRESCENT,
]

WALKABLE_TILES = jnp.array(
    [
        TileType.OPEN_FAST,
        TileType.OPEN_MEDIUM,
        TileType.GOAL,
        *COLLECTIBLE_ITEMS,
    ],
    dtype=jnp.int32,
)

NUM_TILE_TYPES = int(TileType.CRESCENT) + 1
WALKABLE_MASK = (
    jnp.zeros((NUM_TILE_TYPES,), dtype=jnp.bool_).at[WALKABLE_TILES].set(True)
)

INITIAL_AGENT_POSITION_TILES = jnp.array(
    [
        TileType.OPEN_FAST,
        TileType.OPEN_MEDIUM,
    ],
    dtype=jnp.int32,
)

DEFAULT_GRID_SIZE = 5
DEFAULT_MAX_STEPS = 20

NUM_COLORS = int(Colors.BLACK) + 1
NUM_CHANNELS = 2 + NUM_COLORS

DIR_TO_VEC = jnp.array(
    [
        [-1, 0],  # 0=Up
        [0, +1],  # 1=Right
        [+1, 0],  # 2=Down
        [0, -1],  # 3=Left
    ],
    dtype=jnp.int32,
)

COLLECTIBLE_TILES_ARRAY = jnp.array(COLLECTIBLE_ITEMS, dtype=jnp.int32)

ALL_COLORS = [
    Colors.RED,
    Colors.GREEN,
    Colors.BLUE,
    Colors.PURPLE,
    Colors.YELLOW,
    Colors.GREY,
    Colors.WHITE,
    Colors.BROWN,
    Colors.PINK,
    Colors.ORANGE,
    Colors.CYAN,
    Colors.LIME,
]

COLORED_ITEMS = []
for item in COLLECTIBLE_ITEMS:
    for color in ALL_COLORS:
        COLORED_ITEMS.append((item, color))

_ITEM_TO_TILE = {
    ItemType.KEY: TileType.KEY,
    ItemType.BALL: TileType.BALL,
    ItemType.MAP: TileType.MAP,
    ItemType.TRIANGLE: TileType.TRIANGLE,
    ItemType.STAR: TileType.STAR,
    ItemType.HEX: TileType.HEX,
    ItemType.SQUARE: TileType.SQUARE,
    ItemType.PYRAMID: TileType.PYRAMID,
    ItemType.DIAMOND: TileType.DIAMOND,
    ItemType.CRESCENT: TileType.CRESCENT,
}

_INV_ITEM_TO_TILE = {tile: item for item, tile in _ITEM_TO_TILE.items()}

ITEM_TO_TILE = jnp.array(
    [
        TileType.KEY,
        TileType.BALL,
        TileType.MAP,
        TileType.TRIANGLE,
        TileType.STAR,
        TileType.HEX,
        TileType.SQUARE,
        TileType.PYRAMID,
        TileType.DIAMOND,
        TileType.CRESCENT,
    ],
    dtype=jnp.int32,
)

_MAX_TILE = int(TileType.CRESCENT) + 1
TILE_TO_ITEM = jnp.full((_MAX_TILE,), -1, dtype=jnp.int32)
TILE_TO_ITEM = TILE_TO_ITEM.at[TileType.KEY].set(ItemType.KEY)
TILE_TO_ITEM = TILE_TO_ITEM.at[TileType.BALL].set(ItemType.BALL)
TILE_TO_ITEM = TILE_TO_ITEM.at[TileType.BALL_2].set(ItemType.BALL)
TILE_TO_ITEM = TILE_TO_ITEM.at[TileType.BALL_3].set(ItemType.BALL)
TILE_TO_ITEM = TILE_TO_ITEM.at[TileType.BALL_4].set(ItemType.BALL)
TILE_TO_ITEM = TILE_TO_ITEM.at[TileType.MAP].set(ItemType.MAP)
TILE_TO_ITEM = TILE_TO_ITEM.at[TileType.MAP_2].set(ItemType.MAP)
TILE_TO_ITEM = TILE_TO_ITEM.at[TileType.MAP_3].set(ItemType.MAP)
TILE_TO_ITEM = TILE_TO_ITEM.at[TileType.MAP_4].set(ItemType.MAP)
TILE_TO_ITEM = TILE_TO_ITEM.at[TileType.TRIANGLE].set(ItemType.TRIANGLE)
TILE_TO_ITEM = TILE_TO_ITEM.at[TileType.STAR].set(ItemType.STAR)
TILE_TO_ITEM = TILE_TO_ITEM.at[TileType.HEX].set(ItemType.HEX)
TILE_TO_ITEM = TILE_TO_ITEM.at[TileType.SQUARE].set(ItemType.SQUARE)
TILE_TO_ITEM = TILE_TO_ITEM.at[TileType.PYRAMID].set(ItemType.PYRAMID)
TILE_TO_ITEM = TILE_TO_ITEM.at[TileType.DIAMOND].set(ItemType.DIAMOND)
TILE_TO_ITEM = TILE_TO_ITEM.at[TileType.CRESCENT].set(ItemType.CRESCENT)


class T:
    KEY = TileType.KEY
    BALL = TileType.BALL
    MAP = TileType.MAP
    STAR = TileType.STAR
    HEX = TileType.HEX
    SQUARE = TileType.SQUARE
    PYRAMID = TileType.PYRAMID
    DIAMOND = TileType.DIAMOND
    CRESCENT = TileType.CRESCENT


LEAFABLE = [
    T.KEY,
    T.BALL,
    T.MAP,
    T.STAR,
    T.HEX,
    T.SQUARE,
    T.PYRAMID,
    T.DIAMOND,
    T.CRESCENT,
]
COMBINABLE = LEAFABLE
NEARABLE = [T.KEY, T.BALL, T.STAR, T.DIAMOND]
