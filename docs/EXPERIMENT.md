# Repaired learnability experiment

## Purpose

This experiment asks a prerequisite question: can the reconstructed
Banyan-inspired environment support reliable depth-6 learning at low diversity?
It contains one task distribution and therefore cannot measure forward
transfer, backward transfer, continual specialization, or gradient conflict
across distributions. No continual job is submitted.

## Factorial design

There are 16 independent runs:

| Factor | Levels |
|---|---|
| Algorithm | PPO, PPO + CBP |
| Layout/topology diversity | `n=1`, `n=4` |
| Catalog seed | `260600880`, `260600881` |
| Training seed | `0`, `1` |

For diversity `n`, the environment independently samples one of `n` layouts
and one of `n` nested topology families. This exposes all `n^2`
layout-topology combinations and six depths per combination. Each topology has
one deterministic object grounding. Layouts within a catalog are rotations of
the same procedurally generated connected base map, including rotated agent
starts and object placements. Objects occupy the nearest reachable cells and
are at most three moves from the start. The wall maps remain distinct,
connected, and isometric rotations. This deliberately prevents the first
binary merge from becoming a long navigation/exploration problem.

The catalog seed generates both the topologies and the base layout family. It
therefore measures task-set variation that the earlier five optimizer seeds
did not cover. Training seeds control initialization, action sampling, task
sampling, and PPO minibatch order.

## Relationship to Banyan

The paper defines a task using a layout, task-tree topology, and concrete
object assignment. It uses nested depths 1 through 6, whole-grid observations,
fixed object-interaction dynamics, terminal `+1` success, terminal `-1`
dead-end penalties, and the actions move/stay/pickup/drop/toggle. This
implementation retains those qualitative properties.

The authors' code and full hyperparameters were unavailable when this
repository was built. The deterministic object algebra, topology generator,
layout generator, object features, model details, and recurrent CBP rule are
local reconstruction choices. Passing this diagnostic establishes
learnability only in this reconstruction; it does not make it an exact Banyan
replication.

## Diagnosed failure and repair

The earlier learnability run was not merely slow at depths 3 through 6. Across
703 valid evaluations, depth-5 and depth-6 success were exactly zero. Direct
checkpoint rollouts showed a typical depth-3 policy repeatedly selecting
`DROP` at its start position until timeout. Random exploration found only two
depth-3 successes in 10.24 million transitions and no deeper successes. The
agent had learned navigation and unary toggles, but almost never received a
positive example of binary composition.

Two implementation choices amplified that sparse-reward failure. The random
`n=1` topology could make both shallow tasks unary, so the intended shallow
curriculum never taught pickup/drop/merge. Object placement also deliberately
maximized distance. Finally, an interaction absent from the current task was
always fatal, even when that object pair had no transformation in the global
dynamics. The repaired environment distinguishes globally valid but
task-wrong transformations (a terminal dead end) from physically invalid
interactions (a no-op), as required by the fixed-dynamics task definition.

Every distribution now starts with a nested anchor: depth 1 requires one
binary merge, and each deeper level adds one operation. Random topologies still
fill the remainder of the catalog. This makes `n=1` a controlled repeated
composition test rather than an accidental unary-only condition.

## Finite training curriculum

The scientific budget is unchanged. Its first half is split into six complete,
batch-aligned stages:

```text
8,388,608 transitions/stage x maximum depths 1, 2, 3, 4, 5, 6
= 50,331,648 bootstrap transitions
```

At a stage with maximum depth `d`, the environment samples uniformly from
depths 1 through `d`. Dead ends still terminate during this bootstrap, but
their reward is 0 rather than -1. This is important: before the first merge is
discovered, the original penalty makes "never manipulate anything" a locally
safe policy. A focused CPU experiment confirmed that neutralizing this
penalty changed a one-merge task from 0% to 53% success within 262,144
transitions; increasing entropy or simply presenting shallow tasks did not.

The remaining 50,331,648 transitions sample all six depths under the original
`+1` success and `-1` dead-end rewards. All evaluations—including those made
during curriculum stages—always use all six depths and the original reward.
Thus the curriculum affects training exploration, not the reported task or
the pass criterion. Curriculum stage, active maximum depth, and training
dead-end reward are recorded in every evaluation and optimization record.

## PPO and compute

Each run uses 100,663,296 environment transitions. The recurrent batch is:

```text
256 environments x 256 recurrent steps = 65,536 transitions/update
100,663,296 / 65,536 = 1,536 PPO updates
```

This doubles the recurrent window, quarters the old environment count, halves
the old rollout batch, and produces four times the old number of PPO update
cycles per phase. Each PPO update uses two epochs and four complete-environment
minibatches of 64 sequences, for 12,288 optimizer minibatch steps over the run.
Other settings remain Adam `2.5e-4`, `gamma=0.99`, GAE `lambda=0.95`, clip
`0.2`, value coefficient `0.5`, entropy coefficient `0.01`, and gradient norm
clip `0.5`.

The 256-step recurrent window exceeds the 192-step episode horizon, so an
episode that begins at a rollout boundary can be differentiated end-to-end.
Episodes that cross a boundary retain their GRU state, but gradients do not
cross between rollouts.

## CBP control

PPO + CBP uses contribution utility, replacement rate `1e-4`, decay `0.99`,
and maturity 100 optimizer minibatches. Plain PPO uses standard Adam with the
same learning settings. Because this is the first and only distribution, a
plain-PPO advantage would implicate the custom recurrent CBP adaptation rather
than ordinary loss of plasticity from repeated distribution shifts.

## Paper-aligned evaluation

Evaluation occurs before training and every 2,097,152 transitions through the
100,663,296-step endpoint. Each checkpoint uses exactly 1,536 episodes: 256
independent episodes at each depth. The reported overall success rate is:

```text
(success_depth_1 + ... + success_depth_6) / 6
```

Timeout, dead-end, action, and effective-manipulation headline rates use the
same equal-depth average. This prevents shallow episodes from receiving extra
weight merely because they terminate faster. Raw depth-specific success,
timeout, dead-end, and effective-manipulation rates are retained.

An effective manipulation is a pickup, drop, merge, or toggle that changes the
environment state. The logs separately retain action-attempt rates, successful
merge/toggle rates, movement effectiveness, and the fraction of actions that
produce no state change.

## Validation criterion

The diagnostic gate passes only when, for PPO + CBP at both `n=1` and `n=4`,
and separately for every depth 1 through 6:

1. mean endpoint success across the four catalog/training run clusters
   is at least 0.10; and
2. at least three of four clusters achieve success of at least 0.05.

The threshold is fixed in advance and never changes the training budget. The
analysis and figures are written whether the gate passes or fails. Nothing
continual is conditionally launched.
