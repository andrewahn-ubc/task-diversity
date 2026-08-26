# Exact experiment definition

## Relationship to Banyan

The paper defines a task as a layout, a task-tree topology, and a concrete
object assignment. It uses depths 1 through 6, nested shallow tasks, terminal
rewards `+1` for the root goal and `-1` for a task-valid but target-invalid
operation, whole-grid observations, and the actions up/down/left/right/stay,
pickup/drop/toggle. This implementation retains those properties.

The authors had not released their code or full hyperparameter configuration
as of 2026-08-26. Consequently, the following details are explicit pilot
choices:

- one fixed 9x9 wall layout and fixed object slots in every condition;
- one deterministic concrete grounding for every abstract topology;
- deterministic global unary and binary object rules shared by all phases;
- 1,024 disjoint depth-6 topology families generated once, round-robin
  assigned to `d1` through `d4`, and nested at depths 1 through 6;
- recurrent convolutional PPO without Continual Backprop;
- 25,165,824 environment steps per phase (192 complete PPO rollouts), 1,024
  vector environments, and 1,024 independent evaluation episodes per curve
  point.

Thus the manipulated variable is the number of topology families sampled by a
phase (`1`, `16`, or `256`). Layout count, number of groundings per topology,
total updates, architecture, and catalog are otherwise matched. A condition is
a prefix of the same deterministic phase catalog, making the comparison paired
without allowing topology overlap across phases.

## PPO

The policy is a two-layer convolutional whole-grid encoder, a GRU with 256
hidden units, and separate linear policy/value heads. PPO uses 128-step
rollouts, 2 epochs, environment-sequence minibatches, Adam at `2.5e-4`,
`gamma=0.99`, GAE `lambda=0.95`, clip `0.2`, value coefficient `0.5`, entropy
coefficient `0.01`, and gradient-norm clipping at `0.5`. Recurrent state is
reset at episode boundaries and minibatches preserve whole rollout sequences.

## Phase measurements

Evaluation occurs before any update in each phase, every 1M environment steps,
and at phase end. `S_start(di)` and `S_end(di)` are therefore literal
independent-evaluation measurements. At each phase end the policy is also
evaluated on `d1`. The aggregator computes:

- `TransferGap_i = S_end(d{i-1}) - S_start(di)`;
- `SpecializationGain_i = S_end(di) - S_start(di)`; and
- `B(i,1) = performance(d1 after phase i) - performance(d1 after phase 1)`.

## Gradient diagnostic

At phase fractions 0, 0.5, and 1 in `d2` through `d4`, the current frozen
policy collects a fresh current-distribution rollout and a fresh rollout from
a balanced mixture of all previous distributions. The same PPO surrogate,
value, and entropy objective is differentiated separately. No optimizer step
occurs. Cosines are saved for all parameters and for shared, policy-head, and
value-head parameter groups. Each record also stores checkpoint success; the
aggregator joins it to phase-end success to obtain subsequent specialization.

## Compute choice

Narval exposes 40GB A100 GPUs. Each run requests one full A100, 6 CPU cores,
32GB RAM, and 10 hours. The 15 primary conditions are a SLURM array, so elapsed
time is governed by one run rather than the sum of all seeds when capacity is
available. Smoke jobs request one hour. The maximum requested time remains far
below Narval's 168-hour limit.
