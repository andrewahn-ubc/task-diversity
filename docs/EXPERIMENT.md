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
- recurrent convolutional PPO with Continual Backprop (CBP) applied to both
  convolutional feature layers, the pre-GRU layer, and GRU hidden features;
- 50,331,648 environment steps per phase (384 complete PPO rollouts), 1,024
  vector environments, and 1,024 independent evaluation episodes per curve
  point.

Thus the manipulated variable is the number of topology families sampled by a
phase (`1`, `4`, `16`, `64`, or `256`). Layout count, number of groundings per topology,
total updates, architecture, and catalog are otherwise matched. A condition is
a prefix of the same deterministic phase catalog, making the comparison paired
without allowing topology overlap across phases.

The original `n=1`, `n=16`, and `n=256` conditions remain the low,
intermediate, and high anchor points used by the preregistered qualitative
criteria. The added `n=4` and `n=64` conditions estimate the shape of the
diversity-response curve without redefining those criteria after observing
results.

## PPO

The policy is a two-layer convolutional whole-grid encoder, a GRU with 256
hidden units, and separate linear policy/value heads. PPO uses 128-step
rollouts, 2 epochs, environment-sequence minibatches, Adam at `2.5e-4`,
`gamma=0.99`, GAE `lambda=0.95`, clip `0.2`, value coefficient `0.5`, entropy
coefficient `0.01`, and gradient-norm clipping at `0.5`. Recurrent state is
reset at episode boundaries and minibatches preserve whole rollout sequences.

## Continual Backprop

The generate-and-test rule is adapted from Dohare et al.'s official
[`loss-of-plasticity`](https://github.com/shibhansh/loss-of-plasticity)
implementation at commit
`a6b79580d85f3025bdb601566d3627c5f489f13b`. That implementation provides
feed-forward `GnT`, convolutional `ConvGnT`, PPO integration, and an Adam
variant with element-wise step counters. It does not provide a recurrent or
GRU implementation, so this repository extends the same feature-replacement
invariant to the recurrent state instead of leaving the largest shared feature
layer uncontrolled.

After every PPO optimizer minibatch, CBP updates exponential-moving-average
contribution utility and replaces the lowest-utility mature features. The
fixed settings are replacement rate `1e-4`, utility decay `0.99`, and maturity
threshold `100` optimizer minibatches. The threshold is the official
`ConvGnT` default and is low enough that CBP is active in the reduced pilot;
replacement counts are recorded per layer.

For a convolutional or linear feature, replacement reinitializes incoming
weights at their original initialization norm, zeros the incoming bias and all
outgoing weights, and clears the corresponding Adam first moment, second
moment, and element-wise bias-correction age. Convolution-to-linear utility is
computed at every spatial position before reducing to a channel utility, as in
the official convolutional implementation. For pre-GRU linear features, the
outgoing GRU bias is adjusted by the feature's bias-corrected mean activation
before its outgoing weights are cleared.

A GRU hidden feature is one recurrent block: the corresponding row in each of
the reset, update, and new gates is reinitialized for both input and recurrent
weights; the corresponding recurrent column and policy/value-head columns are
zeroed; affected biases and Adam state are reset; and the matching live hidden
state is cleared. Bias compensation is applied to unaffected recurrent gates
and the two heads before the outgoing columns are cleared. Because recurrent
rows and columns intersect, outgoing columns are zeroed after row generation,
including the new feature's self-connection. This recurrent extension is a
documented implementation choice, not code released or validated by the
Banyan or CBP authors.

CBP is a direct control for conventional loss of plasticity: if the plateau
persists while replacements are active, that explanation becomes less likely.
It does not by itself prove that any remaining plateau is caused by gradient
interference; the gradient analysis remains correlational.

## Phase measurements

Evaluation occurs before any update in each phase, every 2,097,152 environment
steps, and at phase end. `S_start(di)` and `S_end(di)` are therefore literal
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
and 32GB RAM. Every scheduler allocation is limited to one hour. Three
minutes before that limit, the run checkpoints the policy, optimizer, CBP,
environment, recurrent state, and random generators, then requeues only that
unfinished array task. The next allocation restores the same trajectory; it is
not a new seed or policy. This keeps the previous 1.5x runtime safety allowance
and permits further continuation if the estimate is low, while presenting
short requests to the scheduler. The 25 primary conditions are a SLURM array,
so elapsed time is governed by the slowest run rather than the sum of all seeds
when capacity is available. The smoke array runs one full-budget phase for each
seed-0 condition. Those five checkpoints are the corresponding main runs, so
successful smoke computation is reused rather than discarded.
