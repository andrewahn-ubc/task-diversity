# Third-party notices

The Continual Backprop generate-and-test rule and element-wise Adam-state
reset in `src/banyan_pilot/continual_backprop.py` and
`src/banyan_pilot/continual_adam.py` are adapted from:

- Shibhansh Dohare, [`loss-of-plasticity`](https://github.com/shibhansh/loss-of-plasticity),
  commit `a6b79580d85f3025bdb601566d3627c5f489f13b`.
- Relevant upstream files: `lop/algos/gnt.py`, `lop/algos/convGnT.py`,
  `lop/algos/rl/ppo.py`, and `lop/utils/AdamGnT.py`.

The local implementation adds support for this repository's convolutional GRU
actor-critic, preserves its orthogonal initialization scale, and includes
checkpointable CBP state. Those extensions are not part of the upstream code.

Upstream license:

> MIT License
>
> Copyright (c) 2024 Shibhansh Dohare
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.
