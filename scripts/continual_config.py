"""Shared experiment defaults and the deliberately small pilot search space."""

DEFAULTS = {
    "num_envs": 128, "rollout_steps": 128, "minibatches": 4,
    "update_epochs": 4, "width": 128, "learning_rate": 3e-4,
    "entropy_coef": .01, "value_coef": .5, "clip_eps": .2,
    "gamma": .99, "gae_lambda": .95, "max_grad_norm": .5,
    "cbp_rate": 2e-4, "cbp_decay": .99, "cbp_maturity": 100,
    "eval_episodes_per_depth": 64, "eval_every_steps": 5_000_000,
    "checkpoint_every_steps": 1_000_000, "max_steps": 100,
}
VARIANTS = {
    "base": {},
    "lower_lr": {"learning_rate": 1e-4},
    "more_entropy": {"entropy_coef": .03},
    "more_cbp": {"cbp_rate": 1e-3},
}
N_VALUES = (1, 4, 16, 64, 256)
