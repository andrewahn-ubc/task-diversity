from .environment import Banyan


def make(env_id: str, **env_kwargs):
    """A JAX-version of OpenAI's env.make(env_name), built off Gymnax"""
    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not in registered banyan_grid environments.")

    if env_id == "Banyan":
        env = Banyan(**env_kwargs)

    return env


registered_envs = [
    "Banyan",
]
