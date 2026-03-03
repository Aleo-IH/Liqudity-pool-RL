"""poker_env -- Multi-agent No-Limit Texas Hold'em (RLlib MultiAgentEnv).

Usage::

    from poker_env.env import PokerEnv

    env = PokerEnv({"num_players": 6})
    obs, info = env.reset(seed=42)
    # obs = {"player_3": {"observation": ..., "action_mask": ...}}
    action_dict = {"player_3": env.action_space.sample()}
    obs, rewards, terminateds, truncateds, infos = env.step(action_dict)
"""

from .env import PokerEnv

__all__ = ["PokerEnv"]
