#!/usr/bin/env python3
"""Visualise poker games using a trained RLlib model.

Loads a saved RLlib checkpoint and plays test hands, rendering each
decision with the agent's action probabilities.

Usage
-----
    python play.py checkpoints_rllib
    python play.py checkpoints_rllib --hands 20
    python play.py checkpoints_rllib --deterministic
    python play.py checkpoints_rllib --vs_random 1
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import ray
import torch

from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog

from poker_env.env import PokerEnv
from poker_env.game import NUM_ACTIONS, Action
from poker_env.model import ActionMaskModel

ACTION_NAMES = {
    Action.FOLD: "Fold",
    Action.CHECK_CALL: "Check/Call",
    Action.RAISE_MIN: "Raise Min",
    Action.RAISE_HALF_POT: "Raise 1/2 Pot",
    Action.RAISE_POT: "Raise Pot",
    Action.ALL_IN: "All-In",
}


def print_action_probs(policy, obs: dict, chosen_action: int) -> None:
    """Display the policy's probability over each legal action."""
    with torch.no_grad():
        obs_t = {
            k: torch.from_numpy(v).unsqueeze(0).float() for k, v in obs.items()
        }
        logits, _ = policy.model({"obs": obs_t}, [], None)
        probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
        value = policy.model.value_function().item()

    mask = obs["action_mask"]
    parts = []
    for i in range(NUM_ACTIONS):
        if mask[i]:
            marker = " <--" if i == chosen_action else ""
            parts.append(f"    {ACTION_NAMES[Action(i)]:14s}  {probs[i]:5.1%}{marker}")
    print(f"  V={value:+.1f}  Action probabilities:")
    print("\n".join(parts))


def play_hands(
    policy,
    env_config: dict,
    num_hands: int,
    deterministic: bool = False,
    random_seats: set[int] | None = None,
    cash_game: bool = False,
) -> None:
    """Play and render hands using the trained policy."""
    env = PokerEnv(env_config)
    rng = np.random.default_rng(42)
    num_players = env_config["num_players"]
    total_rewards = {f"player_{i}": 0.0 for i in range(num_players)}
    hands_done = 0

    if cash_game:
        obs_dict, _ = env.reset(seed=42)
        hand_idx = 0
        while obs_dict:
            hand_idx += 1
            print(f"\n{'#' * 60}")
            print(f"  HAND {hand_idx} (cash game session)")
            print(f"{'#' * 60}")
            while obs_dict:
                agent = next(iter(obs_dict))
                obs = obs_dict[agent]
                pid = int(agent.split("_")[1])
                mask = obs["action_mask"]

                env.render()

                if random_seats and pid in random_seats:
                    legal = [i for i, ok in enumerate(mask) if ok]
                    action = int(rng.choice(legal))
                    print(f"  {agent} (random): {ACTION_NAMES[Action(action)]}")
                else:
                    action, _, _ = policy.compute_single_action(
                        obs, explore=not deterministic
                    )
                    action = int(action)
                    print(f"  {agent}:")
                    print_action_probs(policy, obs, action)

                obs_dict, rewards, terminateds, truncateds, infos = env.step(
                    {agent: action}
                )
                if rewards:
                    for agent in rewards:
                        total_rewards[agent] += rewards[agent]
                    hands_done += 1
                    print(f"\n  --- Hand {hand_idx} Results ---")
                    for a in sorted(rewards):
                        print(f"    {a}: {rewards[a]:+.0f}")
                if terminateds.get("__all__", False):
                    print("\n  Session over.")
                    break
            if terminateds.get("__all__", False):
                break
    else:
        for hand_idx in range(1, num_hands + 1):
            print(f"\n{'#' * 60}")
            print(f"  HAND {hand_idx}/{num_hands}")
            print(f"{'#' * 60}")

            obs_dict, _ = env.reset(seed=hand_idx)
            hand_rewards = {}

            while obs_dict:
                agent = next(iter(obs_dict))
                obs = obs_dict[agent]
                pid = int(agent.split("_")[1])
                mask = obs["action_mask"]

                env.render()

                if random_seats and pid in random_seats:
                    legal = [i for i, ok in enumerate(mask) if ok]
                    action = int(rng.choice(legal))
                    print(f"  {agent} (random): {ACTION_NAMES[Action(action)]}")
                else:
                    action, _, _ = policy.compute_single_action(
                        obs, explore=not deterministic
                    )
                    action = int(action)
                    print(f"  {agent}:")
                    print_action_probs(policy, obs, action)

                obs_dict, rewards, terminateds, truncateds, infos = env.step(
                    {agent: action}
                )
                hand_rewards.update(rewards)

                if terminateds.get("__all__", False):
                    break

            env.render()
            print(f"\n  --- Hand {hand_idx} Results ---")
            for agent in sorted(hand_rewards):
                r = hand_rewards[agent]
                total_rewards[agent] += r
                print(f"    {agent}: {r:+.0f}")
            hands_done += 1

    print(f"\n{'=' * 60}")
    print(f"  CUMULATIVE RESULTS over {hands_done} hands")
    print(f"{'=' * 60}")
    for agent in sorted(total_rewards):
        avg = total_rewards[agent] / max(hands_done, 1)
        print(f"  {agent}: total={total_rewards[agent]:+.0f}  avg={avg:+.1f}/hand")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise poker games with a trained RLlib model"
    )
    parser.add_argument("checkpoint", type=str, help="Path to RLlib checkpoint dir")
    parser.add_argument("--hands", type=int, default=5)
    parser.add_argument("--num_players", type=int, default=None)
    parser.add_argument("--initial_stack", type=int, default=1000)
    parser.add_argument("--small_blind", type=int, default=5)
    parser.add_argument("--big_blind", type=int, default=10)
    parser.add_argument("--deterministic", action="store_true", help="Greedy actions")
    parser.add_argument(
        "--cash_game",
        action="store_true",
        help="Run one cash game session (stacks persist until elimination or max hands)",
    )
    parser.add_argument(
        "--vs_random",
        type=int,
        nargs="*",
        default=None,
        help="Seat indices that play random (e.g. --vs_random 1)",
    )
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True)
    ModelCatalog.register_custom_model("action_mask_model", ActionMaskModel)

    algo = PPO.from_checkpoint(os.path.abspath(args.checkpoint))
    print(f"Loaded checkpoint: {args.checkpoint}")

    train_cfg = algo.config.env_config
    env_config = {
        "num_players": args.num_players or train_cfg.get("num_players", 2),
        "initial_stack": args.initial_stack,
        "small_blind": args.small_blind,
        "big_blind": args.big_blind,
    }
    if args.cash_game:
        env_config["cash_game"] = True
        env_config["max_hands_per_episode"] = args.hands

    policy = algo.get_policy("shared_policy")
    random_seats = set(args.vs_random) if args.vs_random else None

    play_hands(
        policy,
        env_config,
        args.hands,
        args.deterministic,
        random_seats,
        cash_game=args.cash_game,
    )

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
