# Group Project Plan: Reinforcement Learning for No-Limit Texas Hold'em

## Objective
Train autonomous agents to play **No-Limit Texas Hold'em (NLHE)** using **Reinforcement Learning** in a multi-agent, self-play setting. The environment supports both single-hand (tournament-style) and **cash game** modes, with stacks persisting across hands and elimination until one player remains or a maximum number of hands is reached.

## Group 
- IBRAHIM HOUMED Aléo
- ADLY Maxence

## Technical Stack
- **RL framework:** Ray RLlib (PPO)
- **Environment API:** RLlib `MultiAgentEnv`, turn-based
- **Model:** Custom PyTorch Actor-Critic with **action masking** (illegal actions disabled via mask)
- **Game logic:** Custom implementation (deck, blinds, streets, side-pots, showdown; 2–10 players)
- **Hand evaluation:** Treys-based evaluator for hand ranking

## Environment Design
- **Observation (per player):** Fixed-size vector in [0,1]: hole cards (52), board (52), street (4), position, active mask, pot, stacks (normalised + relative to chip leader), current bets, amount to call, min/max raise, betting history per street. All monetary quantities normalised for scale-invariance and stability.
- **Actions:** Fold, Check/Call, Raise Min, Raise ½ Pot, Raise Pot, All-In. Mask ensures only legal actions are chosen.
- **Rewards:** Sparse, zero-sum; chip delta vs. start of hand at hand end.
- **Modes:** (1) **Single-hand:** one episode = one hand, stacks reset each episode. (2) **Cash game:** one episode = one session; stacks persist, eliminated players (stack=0) sit out; episode ends when ≤1 player remains or max hands reached.

## Training Setup
- **Algorithm:** PPO (Proximal Policy Optimization)
- **Self-play:** All agents share one policy (`shared_policy`)
- **CLI:** `train.py` with options for players, stack, blinds, `--cash_game`, `--max_hands_per_episode`, hidden size, workers, save interval, resume checkpoint
- **Checkpointing:** Periodic saves to a configurable directory for evaluation and GUI

## Evaluation & Demo
- **GUI (Tkinter):** Load checkpoint, run hands (single or continuous), display table, hole/board cards (suit images), pot, stacks, action log, action probabilities and value estimate; optional cash game mode with persistent stacks.
- **CLI (`play.py`):** Run N hands or one cash game session with a loaded policy; optional deterministic mode and random opponents on selected seats.

## Success Criteria
- Agents learn to play legally (action masking) and improve in terms of episode/ session return.
- Cash game agents experience variable stack sizes and elimination; observation includes relative stack (vs. chip leader) for context.
- Reproducible training and evaluation via CLI and GUI.
