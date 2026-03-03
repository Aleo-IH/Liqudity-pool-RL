"""Build the flat observation vector and action mask for a given player.

The observation encodes the player's *information set* -- everything a
rational agent at the table could know -- following conventions used in
state-of-the-art poker AI research (Pluribus, Deep CFR).
"""

from __future__ import annotations

import numpy as np

from .game import NUM_ACTIONS, PokerGame, Street

class ObservationBuilder:
    """Constructs fixed-size observation vectors for any seat at the table."""

    def __init__(self, num_players: int, max_history_per_street: int = 10):
        self.num_players = num_players
        self.max_history = max_history_per_street
        self.obs_dim = self._compute_obs_dim()

    def _compute_obs_dim(self) -> int:
        n = self.num_players
        h = self.max_history
        return (
            52          # hole cards  (one-hot)
            + 52        # board cards (multi-hot)
            + 4         # street      (one-hot)
            + n         # relative position to dealer (one-hot)
            + n         # active players mask
            + 1         # pot size (normalised)
            + n         # stacks   (normalised by initial_stack)
            + n         # stacks relative to max (stack[i]/max_stack)
            + n         # current street bets (normalised)
            + 1         # amount to call (normalised)
            + 2         # min raise, max raise (normalised)
            + 4 * h * (n + NUM_ACTIONS)  # betting history
        )

    def build(self, game: PokerGame, player: int) -> np.ndarray:
        """Return a float32 vector of shape (obs_dim,) in [0, 1]. Clipped to avoid NaN/Inf."""
        parts: list[np.ndarray] = []
        total_chips_f = float(max(sum(game.stacks), 1))

        # --- Hole cards (one-hot 52) ---
        hole = np.zeros(52, dtype=np.float32)
        for c in game.hole_cards[player]:
            hole[c] = 1.0
        parts.append(hole)

        # --- Board cards (multi-hot 52) ---
        board = np.zeros(52, dtype=np.float32)
        for c in game.board:
            board[c] = 1.0
        parts.append(board)

        # --- Street (one-hot 4) ---
        street = np.zeros(4, dtype=np.float32)
        st = min(int(game.street), 3)
        street[st] = 1.0
        parts.append(street)

        # --- Relative position to dealer (one-hot N) ---
        pos = np.zeros(self.num_players, dtype=np.float32)
        relative = (player - game.dealer_idx) % self.num_players
        pos[relative] = 1.0
        parts.append(pos)

        # --- Active players mask (binary N) ---
        active = np.array(
            [0.0 if game.folded[i] else 1.0 for i in range(self.num_players)],
            dtype=np.float32,
        )
        parts.append(active)

        # --- Pot size (normalised by total chips, in [0,1]) ---
        parts.append(np.array([min(1.0, game.pot / total_chips_f)], dtype=np.float32))

        # --- Stacks (normalised by total chips so sum <= 1, each in [0,1]) ---
        stacks = np.array(
            [game.stacks[i] / total_chips_f for i in range(self.num_players)],
            dtype=np.float32,
        )
        parts.append(stacks)

        # --- Stacks relative to max among eligible players ---
        eliminated = getattr(game, "eliminated", [False] * self.num_players)
        max_stack = max(
            (game.stacks[i] for i in range(self.num_players) if not eliminated[i]),
            default=1,
        )
        max_stack = max(max_stack, 1)
        stacks_rel_max = np.array(
            [
                (game.stacks[i] / max_stack) if not eliminated[i] else 0.0
                for i in range(self.num_players)
            ],
            dtype=np.float32,
        )
        parts.append(stacks_rel_max)

        # --- Current street bets (normalised by total chips to stay in [0,1]) ---
        denom_bet = max(game.pot, total_chips_f, 1.0)
        bets = np.array(
            [game.bets[i] / denom_bet for i in range(self.num_players)],
            dtype=np.float32,
        )
        parts.append(bets)

        # --- Amount to call (normalised) ---
        current_bet = max(game.bets)
        to_call = current_bet - game.bets[player]
        parts.append(np.array([min(1.0, to_call / total_chips_f)], dtype=np.float32))

        # --- Min raise / max raise (normalised) ---
        min_raise_total = current_bet + game._min_raise
        min_raise_cost = max(min_raise_total - game.bets[player], 0)
        max_raise_cost = game.stacks[player]
        parts.append(
            np.array(
                [
                    min(1.0, min_raise_cost / total_chips_f),
                    min(1.0, max_raise_cost / total_chips_f),
                ],
                dtype=np.float32,
            )
        )

        # --- Betting history (4 streets x H x (N + NUM_ACTIONS)) ---
        history = np.zeros(
            (4, self.max_history, self.num_players + NUM_ACTIONS),
            dtype=np.float32,
        )
        counts = [0, 0, 0, 0]
        for st_val, pid, act, _amount in game.action_history:
            if st_val > 3:
                continue
            slot = counts[st_val]
            if slot >= self.max_history:
                continue
            history[st_val, slot, pid] = 1.0
            history[st_val, slot, self.num_players + act] = 1.0
            counts[st_val] += 1
        parts.append(history.flatten())

        obs = np.concatenate(parts)
        np.clip(obs, 0.0, 1.0, out=obs)
        np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0, copy=False)
        return obs

    def build_action_mask(self, game: PokerGame) -> np.ndarray:
        legal = game.get_legal_actions()
        return np.array(legal, dtype=np.int8)
