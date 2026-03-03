"""Core Texas Hold'em No-Limit game logic.

Manages deck, dealing, blinds, betting rounds, street progression,
side-pot computation, and showdown resolution.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np

from . import evaluator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class Street(enum.IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    SHOWDOWN = 4

class Action(enum.IntEnum):
    FOLD = 0
    CHECK_CALL = 1
    RAISE_MIN = 2
    RAISE_HALF_POT = 3
    RAISE_POT = 4
    ALL_IN = 5

NUM_ACTIONS = len(Action)


# ---------------------------------------------------------------------------
# Side-pot helper
# ---------------------------------------------------------------------------

@dataclass
class SidePot:
    amount: int = 0
    eligible: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PokerGame
# ---------------------------------------------------------------------------

class PokerGame:
    """Full state machine for one hand of NLHE."""

    def __init__(
        self,
        num_players: int = 6,
        initial_stack: int = 1000,
        small_blind: int = 5,
        big_blind: int = 10,
        rng: np.random.Generator | None = None,
        cash_game: bool = False,
    ):
        assert 2 <= num_players <= 10
        self.num_players = num_players
        self.initial_stack = initial_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.rng = rng or np.random.default_rng()
        self.cash_game = cash_game

        # Persistent across hands
        self.dealer_idx: int = 0
        self.eliminated: list[bool] = [False] * num_players

        # Per-hand state (initialised in new_hand)
        self.stacks: list[int] = []
        self.hole_cards: list[list[int]] = []
        self.board: list[int] = []
        self.street: Street = Street.PREFLOP
        self.pot: int = 0
        self.bets: list[int] = []  # amount bet *this street*
        self.folded: list[bool] = []
        self.all_in: list[bool] = []
        self.action_history: list[tuple[int, int, int, int]] = []
        self.current_player: int = 0
        self._last_raiser: int = -1
        self._num_acted_this_street: int = 0
        self._min_raise: int = 0
        self._hand_over: bool = True
        self._deck: list[int] = []
        self._initial_stacks: list[int] = []
        self._total_bets: list[int] = []  # cumulative bets over entire hand

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def new_hand(self) -> None:
        """Start a fresh hand: shuffle, deal hole cards, post blinds."""
        self._deck = list(range(52))
        self.rng.shuffle(self._deck)

        if self.cash_game:
            if not self.stacks:
                self.stacks = [self.initial_stack] * self.num_players
            # else: keep current stacks, do not reset eliminated
        else:
            self.stacks = [self.initial_stack] * self.num_players

        self._initial_stacks = list(self.stacks)
        self.hole_cards = [[] for _ in range(self.num_players)]
        self.board = []
        self.street = Street.PREFLOP
        self.pot = 0
        self.bets = [0] * self.num_players
        self._total_bets = [0] * self.num_players
        self.all_in = [False] * self.num_players
        self.action_history = []
        self._hand_over = False

        if self.cash_game:
            self.folded = [self.eliminated[i] for i in range(self.num_players)]
            for i in range(self.num_players):
                if self.eliminated[i]:
                    self.hole_cards[i] = []
                else:
                    self.hole_cards[i] = [self._deck.pop(), self._deck.pop()]
            if self.count_eligible() < 2:
                self._hand_over = True
                return
            # dealer_idx already advanced in _finish_hand
            sb_idx = self._next_eligible(self.dealer_idx)
            bb_idx = self._next_eligible(sb_idx)
        else:
            self.folded = [False] * self.num_players
            for i in range(self.num_players):
                self.hole_cards[i] = [self._deck.pop(), self._deck.pop()]
            sb_idx = self._next_active(self.dealer_idx)
            bb_idx = self._next_active(sb_idx)

        self._post_blind(sb_idx, self.small_blind)
        self._post_blind(bb_idx, self.big_blind)

        self._min_raise = self.big_blind
        self._last_raiser = bb_idx
        self._num_acted_this_street = 0

        if self.cash_game:
            self.current_player = self._next_eligible(bb_idx)
            if self.count_eligible() == 2:
                self.current_player = sb_idx
        else:
            self.current_player = self._next_active(bb_idx)
            if self.num_players == 2:
                self.current_player = sb_idx

        if self._count_can_act() <= 1:
            self._run_out_board()

    def get_legal_actions(self) -> list[bool]:
        """Return a boolean mask of length NUM_ACTIONS for the current player."""
        mask = [False] * NUM_ACTIONS
        if self._hand_over:
            return mask

        p = self.current_player
        current_bet = max(self.bets)
        to_call = current_bet - self.bets[p]
        stack = self.stacks[p]

        # Fold is always legal (unless no bet to face -- still allow it)
        mask[Action.FOLD] = True

        # Check / Call
        if to_call == 0:
            mask[Action.CHECK_CALL] = True  # check
        elif stack >= to_call:
            mask[Action.CHECK_CALL] = True  # call
        # If stack < to_call, calling would be all-in (handled by ALL_IN)

        # Raises: only if player has enough chips beyond calling
        if stack > to_call:
            min_total_raise = current_bet + self._min_raise
            raise_min_amount = min_total_raise - self.bets[p]

            if stack >= raise_min_amount:
                mask[Action.RAISE_MIN] = True

                # Half pot / pot raises need enough chips
                half_pot = self._raise_amount(Action.RAISE_HALF_POT)
                if half_pot is not None and stack >= (half_pot - self.bets[p]):
                    mask[Action.RAISE_HALF_POT] = True

                pot_raise = self._raise_amount(Action.RAISE_POT)
                if pot_raise is not None and stack >= (pot_raise - self.bets[p]):
                    mask[Action.RAISE_POT] = True

        # All-in is always legal if player has chips
        if stack > 0:
            mask[Action.ALL_IN] = True

        return mask

    def apply_action(self, action: int) -> None:
        """Execute *action* for current_player and advance the game."""
        assert not self._hand_over, "Hand is already over"
        p = self.current_player
        current_bet = max(self.bets)
        to_call = current_bet - self.bets[p]

        if action == Action.FOLD:
            self.folded[p] = True
            self.action_history.append((int(self.street), p, action, 0))

        elif action == Action.CHECK_CALL:
            amount = min(to_call, self.stacks[p])
            self._place_bet(p, amount)
            self.action_history.append((int(self.street), p, action, amount))

        elif action in (Action.RAISE_MIN, Action.RAISE_HALF_POT, Action.RAISE_POT):
            total_raise = self._raise_amount(action)
            if total_raise is None:
                total_raise = current_bet + self._min_raise
            amount = total_raise - self.bets[p]
            amount = min(amount, self.stacks[p])
            actual_raise_size = (self.bets[p] + amount) - current_bet
            if actual_raise_size >= self._min_raise:
                self._min_raise = actual_raise_size
            self._last_raiser = p
            self._num_acted_this_street = 0
            self._place_bet(p, amount)
            self.action_history.append((int(self.street), p, action, amount))

        elif action == Action.ALL_IN:
            amount = self.stacks[p]
            if amount + self.bets[p] > current_bet:
                actual_raise_size = (self.bets[p] + amount) - current_bet
                if actual_raise_size >= self._min_raise:
                    self._min_raise = actual_raise_size
                self._last_raiser = p
                self._num_acted_this_street = 0
            self._place_bet(p, amount)
            self.all_in[p] = True
            self.action_history.append((int(self.street), p, action, amount))

        else:
            raise ValueError(f"Unknown action {action}")

        self._num_acted_this_street += 1

        # Check if hand ends because only one player remains
        active_not_folded = [
            i for i in range(self.num_players) if not self.folded[i]
        ]
        if len(active_not_folded) == 1:
            self._finish_hand()
            return

        # Check if street betting is complete
        if self._is_street_complete():
            if self.street == Street.RIVER or self._count_can_act() == 0:
                if self.street < Street.RIVER:
                    self._run_out_board()
                else:
                    self._finish_hand()
            else:
                self._advance_street()
        else:
            self.current_player = self._next_can_act(p)

    def is_hand_over(self) -> bool:
        return self._hand_over

    def get_rewards(self) -> dict[int, float]:
        """Return per-player reward = final_stack - initial_stack (zero-sum)."""
        return {
            i: float(self.stacks[i] - self._initial_stacks[i])
            for i in range(self.num_players)
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _post_blind(self, player: int, amount: int) -> None:
        actual = min(amount, self.stacks[player])
        self._place_bet(player, actual)
        if self.stacks[player] == 0:
            self.all_in[player] = True

    def _place_bet(self, player: int, amount: int) -> None:
        self.stacks[player] -= amount
        self.bets[player] += amount
        self._total_bets[player] += amount
        self.pot += amount
        if self.stacks[player] == 0 and not self.folded[player]:
            self.all_in[player] = True

    def _raise_amount(self, action: Action) -> int | None:
        """Compute the *total bet level* for a given raise action."""
        current_bet = max(self.bets)
        pot_after_call = self.pot + (current_bet - self.bets[self.current_player])

        if action == Action.RAISE_MIN:
            return current_bet + self._min_raise
        elif action == Action.RAISE_HALF_POT:
            size = pot_after_call // 2
            total = current_bet + max(size, self._min_raise)
            return total
        elif action == Action.RAISE_POT:
            total = current_bet + max(pot_after_call, self._min_raise)
            return total
        return None

    def _next_eligible(self, idx: int) -> int:
        """Next seat that is not eliminated (for dealer/SB/BB in cash game)."""
        for offset in range(1, self.num_players + 1):
            nxt = (idx + offset) % self.num_players
            if not self.eliminated[nxt]:
                return nxt
        return idx

    def count_eligible(self) -> int:
        """Number of players not eliminated."""
        return sum(1 for i in range(self.num_players) if not self.eliminated[i])

    def _next_active(self, idx: int) -> int:
        """Next seat that has not folded (wraps around)."""
        for offset in range(1, self.num_players + 1):
            nxt = (idx + offset) % self.num_players
            if not self.folded[nxt]:
                return nxt
        return idx

    def _next_can_act(self, idx: int) -> int:
        """Next seat that can still bet (not folded, not all-in)."""
        for offset in range(1, self.num_players + 1):
            nxt = (idx + offset) % self.num_players
            if not self.folded[nxt] and not self.all_in[nxt]:
                return nxt
        return idx

    def _count_can_act(self) -> int:
        return sum(
            1 for i in range(self.num_players)
            if not self.folded[i] and not self.all_in[i]
        )

    def _is_street_complete(self) -> bool:
        """True when all players who can act have acted and bets are equalised."""
        active = [
            i for i in range(self.num_players)
            if not self.folded[i] and not self.all_in[i]
        ]
        if not active:
            return True

        current_bet = max(self.bets)
        bets_equal = all(self.bets[i] == current_bet for i in active)
        all_acted = self._num_acted_this_street >= len(active)
        return bets_equal and all_acted

    def _advance_street(self) -> None:
        """Move to the next street: collect bets, deal community cards."""
        self.bets = [0] * self.num_players
        self.street = Street(int(self.street) + 1)
        self._last_raiser = -1
        self._num_acted_this_street = 0
        self._min_raise = self.big_blind

        if self.street == Street.FLOP:
            self._deck.pop()  # burn
            self.board.extend([self._deck.pop() for _ in range(3)])
        elif self.street == Street.TURN:
            self._deck.pop()
            self.board.append(self._deck.pop())
        elif self.street == Street.RIVER:
            self._deck.pop()
            self.board.append(self._deck.pop())

        # First to act post-flop: left of dealer
        self.current_player = self._next_can_act(self.dealer_idx)

        # If <= 1 player can act, run out the remaining board
        if self._count_can_act() <= 1:
            if self.street < Street.RIVER:
                self._run_out_board()
            else:
                self._finish_hand()

    def _run_out_board(self) -> None:
        """Deal remaining community cards when no more betting is possible."""
        while len(self.board) < 5:
            if self.board or self.street >= Street.FLOP:
                self._deck.pop()  # burn
            else:
                self._deck.pop()
            self.board.append(self._deck.pop())
        self._finish_hand()

    def _finish_hand(self) -> None:
        """Resolve showdown and distribute pot(s), then mark hand as over."""
        self.street = Street.SHOWDOWN
        active = [i for i in range(self.num_players) if not self.folded[i]]

        if len(active) == 1:
            self.stacks[active[0]] += self.pot
        else:
            self._resolve_showdown(active)

        self.pot = 0
        self._hand_over = True

        if self.cash_game:
            for i in range(self.num_players):
                self.eliminated[i] = self.stacks[i] == 0
            self.dealer_idx = self._next_eligible(self.dealer_idx)
        else:
            self.dealer_idx = (self.dealer_idx + 1) % self.num_players

    def _resolve_showdown(self, active: list[int]) -> None:
        """Distribute pot(s) among active players using side-pot logic."""
        side_pots = self._compute_side_pots(active)

        for sp in side_pots:
            eligible_hands = {
                p: self.hole_cards[p] for p in sp.eligible if not self.folded[p]
            }
            if not eligible_hands:
                continue
            winners = evaluator.determine_winners(eligible_hands, self.board)
            share = sp.amount // len(winners)
            remainder = sp.amount % len(winners)
            for w in winners:
                self.stacks[w] += share
            # Give remainder chip(s) to earliest position from dealer
            if remainder > 0:
                for offset in range(self.num_players):
                    candidate = (self.dealer_idx + 1 + offset) % self.num_players
                    if candidate in winners:
                        self.stacks[candidate] += remainder
                        break

    def _compute_side_pots(self, active: list[int]) -> list[SidePot]:
        """Compute main pot and side pots from total bets."""
        contributions = sorted(set(self._total_bets[i] for i in active))
        pots: list[SidePot] = []
        prev_level = 0

        for level in contributions:
            if level <= prev_level:
                continue
            sp = SidePot()
            marginal = level - prev_level
            for i in range(self.num_players):
                contrib = min(self._total_bets[i] - prev_level, marginal)
                if contrib > 0:
                    sp.amount += contrib
                    if i in active:
                        sp.eligible.append(i)
            pots.append(sp)
            prev_level = level

        return pots
