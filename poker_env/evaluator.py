"""Hand evaluation wrapper around the treys library.

Converts between our internal card representation (0-51 integers) and
the treys bit-packed format, then exposes helpers used at showdown.
"""

from __future__ import annotations

from treys import Card, Evaluator as _TreysEvaluator

RANKS = "23456789TJQKA"
SUITS = "shdc"

_treys_evaluator = _TreysEvaluator()


def _idx_to_treys(card_idx: int) -> int:
    """Convert a 0-51 card index to a treys bit-packed integer.

    Layout: index = rank * 4 + suit
      rank 0 = '2', rank 12 = 'A'
      suit 0 = 's', 1 = 'h', 2 = 'd', 3 = 'c'
    """
    rank = card_idx // 4
    suit = card_idx % 4
    return Card.new(RANKS[rank] + SUITS[suit])


def _idxs_to_treys(card_idxs: list[int]) -> list[int]:
    return [_idx_to_treys(c) for c in card_idxs]


def evaluate(hole: list[int], board: list[int]) -> int:
    """Return a hand-strength score (lower is better, 1 = Royal Flush).

    ``hole`` has exactly 2 cards; ``board`` has 3-5 cards.
    """
    return _treys_evaluator.evaluate(_idxs_to_treys(board), _idxs_to_treys(hole))


def determine_winners(
    hands: dict[int, list[int]],
    board: list[int],
) -> list[int]:
    """Return the list of player indices that share the best hand.

    ``hands`` maps player_idx -> [card_a, card_b].
    """
    scores: dict[int, int] = {}
    for pid, hole in hands.items():
        scores[pid] = evaluate(hole, board)

    best = min(scores.values())
    return [pid for pid, s in scores.items() if s == best]


def hand_rank_string(hole: list[int], board: list[int]) -> str:
    """Human-readable rank class (e.g. 'Full House')."""
    score = evaluate(hole, board)
    return _treys_evaluator.class_to_string(_treys_evaluator.get_rank_class(score))


def card_to_str(card_idx: int) -> str:
    """Pretty-print a card index as e.g. 'As', 'Kh'."""
    rank = card_idx // 4
    suit = card_idx % 4
    return RANKS[rank] + SUITS[suit]
