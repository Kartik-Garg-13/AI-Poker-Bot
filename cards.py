from collections import Counter
from itertools import combinations

ranks = "23456789TJQKA"
suits = "CDHS"
CATEGORY_NAMES = ["high card", "one pair", "two pair", "trips", "straight",
                  "flush", "full house", "quads", "straight flush"]


def make_deck():
    return list(range(52))


def rank_of(card):
    return card // 4


def suit_of(card):
    return card % 4


def card_str(card):
    return ranks[rank_of(card)] + suits[suit_of(card)]


def is_flush(cards):
    return len({suit_of(c) for c in cards}) == 1


def straight_high(cards):
    s = {rank_of(c) for c in cards}
    if len(s) != 5:
        return None
    if s == {12, 0, 1, 2, 3}:
        return 3
    if max(s) - min(s) == 4:
        return max(s)
    return None


def shape(cards):
    return sorted(((count, rank) for rank, count in Counter(map(rank_of, cards)).items()), reverse=True)


def evaluate_5(cards):
    pairs = shape(cards)
    counts = [p[0] for p in pairs]
    r = [p[1] for p in pairs]
    flush = is_flush(cards)
    sh = straight_high(cards)
    if flush and sh is not None:
        return (8 << 20) | (sh << 16)
    if flush:
        return (5 << 20) | (r[0] << 16) | (r[1] << 12) | (r[2] << 8) | (r[3] << 4) | r[4]
    if sh is not None:
        return (4 << 20) | (sh << 16)
    if counts == [3, 2]:
        return (6 << 20) | (r[0] << 16) | (r[1] << 12)
    if counts == [4, 1]:
        return (7 << 20) | (r[0] << 16) | (r[1] << 12)
    if counts == [3, 1, 1]:
        return (3 << 20) | (r[0] << 16) | (r[1] << 12) | (r[2] << 8)
    if counts == [2, 2, 1]:
        return (2 << 20) | (r[0] << 16) | (r[1] << 12) | (r[2] << 8)
    if counts == [2, 1, 1, 1]:
        return (1 << 20) | (r[0] << 16) | (r[1] << 12) | (r[2] << 8) | (r[3] << 4)
    return (r[0] << 16) | (r[1] << 12) | (r[2] << 8) | (r[3] << 4) | r[4]


def evaluate_7(cards):
    return max(evaluate_5(hand) for hand in combinations(cards, 5))


def _straight_in_mask(mask):
    for high in range(12, 3, -1):
        if (mask >> (high - 4)) & 0x1F == 0x1F:
            return high
    if mask & 0x100F == 0x100F:
        return 3
    return None


def _pack(category, rs):
    v = category << 20
    for i, r in enumerate(rs):
        v |= r << (16 - 4 * i)
    return v


def evaluate_fast(cards):
    counts = [0] * 13
    suit_masks = [0, 0, 0, 0]
    for c in cards:
        r = c >> 2
        counts[r] += 1
        suit_masks[c & 3] |= 1 << r
    for m in suit_masks:
        if m.bit_count() >= 5:
            sh = _straight_in_mask(m)
            if sh is not None:
                return (8 << 20) | (sh << 16)
            return _pack(5, [r for r in range(12, -1, -1) if m >> r & 1][:5])
    groups = sorted(((n, r) for r, n in enumerate(counts) if n), reverse=True)
    top = groups[0][0]
    if top == 4:
        return _pack(7, [groups[0][1], max(r for n, r in groups[1:])])
    if top == 3 and groups[1][0] >= 2:
        return _pack(6, [groups[0][1], groups[1][1]])
    sh = _straight_in_mask(suit_masks[0] | suit_masks[1] | suit_masks[2] | suit_masks[3])
    if sh is not None:
        return (4 << 20) | (sh << 16)
    if top == 3:
        return _pack(3, [groups[0][1], groups[1][1], groups[2][1]])
    if top == 2 and groups[1][0] == 2:
        return _pack(2, [groups[0][1], groups[1][1], max(r for n, r in groups[2:])])
    if top == 2:
        return _pack(1, [groups[0][1]] + [r for n, r in groups[1:4]])
    return _pack(0, [r for n, r in groups[:5]])


def category_name(score):
    return CATEGORY_NAMES[score >> 20]
