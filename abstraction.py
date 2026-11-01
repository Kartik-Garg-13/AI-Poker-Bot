import json
import random
from functools import lru_cache
from pathlib import Path

import numpy as np

from cards import evaluate_fast, rank_of, suit_of

FOLD, CALL, HALF, POT, ALLIN = range(5)
ACTION_NAMES = ["fold", "check/call", "bet 1/2 pot", "bet pot", "all-in"]
N_ACTIONS = len(ACTION_NAMES)
MAX_SIZED_RAISES = 2
N_BUCKETS = 8
FEATURE_NAMES = ["preflop", "flop", "turn", "river", "equity"] + \
                [f"bucket {i}" for i in range(N_BUCKETS)] + \
                ["pot size", "pot odds", "my stack", "opp stack", "in position",
                 "raises this street", "opp raises (hand)", "my raises (hand)"]
N_FEATURES = len(FEATURE_NAMES)

_nprng = np.random.default_rng(2024)
_AR = np.arange(13)


def evaluate_batch(c):
    r, s = c >> 2, c & 3
    n = len(c)
    rc = (r[:, :, None] == _AR).sum(1)
    sc = (s[:, :, None] == np.arange(4)).sum(1)
    flush = sc.max(1) >= 5
    fmask = np.where(s == sc.argmax(1)[:, None], 1 << r, 0).sum(1)
    rmask = ((rc > 0) << _AR).sum(1)

    def straight(mask):
        hi = np.full(n, -1)
        for h in range(12, 3, -1):
            hi = np.where((hi < 0) & ((mask >> (h - 4)) & 31 == 31), h, hi)
        return np.where((hi < 0) & (mask & 0x100F == 0x100F), 3, hi)

    def top(avail, k):
        rev = avail[:, ::-1]
        cs = np.cumsum(rev, 1)
        return [12 - np.argmax(rev & (cs == i), 1) for i in range(1, k + 1)]

    def pack(cat, rs):
        v = np.full(n, cat << 20)
        for i, x in enumerate(rs):
            v |= x << (16 - 4 * i)
        return v

    sf = straight(np.where(flush, fmask, 0))
    st = straight(rmask)
    t = top(rc >= 3, 1)[0]
    pairs = top(rc == 2, 2)
    has3 = (rc >= 3).any(1)
    fh_pair = top((rc >= 2) & (_AR != t[:, None]), 1)[0]
    conds = [flush & (sf >= 0), (rc == 4).any(1), has3 & ((rc >= 2).sum(1) >= 2), flush, st >= 0,
             has3, (rc == 2).sum(1) >= 2, (rc == 2).sum(1) == 1]
    vals = [(8 << 20) | (sf << 16),
            pack(7, [top(rc == 4, 1)[0], top((rc > 0) & (rc != 4), 1)[0]]),
            pack(6, [t, fh_pair]),
            pack(5, top(((fmask[:, None] >> _AR) & 1).astype(bool), 5)),
            (4 << 20) | (st << 16),
            pack(3, [t] + top(rc == 1, 2)),
            pack(2, pairs + top((rc > 0) & (_AR != pairs[0][:, None]) & (_AR != pairs[1][:, None]), 1)),
            pack(1, pairs[:1] + top(rc == 1, 3))]
    return np.select(conds, vals, pack(0, top(rc == 1, 5)))


def _monte_carlo(hole, board, n):
    dead = set(hole) | set(board)
    rest = np.array([c for c in range(52) if c not in dead])
    draws = rest[np.argsort(_nprng.random((n, len(rest))), 1)[:, :7 - len(board)]]
    full = np.hstack([np.tile(np.array(board, int).reshape(1, -1), (n, 1)), draws[:, 2:]])
    hero = np.hstack([np.tile(np.array(hole), (n, 1)), full])
    score = evaluate_batch(np.vstack([hero, np.hstack([draws[:, :2], full])]))
    return float(((score[:n] > score[n:]) + 0.5 * (score[:n] == score[n:])).mean())


PREFLOP_FILE = Path(__file__).parent / "models" / "preflop_equity.json"
_preflop_table = None


def _preflop(hi, lo, suited):
    global _preflop_table
    if _preflop_table is None:
        if PREFLOP_FILE.exists():
            _preflop_table = json.loads(PREFLOP_FILE.read_text())
        else:
            print("Computing the preflop equity table once (~15s)...")
            _preflop_table = {f"{h},{l},{int(s)}": _monte_carlo([h * 4, l * 4 + (0 if s else 1)], [], 6000)
                              for h in range(13) for l in range(h + 1) for s in (False, True) if not (s and h == l)}
            PREFLOP_FILE.parent.mkdir(exist_ok=True)
            PREFLOP_FILE.write_text(json.dumps(_preflop_table))
    return _preflop_table[f"{hi},{lo},{int(suited)}"]


@lru_cache(maxsize=300_000)
def _postflop(hole, board):
    return _monte_carlo(hole, board, 300)


def equity(hole, board):
    if not board:
        r = sorted((rank_of(c) for c in hole), reverse=True)
        return _preflop(r[0], r[1], suit_of(hole[0]) == suit_of(hole[1]) and r[0] != r[1])
    return _postflop(tuple(sorted(hole)), tuple(sorted(board)))


def raw_features(street, eq, pot, odds, my_stack, opp_stack, in_pos, raises, opp_raises, my_raises):
    x = np.zeros(N_FEATURES, np.float32)
    x[street] = 1
    x[4] = eq
    x[5 + min(int(eq * N_BUCKETS), N_BUCKETS - 1)] = 1
    x[13:] = [pot, odds, my_stack, opp_stack, in_pos, raises / 3, opp_raises / 4, my_raises / 4]
    return x


def features(g, p, eq=None):
    if eq is None:
        eq = equity(g.hole[p], g.board)
    total = sum(g.start_stacks)
    owed, pot = g.owed(p), g.pot()
    raises = [sum(1 for h in g.history if h[1] == q and h[2] == "raise") for q in (0, 1)]
    return raw_features(g.street, eq, pot / total, owed / (pot + owed), g.stacks[p] / total,
                        g.stacks[1 - p] / total, float(p == g.button), g.num_raises,
                        raises[1 - p], raises[p])


def raise_to(g, a):
    if a == ALLIN:
        return g.max_raise_to()
    owed = g.owed()
    target = max(g.street_bet) + round((0.5 if a == HALF else 1.0) * (g.pot() + owed))
    return min(max(target, g.min_raise_to()), g.max_raise_to())


def legal_mask(g):
    acts = g.legal_actions()
    m = np.zeros(N_ACTIONS, np.float32)
    m[CALL] = 1
    if "fold" in acts:
        m[FOLD] = 1
    if "raise" in acts:
        hi, seen = g.max_raise_to(), set()
        if g.num_raises < MAX_SIZED_RAISES:
            for a in (HALF, POT):
                amt = raise_to(g, a)
                if amt < hi and amt not in seen:
                    m[a] = 1
                    seen.add(amt)
        m[ALLIN] = 1
    return m


def to_engine(g, a):
    if a == FOLD:
        return ("fold", None)
    if a == CALL:
        return ("call", None) if g.owed() > 0 else ("check", None)
    return ("raise", raise_to(g, a))


def apply_abstract(g, a):
    g.apply_action(*to_engine(g, a))


def describe(g, action, amount):
    if action == "fold":
        return "folds"
    if action == "check":
        return "checks"
    if action == "call":
        return f"calls {min(g.owed(), g.stacks[g.to_act])}"
    allin = amount == g.max_raise_to()
    verb = "bets" if max(g.street_bet) == 0 else "raises to"
    return f"{verb} {amount}" + (" (all-in)" if allin else "")


def nearest_abstract(g, action, amount):
    if action == "fold":
        return FOLD
    if action in ("check", "call"):
        return CALL
    m = legal_mask(g)
    cands = [a for a in (HALF, POT, ALLIN) if m[a]]
    return min(cands, key=lambda a: abs(raise_to(g, a) - amount))


def regret_matching(adv, mask):
    pos = np.maximum(adv, 0) * mask
    s = pos.sum()
    if s > 0:
        return pos / s
    out = np.zeros_like(mask)
    out[np.argmax(np.where(mask > 0, adv, -np.inf))] = 1
    return out


def normalize_policy(out, mask):
    p = np.maximum(out, 0) * mask
    s = p.sum()
    return p / s if s > 0 else mask / mask.sum()


if __name__ == "__main__":
    from cards import ranks, suits
    hands = np.array([random.sample(range(52), 7) for _ in range(20000)])
    assert (evaluate_batch(hands) == [evaluate_fast(list(h)) for h in hands]).all(), "batch != fast"
    print("evaluate_batch == evaluate_fast on 20000 random hands")
    c = lambda s: [ranks.index(t[0]) * 4 + suits.index(t[1]) for t in s.split()]
    aa, k7 = equity(c("AS AH"), []), equity(c("7C 2D"), [])
    print(f"preflop equity AA {aa:.3f} (~0.85), 72o {k7:.3f} (~0.35)")
    assert 0.82 < aa < 0.88 and 0.30 < k7 < 0.39
    nuts = equity(c("AS KS"), c("QS JS TS 2D 3C"))
    assert nuts == 1.0, nuts
    print("abstraction self-check passed")
