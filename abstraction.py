import json
import random
from functools import lru_cache
from pathlib import Path

import numpy as np

from cards import _straight_in_mask, evaluate_fast, rank_of, suit_of

FOLD, CALL, B33, B50, B75, POT, B200, ALLIN = range(8)
ACTION_NAMES = ["fold", "check/call", "bet 1/3 pot", "bet 1/2 pot", "bet 3/4 pot", "bet pot", "bet 2x pot", "all-in"]
N_ACTIONS = len(ACTION_NAMES)
SIZES = {B33: 1 / 3, B50: 0.5, B75: 0.75, POT: 1.0, B200: 2.0}
OPENING, FACING = (B33, B75, POT, B200), (B50, POT)
BETS = list(SIZES) + [ALLIN]
MAX_SIZED_RAISES = 2
N_BUCKETS = 8
FEATURE_NAMES = ["preflop", "flop", "turn", "river", "equity"] +                 [f"bucket {i}" for i in range(N_BUCKETS)] +                 ["pot size", "pot odds", "my stack", "opp stack", "in position",
                 "raises this street", "opp raises (hand)", "my raises (hand)",
                 "stack depth", "stack-to-pot", "made hand", "flush draw", "straight draw",
                 "board paired", "board suited", "board connected", "board high card",
                 "my initiative", "opp initiative"]
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


def _mc_rows(hole, board, n):
    dead = set(hole) | set(board)
    rest = np.array([c for c in range(52) if c not in dead])
    draws = rest[np.argsort(_nprng.random((n, len(rest))), 1)[:, :7 - len(board)]]
    full = np.hstack([np.tile(np.array(board, int).reshape(1, -1), (n, 1)), draws[:, 2:]])
    return np.hstack([np.tile(np.array(hole), (n, 1)), full]), np.hstack([draws[:, :2], full])


def _monte_carlo(hole, board, n):
    hero, opp = _mc_rows(hole, board, n)
    score = evaluate_batch(np.vstack([hero, opp]))
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


def _has_straight(mask):
    return _straight_in_mask(mask) is not None


@lru_cache(maxsize=300_000)
def hand_shape(hole, board):
    if not board:
        return float(rank_of(hole[0]) == rank_of(hole[1])) / 8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    cards = hole + board
    suits = [sum(1 for c in cards if suit_of(c) == s) for s in range(4)]
    bsuits = [sum(1 for c in board if suit_of(c) == s) for s in range(4)]
    mask = bmask = 0
    for c in cards:
        mask |= 1 << rank_of(c)
    for c in board:
        bmask |= 1 << rank_of(c)
    live = len(board) < 5
    flush_draw = live and any(suits[s] == 4 and bsuits[s] < 4 for s in range(4))
    outs = 0
    if live and not _has_straight(mask):
        outs = sum(1 for r in range(13) if _has_straight(mask | 1 << r) and not _has_straight(bmask | 1 << r))
    ext = (bmask << 1) | (bmask >> 12)
    connected = any(bin((ext >> lo) & 31).count("1") >= 3 for lo in range(10))
    return ((evaluate_fast(list(cards)) >> 20) / 8, float(flush_draw), min(outs, 2) / 2,
            float(len({rank_of(c) for c in board}) < len(board)), max(0, max(bsuits) - 2) / 3,
            float(connected), max(rank_of(c) for c in board) / 12)


def raw_features(street, eq, *rest):
    x = np.zeros(N_FEATURES, np.float32)
    x[street] = 1
    x[4] = eq
    x[5 + min(int(eq * N_BUCKETS), N_BUCKETS - 1)] = 1
    x[13:13 + len(rest)] = rest
    return x


def features(g, p, hole=None):
    hole = tuple(sorted(g.hole[p] if hole is None else hole))
    board = tuple(sorted(g.board))
    total = sum(g.start_stacks)
    owed, pot = g.owed(p), g.pot()
    raises = [0, 0]
    init = None
    for street, q, action, _ in g.history:
        if action == "raise":
            raises[q] += 1
            if street < g.street:
                init = q
    eff = min(g.stacks)
    return raw_features(g.street, equity(hole, board), pot / total, owed / (pot + owed), g.stacks[p] / total,
                        g.stacks[1 - p] / total, float(p == g.button), g.num_raises / 3,
                        raises[1 - p] / 4, raises[p] / 4, np.log1p(eff / g.bb) / np.log1p(250),
                        np.log1p(eff / pot) / np.log1p(100), *hand_shape(hole, board),
                        float(init == p), float(init == 1 - p))


def raise_to(g, a):
    if a == ALLIN:
        return g.max_raise_to()
    target = max(g.street_bet) + round(SIZES[a] * (g.pot() + g.owed()))
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
            for a in (FACING if g.owed() > 0 else OPENING):
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


def translate(g, action, amount):
    if action == "fold":
        return [(FOLD, 1.0)]
    if action in ("check", "call"):
        return [(CALL, 1.0)]
    m = legal_mask(g)
    base = g.pot() + g.owed()
    frac = lambda amt: (amt - max(g.street_bet)) / base
    sizes = sorted((frac(raise_to(g, a)), a) for a in BETS if m[a])
    x = frac(amount)
    if x <= sizes[0][0]:
        return [(sizes[0][1], 1.0)]
    for (lo, a), (hi, b) in zip(sizes, sizes[1:]):
        if x <= hi:
            f = (hi - x) * (1 + lo) / ((hi - lo) * (1 + x))
            return [(a, f), (b, 1 - f)]
    return [(sizes[-1][1], 1.0)]


def nearest_abstract(g, action, amount):
    return max(translate(g, action, amount), key=lambda t: t[1])[0]


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
    from engine import GameState
    g = GameState([200, 200], 0, random.Random(1))
    dealt = c("7D KS 2C AS QS JS 3H 9C TD")
    g.deck = [x for x in g.deck if x not in dealt] + dealt[::-1]
    g.start_hand()
    g.apply_action("call")
    g.apply_action("check")
    x = dict(zip(FEATURE_NAMES, features(g, 1)))
    assert x["flush draw"] == 1 and x["straight draw"] == 0.5 and x["board connected"] == 0, x
    assert legal_mask(g)[[B33, B75, POT, B200]].all() and not legal_mask(g)[B50]
    g.apply_action("raise", raise_to(g, POT))
    assert legal_mask(g)[[FOLD, B50, POT, ALLIN]].all() and not legal_mask(g)[B33]
    mix = translate(g, "raise", raise_to(g, POT) + 6)
    assert len(mix) == 2 and abs(sum(w for _, w in mix) - 1) < 1e-9, mix
    print("abstraction self-check passed")
