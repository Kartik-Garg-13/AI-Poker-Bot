import random
from pathlib import Path

import numpy as np

from abstraction import equity, normalize_policy
from nn import MLP

FOLD, CALL, HALF, POT, ALLIN = range(5)
MODEL = Path(__file__).parent / "models" / "v1_holdem.npz"


def features(g, p, hole=None):
    x = np.zeros(21, np.float32)
    eq = equity(g.hole[p] if hole is None else hole, g.board)
    total = sum(g.start_stacks)
    owed, pot = g.owed(p), g.pot()
    raises = [sum(1 for h in g.history if h[1] == q and h[2] == "raise") for q in (0, 1)]
    x[g.street] = 1
    x[4] = eq
    x[5 + min(int(eq * 8), 7)] = 1
    x[13:] = [pot / total, owed / (pot + owed), g.stacks[p] / total, g.stacks[1 - p] / total,
              float(p == g.button), g.num_raises / 3, raises[1 - p] / 4, raises[p] / 4]
    return x


def raise_to(g, a):
    if a == ALLIN:
        return g.max_raise_to()
    target = max(g.street_bet) + round((0.5 if a == HALF else 1.0) * (g.pot() + g.owed()))
    return min(max(target, g.min_raise_to()), g.max_raise_to())


def legal_mask(g):
    acts = g.legal_actions()
    m = np.zeros(5, np.float32)
    m[CALL] = 1
    if "fold" in acts:
        m[FOLD] = 1
    if "raise" in acts:
        hi, seen = g.max_raise_to(), set()
        if g.num_raises < 2:
            for a in (HALF, POT):
                amt = raise_to(g, a)
                if amt < hi and amt not in seen:
                    m[a] = 1
                    seen.add(amt)
        m[ALLIN] = 1
    return m


def translate(g, action, amount):
    if action == "fold":
        return [(FOLD, 1.0)]
    if action in ("check", "call"):
        return [(CALL, 1.0)]
    m = legal_mask(g)
    return [(next(a for a in (HALF, POT, ALLIN) if m[a] and raise_to(g, a) == amount), 1.0)]


def to_engine(g, a):
    if a == FOLD:
        return "fold", None
    if a == CALL:
        return ("call", None) if g.owed() > 0 else ("check", None)
    return "raise", raise_to(g, a)


class V1Bot:
    name = "v1 bot"

    def __init__(self, path=MODEL):
        self.net = MLP.from_arrays(dict(np.load(path)), "strat_")

    def new_hand(self, rng):
        pass

    def probs_x(self, X, masks, player=None):
        return np.array([normalize_policy(o, m) for o, m in zip(self.net.predict(X), masks)])

    def likelihood(self, steps, player):
        out = 1.0
        for X, masks, aw in steps:
            out = out * (self.probs_x(X, masks) @ aw)
        return out

    def probs(self, g):
        return self.probs_x(features(g, g.to_act)[None], legal_mask(g)[None])[0]

    def act(self, g, rng):
        return to_engine(g, rng.choices(range(5), self.probs(g).astype(float).tolist())[0])


if __name__ == "__main__":
    from engine import GameState
    bot, rng = V1Bot(), random.Random(0)
    for i in range(200):
        g = GameState([200, 200], i % 2, rng)
        g.start_hand(1, 2)
        while not g.hand_over:
            g.apply_action(*bot.act(g, rng))
    print("v1 bot plays 200 legal hands against itself")
