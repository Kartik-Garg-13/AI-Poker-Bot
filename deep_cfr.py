import random

import numpy as np

from abstraction import (CALL, N_ACTIONS, N_FEATURES, ACTION_NAMES, FEATURE_NAMES, apply_abstract,
                         describe, equity, features, legal_mask, normalize_policy, raw_features,
                         regret_matching, to_engine)
from cards import card_str, category_name
from engine import GameState, STREETS
from nn import MLP


class Reservoir:
    def __init__(self, cap, seed=0):
        self.cap, self.size, self.seen = cap, 0, 0
        self.X = np.zeros((cap, N_FEATURES), np.float32)
        self.Y = np.zeros((cap, N_ACTIONS), np.float32)
        self.M = np.zeros((cap, N_ACTIONS), np.float32)
        self.T = np.zeros(cap, np.float32)
        self.rng = np.random.default_rng(seed)

    def add(self, x, y, m, t):
        if self.size < self.cap:
            i = self.size
            self.size += 1
        else:
            i = int(self.rng.integers(0, self.seen + 1))
            if i >= self.cap:
                self.seen += 1
                return
        self.X[i], self.Y[i], self.M[i], self.T[i] = x, y, m, t
        self.seen += 1

    def batch(self, k):
        i = self.rng.integers(0, self.size, k)
        return self.X[i], self.Y[i], self.M[i], self.T[i]


class DeepCFR:
    def __init__(self, stack=200, sb=1, bb=2, hidden=(32, 32), memory=400_000, seed=0, stack_range=None):
        self.stack, self.sb, self.bb = stack, sb, bb
        self.stack_range = stack_range or (stack, stack)
        sizes = [N_FEATURES, *hidden, N_ACTIONS]
        self.adv = [MLP(sizes, seed + 1), MLP(sizes, seed + 2)]
        self.strat = MLP(sizes, seed + 3)
        self.trained = [False, False]
        self.adv_mem = [Reservoir(memory, seed + 4), Reservoir(memory, seed + 5)]
        self.strat_mem = Reservoir(memory, seed + 6)
        self.t = 0
        self.rng = random.Random(seed)

    def new_game(self):
        lo, hi = self.stack_range
        g = GameState([self.rng.randint(lo, hi), self.rng.randint(lo, hi)], self.rng.randrange(2), self.rng)
        g.start_hand(self.sb, self.bb)
        return g

    def current_probs(self, g, x=None, mask=None):
        q = g.to_act
        x = features(g, q) if x is None else x
        mask = legal_mask(g) if mask is None else mask
        if not self.trained[q]:
            return mask / mask.sum()
        return regret_matching(self.adv[q].predict(x[None])[0], mask)

    def traverse(self, g, p):
        if g.hand_over:
            return g.net[p] / self.stack
        q = g.to_act
        x, mask = features(g, q), legal_mask(g)
        sigma = self.current_probs(g, x, mask)
        if q == p:
            vals = np.zeros(N_ACTIONS, np.float32)
            for a in np.flatnonzero(mask):
                child = g.clone()
                apply_abstract(child, a)
                vals[a] = self.traverse(child, p)
            ev = float(sigma @ vals)
            self.adv_mem[p].add(x, (vals - ev) * mask, mask, self.t)
            return ev
        self.strat_mem.add(x, sigma, mask, self.t)
        apply_abstract(g, self.rng.choices(range(N_ACTIONS), sigma)[0])
        return self.traverse(g, p)

    def train(self, net, mem, steps, batch=512, lr=1e-3, every=None, callback=None):
        losses = []
        for s in range(steps):
            X, Y, M, T = mem.batch(batch)
            losses.append(net.train_step(X, Y, M, T / T.mean(), lr))
            if callback and every and (s + 1) % every == 0:
                callback(s + 1, float(np.mean(losses[-every:])))

    def save(self, path, include_strategy=True):
        d = {}
        for i in (0, 1):
            d.update(self.adv[i].arrays(f"adv{i}_"))
        if include_strategy:
            d.update(self.strat.arrays("strat_"))
        np.savez(path, **d, meta=np.array([self.stack, self.sb, self.bb, self.t]))


class BotPolicy:
    def __init__(self, path):
        data = dict(np.load(path))
        self.stack, self.sb, self.bb, self.t = (int(v) for v in data["meta"])
        self.has_strat = "strat_W0" in data
        self.strat = MLP.from_arrays(data, "strat_") if self.has_strat else None
        self.adv = [MLP.from_arrays(data, f"adv{i}_") for i in (0, 1)]

    def probs_x(self, X, masks, player):
        if self.has_strat:
            return np.array([normalize_policy(o, m) for o, m in zip(self.strat.predict(X), masks)])
        return np.array([regret_matching(o, m) for o, m in zip(self.adv[player].predict(X), masks)])

    def probs(self, g):
        return self.probs_x(features(g, g.to_act)[None], legal_mask(g)[None], g.to_act)[0]


def sample_action(probs, rng):
    return rng.choices(range(N_ACTIONS), probs)[0]


def play_match(probs_a, probs_b, hands, stack=200, seed=0):
    rng = random.Random(seed)
    total = 0
    for i in range(hands // 2):
        for seat_a in (0, 1):
            g = GameState([stack, stack], i % 2, random.Random(seed * 1_000_003 + i))
            g.start_hand(1, 2)
            fns = [probs_a, probs_b] if seat_a == 0 else [probs_b, probs_a]
            while not g.hand_over:
                apply_abstract(g, sample_action(fns[g.to_act](g), rng))
            total += g.net[seat_a]
    return 100 * total / 2 / max(2 * (hands // 2), 1)


def random_bot(g):
    m = legal_mask(g)
    return m / m.sum()


def calling_station(g):
    p = np.zeros(N_ACTIONS, np.float32)
    p[CALL] = 1
    return p


def record_hand(probs_fn, hand_id, stack=200, rng=None):
    names = ("Bot A", "Bot B")
    rng = rng or random.Random()
    g = GameState([stack, stack], rng.randrange(2), rng)
    g.start_hand(1, 2)
    hole = [[card_str(c) for c in h] for h in g.hole]
    events = []
    behind = lambda: [g.start_stacks[i] - g.contributed[i] for i in (0, 1)]
    while not g.hand_over:
        p, seen = g.to_act, len(g.board)
        probs = probs_fn(g)
        mask = legal_mask(g)
        a = sample_action(probs, rng)
        action, amount = to_engine(g, a)
        text = describe(g, action, amount)
        g.apply_action(action, amount)
        events.append({"actor": p, "text": text, "chosen": ACTION_NAMES[a],
                       "probs": {ACTION_NAMES[i]: round(float(probs[i]), 3) for i in range(N_ACTIONS) if mask[i]},
                       "board": [card_str(c) for c in g.board[:seen]], "pot": g.pot(), "stacks": behind()})
        if len(g.board) > seen:
            events.append({"deal": "All-in: run it out" if g.hand_over else STREETS[g.street].title(),
                           "board": [card_str(c) for c in g.board], "pot": g.pot(), "stacks": behind()})
    winners = [names[w] for w in g.winners]
    if any(g.folded):
        result = f"{winners[0]} wins {g.pot()} chips (fold)"
    elif len(winners) == 2:
        result = f"Split pot ({category_name(g.scores[0])})"
    else:
        result = f"{winners[0]} wins {g.pot()} chips with {category_name(g.scores[g.winners[0]])}"
    return {"id": hand_id, "names": list(names), "hole": hole, "board": [], "pot": 3,
            "stacks": [stack - (1 if i == g.button else 2) for i in (0, 1)], "button": g.button,
            "events": events, "result": result}


def _pf(hand):
    from cards import ranks, suits
    return equity([ranks.index(t[0]) * 4 + suits.index(t[1]) for t in hand.split()], [])


PROBES = [
    ("Preflop: AA on the button, first to act", lambda: raw_features(0, _pf("AS AH"), 3 / 400, 1 / 4, 199 / 400, 198 / 400, 1, 0, 0, 0)),
    ("Preflop: 7-2 offsuit on the button", lambda: raw_features(0, _pf("7S 2H"), 3 / 400, 1 / 4, 199 / 400, 198 / 400, 1, 0, 0, 0)),
    ("Flop: top pair (eq .80), facing a pot bet", lambda: raw_features(1, 0.80, 24 / 400, 8 / 24, 192 / 400, 184 / 400, 0, 1, 1, 0)),
    ("River: near-nuts (eq .97), checked to me", lambda: raw_features(3, 0.97, 40 / 400, 0, 180 / 400, 180 / 400, 1, 0, 0, 1)),
    ("River: air (eq .08), checked to me -- bluff?", lambda: raw_features(3, 0.08, 40 / 400, 0, 180 / 400, 180 / 400, 1, 0, 0, 1)),
    ("River: bluff-catcher (eq .55), facing all-in", lambda: raw_features(3, 0.55, 220 / 400, 180 / 220, 180 / 400, 0, 0, 1, 2, 0)),
]


def network_view(net, name, kind="advantage"):
    return {"name": name, "kind": kind, "sizes": net.sizes, "steps": net.steps,
            "weights": [np.round(W, 3).tolist() for W in net.W],
            "biases": [np.round(b, 3).tolist() for b in net.b],
            "input_labels": FEATURE_NAMES, "output_labels": ACTION_NAMES,
            "probes": [{"label": label, "x": np.round(f(), 4).tolist()} for label, f in PROBES]}
