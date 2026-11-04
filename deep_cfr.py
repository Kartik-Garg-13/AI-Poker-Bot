import os
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

    def add_many(self, rows):
        if rows is not None:
            for r in zip(*rows):
                self.add(*r)

    def batch(self, k):
        i = self.rng.integers(0, self.size, k)
        return self.X[i], self.Y[i], self.M[i], self.T[i]

    def arrays(self, prefix):
        n = self.size
        return {f"{prefix}X": self.X[:n].astype(np.float16), f"{prefix}Y": self.Y[:n].astype(np.float16),
                f"{prefix}M": self.M[:n].astype(np.uint8), f"{prefix}T": self.T[:n], f"{prefix}seen": np.array(self.seen)}

    def load(self, data, prefix):
        n = min(len(data[f"{prefix}X"]), self.cap)
        self.X[:n], self.Y[:n] = data[f"{prefix}X"][:n], data[f"{prefix}Y"][:n]
        self.M[:n], self.T[:n] = data[f"{prefix}M"][:n], data[f"{prefix}T"][:n]
        self.size, self.seen = n, max(int(data[f"{prefix}seen"]), n)


class Collector:
    def __init__(self):
        self.rows = []

    def add(self, x, y, m, t):
        self.rows.append((x, y, m, t))

    def arrays(self):
        if not self.rows:
            return None
        X, Y, M, T = zip(*self.rows)
        return np.array(X), np.array(Y), np.array(M), np.array(T, np.float32)


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
        self.finite, self.bust_penalty = False, 0.5
        self.session, self.busts = [stack, stack], [0, 0]
        self.human_hands, self.human_share = [], 0.2

    def new_game(self):
        if self.finite:
            if max(self.session) > 5 * self.stack:
                self.session = [self.stack, self.stack]
            for i in (0, 1):
                if self.session[i] == 0:
                    self.session[i] = self.stack
                    self.busts[i] += 1
            stacks = self.session[:]
        else:
            lo, hi = self.stack_range
            stacks = [self.rng.randint(lo, hi), self.rng.randint(lo, hi)]
        g = GameState(stacks, self.rng.randrange(2), self.rng)
        g.start_hand(self.sb, self.bb)
        return g

    def human_game(self):
        h = self.rng.choice(self.human_hands)
        g = GameState(h["start_stacks"], h["button_seat"], self.rng)
        hole, board = h["deal"]["hole"], h["deal"]["board"]
        dealt = [hole[0][0], hole[1][0], hole[0][1], hole[1][1], *board]
        g.deck = [c for c in g.deck if c not in dealt] + dealt[::-1]
        g.start_hand(self.sb, self.bb)
        for action, amount in h["moves"][:self.rng.randrange(len(h["moves"]))]:
            g.apply_action(action, amount)
        self.rng.shuffle(g.deck)
        return g

    def step(self, p):
        if self.human_hands and self.rng.random() < self.human_share:
            self.traverse(self.human_game(), p)
            return
        g = self.new_game()
        real = g.clone() if self.finite else None
        self.traverse(g, p)
        if real is not None:
            while not real.hand_over:
                apply_abstract(real, sample_action(self.current_probs(real), self.rng))
            self.session = real.stacks[:]

    def current_probs(self, g, x=None, mask=None):
        q = g.to_act
        x = features(g, q) if x is None else x
        mask = legal_mask(g) if mask is None else mask
        if not self.trained[q]:
            return mask / mask.sum()
        return regret_matching(self.adv[q].predict(x[None])[0], mask)

    def traverse(self, g, p):
        if g.hand_over:
            busted = self.finite and g.stacks[p] == 0
            return g.net[p] / self.stack - (self.bust_penalty if busted else 0.0)
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
        apply_abstract(g, sample_action(sigma, self.rng))
        return self.traverse(g, p)

    def train(self, net, mem, steps, batch=512, lr=1e-3, every=None, callback=None):
        losses = []
        for s in range(steps):
            X, Y, M, T = mem.batch(batch)
            losses.append(net.train_step(X, Y, M, T / T.mean(), lr))
            if callback and every and (s + 1) % every == 0:
                callback(s + 1, float(np.mean(losses[-every:])))

    def save_memories(self, path):
        d = {"t": np.array(self.t)}
        for name, mem in (("adv0_", self.adv_mem[0]), ("adv1_", self.adv_mem[1]), ("strat_", self.strat_mem)):
            d.update(mem.arrays(name))
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **d)
        os.replace(tmp, path)

    def load_memories(self, path):
        data = np.load(path)
        for name, mem in (("adv0_", self.adv_mem[0]), ("adv1_", self.adv_mem[1]), ("strat_", self.strat_mem)):
            mem.load(data, name)
        return int(data["t"])

    def save(self, path, include_strategy=True):
        d = {}
        for i in (0, 1):
            d.update(self.adv[i].arrays(f"adv{i}_"))
        if include_strategy:
            d.update(self.strat.arrays("strat_"))
        np.savez(path, **d, meta=np.array([self.stack, self.sb, self.bb, self.t]))

    def save_snapshot(self, folder):
        folder.mkdir(parents=True, exist_ok=True)
        d = {}
        for i in (0, 1):
            d.update({k: v.astype(np.float16) for k, v in self.adv[i].arrays(f"adv{i}_").items()})
        np.savez(folder / f"iter_{self.t:05d}.npz", **d)


_worker = None


def worker_init(cfg):
    global _worker
    _worker = DeepCFR(cfg["stack"], hidden=cfg["hidden"], memory=1, seed=cfg["seed"] + os.getpid(),
                      stack_range=cfg["stack_range"])
    _worker.bust_penalty = cfg["bust_penalty"]


def worker_traverse(job):
    d = _worker
    d.adv = [MLP.from_arrays(job["nets"], f"adv{i}_") for i in (0, 1)]
    d.trained, d.t, d.finite = job["trained"], job["t"], job["finite"]
    d.human_hands, d.human_share = job["human"], job["human_share"]
    d.adv_mem, d.strat_mem = [Collector(), Collector()], Collector()
    for _ in range(job["n"]):
        d.step(job["p"])
    return d.adv_mem[job["p"]].arrays(), d.strat_mem.arrays(), d.session[:], d.busts[:]


def pack_ensemble(folder, out, meta, n=64):
    files = sorted(folder.glob("iter_*.npz"))
    pick = [files[round(i)] for i in np.linspace(0, len(files) - 1, min(n, len(files)))]
    pick = list(dict.fromkeys(pick))
    snaps = [dict(np.load(f)) for f in pick]
    d = {"snap_iter": np.array([int(f.stem.split("_")[1]) for f in pick])}
    for key in snaps[0]:
        if key.startswith("adv0_"):
            name = key[5:]
            d[f"snap_{name}"] = np.stack([np.stack([s[f"adv0_{name}"], s[f"adv1_{name}"]]) for s in snaps])
    for i in (0, 1):
        d.update({k: v.astype(np.float32) for k, v in snaps[-1].items() if k.startswith(f"adv{i}_")})
    np.savez(out, **d, meta=np.array(meta))


def regret_matching_batch(adv, mask):
    pos = np.maximum(adv, 0) * mask
    tot = pos.sum(-1, keepdims=True)
    best = np.argmax(np.where(mask > 0, adv, -np.inf), -1)
    greedy = (np.arange(adv.shape[-1]) == best[..., None]).astype(adv.dtype)
    return np.where(tot > 0, pos / np.where(tot > 0, tot, 1), greedy)


class BotPolicy:
    def __init__(self, path):
        data = dict(np.load(path))
        self.stack, self.sb, self.bb, self.t = (int(v) for v in data["meta"])
        self.kind = "ensemble" if "snap_W0" in data else "strategy" if "strat_W0" in data else "advantage"
        self.has_strat = self.kind != "advantage"
        self.strat = MLP.from_arrays(data, "strat_") if self.kind == "strategy" else None
        self.adv = [MLP.from_arrays(data, f"adv{i}_") for i in (0, 1)]
        if self.kind == "ensemble":
            n = sum(1 for k in data if k.startswith("snap_W"))
            self.W = [data[f"snap_W{i}"].astype(np.float32) for i in range(n)]
            self.b = [data[f"snap_b{i}"].astype(np.float32) for i in range(n)]
            self.weights = data["snap_iter"] / data["snap_iter"].sum()
        self.k = None

    def new_hand(self, rng):
        if self.kind == "ensemble":
            self.k = rng.choices(range(len(self.weights)), self.weights.tolist())[0]

    def _snap_probs(self, X, masks, player, k=slice(None)):
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W[k, player] + b[k, player][..., None, :]
            if i < len(self.W) - 1:
                h = np.maximum(h, 0)
        return regret_matching_batch(h, masks)

    def probs_x(self, X, masks, player):
        if self.kind == "ensemble":
            if self.k is not None:
                return self._snap_probs(X, masks, player, self.k)
            return np.tensordot(self.weights, self._snap_probs(X, masks, player), 1)
        if self.kind == "strategy":
            return np.array([normalize_policy(o, m) for o, m in zip(self.strat.predict(X), masks)])
        return regret_matching_batch(self.adv[player].predict(X), masks)

    def likelihood(self, steps, player):
        if self.kind != "ensemble":
            out = 1.0
            for X, masks, aw in steps:
                out = out * (self.probs_x(X, masks, player) @ aw)
            return out
        out = 1.0
        for X, masks, aw in steps:
            out = out * (self._snap_probs(X, masks, player) @ aw)
        return self.weights @ out if steps else np.ones(1)

    def probs(self, g):
        return self.probs_x(features(g, g.to_act)[None], legal_mask(g)[None], g.to_act)[0]

    def act(self, g, rng):
        return to_engine(g, sample_action(self.probs(g), rng))


def sample_action(probs, rng):
    return rng.choices(range(len(probs)), np.asarray(probs, float).tolist())[0]


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


def record_hand(probs_fn, hand_id, stack=200, rng=None, stacks=None):
    names = ("Bot A", "Bot B")
    rng = rng or random.Random()
    g = GameState(list(stacks or [stack, stack]), rng.randrange(2), rng)
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
            "stacks": [g.start_stacks[i] - (1 if i == g.button else 2) for i in (0, 1)], "button": g.button,
            "events": events, "result": result}


def _pf(hand):
    from cards import ranks, suits
    return equity([ranks.index(t[0]) * 4 + suits.index(t[1]) for t in hand.split()], [])


def _probe(street, eq, pot, owed, stacks, in_pos, raises, opp_raises, my_raises, shape):
    total = 400
    eff = min(stacks)
    return lambda: raw_features(street, eq, pot / total, owed / (pot + owed), stacks[0] / total, stacks[1] / total,
                                in_pos, raises / 3, opp_raises / 4, my_raises / 4, np.log1p(eff / 2) / np.log1p(250),
                                np.log1p(eff / pot) / np.log1p(100), *shape, 0, float(opp_raises > 0 and street > 0))


PROBES = [
    ("Preflop: AA on the button, first to act", lambda: _probe(0, _pf("AS AH"), 3, 1, (199, 198), 1, 0, 0, 0, (1 / 8, 0, 0, 0, 0, 0, 0))()),
    ("Preflop: 7-2 offsuit on the button", lambda: _probe(0, _pf("7S 2H"), 3, 1, (199, 198), 1, 0, 0, 0, (0,) * 7)()),
    ("Flop: top pair (eq .80), facing a pot bet", _probe(1, 0.80, 24, 8, (192, 184), 0, 1, 1, 0, (1 / 8, 0, 0, 0, 0, 1, 1))),
    ("Flop: flush draw (eq .45), checked to me", _probe(1, 0.45, 8, 0, (196, 196), 1, 0, 0, 0, (0, 1, 0, 0, 0, 0, 1))),
    ("River: near-nuts (eq .97), checked to me", _probe(3, 0.97, 40, 0, (180, 180), 1, 0, 0, 1, (5 / 8, 0, 0, 0, 1 / 3, 0, 1))),
    ("River: air (eq .08), checked to me -- bluff?", _probe(3, 0.08, 40, 0, (180, 180), 1, 0, 0, 1, (0, 0, 0, 0, 0, 1, 1))),
    ("River: bluff-catcher (eq .55), facing all-in", _probe(3, 0.55, 220, 180, (180, 0), 0, 1, 2, 0, (2 / 8, 0, 0, 1, 0, 0, 1))),
]


def network_view(net, name, kind="advantage"):
    return {"name": name, "kind": kind, "sizes": net.sizes, "steps": net.steps,
            "weights": [np.round(W, 3).tolist() for W in net.W],
            "biases": [np.round(b, 3).tolist() for b in net.b],
            "input_labels": FEATURE_NAMES, "output_labels": ACTION_NAMES,
            "probes": [{"label": label, "x": np.round(f(), 4).tolist()} for label, f in PROBES]}
