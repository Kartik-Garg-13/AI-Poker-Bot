import random

import numpy as np

from abstraction import N_ACTIONS, apply_abstract, equity, features, legal_mask
from cards import evaluate_fast


def sample_hands(dead, k, rng):
    rest = [c for c in range(52) if c not in dead]
    combos = [(a, b) for i, a in enumerate(rest) for b in rest[i + 1:]]
    return rng.sample(combos, min(k, len(combos)))


def range_weights(hands, decisions, who, blueprint):
    w = np.ones(len(hands))
    for snap, actor, a in decisions:
        if actor != who:
            continue
        X = np.array([features(snap, actor, equity(list(h), snap.board)) for h in hands])
        masks = np.tile(legal_mask(snap), (len(hands), 1))
        w *= blueprint.probs_x(X, masks, actor)[:, a]
    if w.sum() <= 0:
        w = np.ones(len(hands))
    return w / w.sum()


class Node:
    __slots__ = ("player", "actions", "children", "fold_by", "contrib", "regret", "strat_sum")

    def __init__(self, g):
        self.regret = self.strat_sum = None
        if g.hand_over:
            self.player, self.contrib = None, list(g.contributed)
            self.fold_by = g.folded.index(True) if any(g.folded) else None
            return
        self.player = g.to_act
        self.actions = [int(a) for a in np.flatnonzero(legal_mask(g))]
        self.children = []
        for a in self.actions:
            c = g.clone()
            apply_abstract(c, a)
            self.children.append(Node(c))


def _rm(R):
    s = R.sum(1, keepdims=True)
    return np.where(s > 0, R / np.where(s > 0, s, 1), 1 / R.shape[1])


class RiverSolver:
    def __init__(self, g, bot, hands, weights):
        self.bot = bot
        score = [np.array([evaluate_fast(list(h) + g.board) for h in hs]) for hs in hands]
        self.V = np.array([[not set(a) & set(b) for b in hands[1]] for a in hands[0]], float)
        self.M = np.sign(score[0][:, None] - score[1][None, :]) * self.V
        self.r = weights
        self.root = Node(g)
        self.t = 0

    def _side(self, n):
        return 0 if n.player == self.bot else 1

    def _terminal(self, n, r0, r1):
        c_bot, c_opp = n.contrib[self.bot], n.contrib[1 - self.bot]
        if n.fold_by is not None:
            u = -c_bot if n.fold_by == self.bot else c_opp
            return u * (self.V @ r1), -u * (self.V.T @ r0)
        m = min(c_bot, c_opp)
        return m * (self.M @ r1), -m * (self.M.T @ r0)

    def _cfr(self, n, r):
        if n.player is None:
            return self._terminal(n, *r)
        s = self._side(n)
        if n.regret is None:
            n.regret = np.zeros((len(r[s]), len(n.children)))
            n.strat_sum = np.zeros_like(n.regret)
        sigma = _rm(n.regret)
        mine = np.zeros_like(n.regret)
        other = 0
        for k, child in enumerate(n.children):
            rr = list(r)
            rr[s] = r[s] * sigma[:, k]
            v = self._cfr(child, rr)
            mine[:, k] = v[s]
            other = other + v[1 - s]
        v_s = (sigma * mine).sum(1)
        n.regret = np.maximum(n.regret + mine - v_s[:, None], 0)
        n.strat_sum += self.t * r[s][:, None] * sigma
        return (v_s, other) if s == 0 else (other, v_s)

    def solve(self, iters):
        for _ in range(iters):
            self.t += 1
            self._cfr(self.root, list(self.r))

    def _br(self, n, s, r):
        if n.player is None:
            return self._terminal(n, *r)[s]
        vals = []
        avg = None if self._side(n) == s else _rm(n.strat_sum)
        for k, child in enumerate(n.children):
            rr = list(r)
            if avg is not None:
                rr[1 - s] = r[1 - s] * avg[:, k]
            vals.append(self._br(child, s, rr))
        return np.max(vals, 0) if avg is None else np.sum(vals, 0)

    def exploitability(self):
        r0, r1 = self.r
        b0 = r0 @ self._br(self.root, 0, [np.zeros_like(r0), r1])
        b1 = r1 @ self._br(self.root, 1, [r0, np.zeros_like(r1)])
        return (b0 + b1) / 2 / (r0 @ self.V @ r1)


def resolve_river(g, decisions, blueprint, n_hands=100, iters=400, rng=None):
    rng = rng or random.Random()
    bot = g.to_act
    me = tuple(sorted(g.hole[bot]))
    dead = set(g.board)
    bot_hands = [me] + [h for h in sample_hands(dead, n_hands, rng) if tuple(sorted(h)) != me][:n_hands - 1]
    opp_hands = sample_hands(dead, n_hands, rng)
    weights = [range_weights(bot_hands, decisions, bot, blueprint),
               range_weights(opp_hands, decisions, 1 - bot, blueprint)]
    solver = RiverSolver(g, bot, [bot_hands, opp_hands], weights)
    solver.solve(iters)
    probs = np.zeros(N_ACTIONS)
    probs[solver.root.actions] = _rm(solver.root.strat_sum)[0]
    info = {"iterations": iters, "range_size": n_hands,
            "exploitability_pct_pot": round(100 * solver.exploitability() / g.pot(), 2),
            "opp_strength": round(float(sum(w * equity(list(h), g.board) for h, w in zip(opp_hands, weights[1]))), 3)}
    return probs, info


if __name__ == "__main__":
    from engine import GameState
    rng = random.Random(3)
    g = GameState([200, 200], 0, rng)
    g.start_hand()
    g.apply_action("call")
    g.apply_action("check")
    while g.street < 3:
        g.apply_action("check")
        g.apply_action("check")
    hands = [sample_hands(set(g.board), 80, rng) for _ in range(2)]
    s = RiverSolver(g, g.to_act, hands, [np.ones(80) / 80, np.ones(80) / 80])
    s.solve(10)
    e10 = s.exploitability()
    s.solve(390)
    e400 = s.exploitability()
    print(f"river subgame exploitability: {100 * e10 / g.pot():.2f}% pot after 10 iters, "
          f"{100 * e400 / g.pot():.2f}% after 400")
    assert e400 < e10 / 10 and e400 / g.pot() < 0.03
    print("search self-check passed")
