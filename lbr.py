import argparse
import random
import time

import numpy as np

import abstraction
import legacy
from abstraction import CALL, FOLD, evaluate_batch, legal_mask, raise_to
from deep_cfr import BotPolicy
from engine import GameState
from evaluate import make_pool


class Model:
    def __init__(self, spec):
        lib = legacy if spec == "v1" else abstraction
        self.bot = legacy.V1Bot() if spec == "v1" else BotPolicy(spec)
        self.features, self.mask, self.translate = lib.features, lib.legal_mask, lib.translate
        self.n = 5 if spec == "v1" else abstraction.N_ACTIONS

    def spot(self, g, hands):
        p = g.to_act
        X = np.array([self.features(g, p, h) for h in hands])
        return X, np.tile(self.mask(g), (len(hands), 1))


def win_prob(hole, board, hands, rng, k=16):
    dead = set(hole) | set(board)
    rows_me, rows_opp = [], []
    for h in hands:
        rest = [c for c in range(52) if c not in dead and c not in h]
        for _ in range(k):
            full = board + rng.sample(rest, 5 - len(board))
            rows_me.append(list(hole) + full)
            rows_opp.append(list(h) + full)
    s = evaluate_batch(np.array(rows_me + rows_opp))
    n = len(rows_me)
    return ((s[:n] > s[n:]) + 0.5 * (s[:n] == s[n:])).reshape(len(hands), k).mean(1)


def lbr_action(g, model, hands, weights, rng):
    me = g.to_act
    if g.street == 0:
        return ("call", None) if g.owed() > 0 else ("check", None)
    valid = np.array([not (set(h) & set(g.board)) for h in hands], float)
    w = weights * valid
    if w.sum() <= 0:
        return ("call", None) if g.owed() > 0 else ("check", None)
    w = w / w.sum()
    wp = win_prob(g.hole[me], g.board, hands, rng)
    pot, owed = g.pot(), g.owed()
    best, choice = 0.0, ("fold", None) if owed > 0 else ("check", None)
    call = float(w @ wp) * (pot + owed) - owed
    if call > best or owed == 0:
        best, choice = call, (("call", None) if owed > 0 else ("check", None))
    m = legal_mask(g)
    for a in np.flatnonzero(m):
        if a in (FOLD, CALL):
            continue
        amount = raise_to(g, a)
        g2 = g.clone()
        g2.apply_action("raise", amount)
        if g2.hand_over:
            continue
        X, M = model.spot(g2, hands)
        pf = model.bot.probs_x(X, M, g2.to_act)[:, FOLD]
        fold = float(w @ pf)
        stay = w * (1 - pf)
        wp_c = float(stay @ wp / stay.sum()) if stay.sum() > 0 else 0.0
        x = amount - g.street_bet[me]
        final = pot + x + (amount - g.street_bet[1 - me])
        u = fold * pot + (1 - fold) * (wp_c * final - x)
        if u > best:
            best, choice = u, ("raise", amount)
    return choice


def lbr_hands(spec, hands, stack, seed, n_range=100):
    model = Model(spec)
    bot = Model(spec).bot
    rng = random.Random(seed)
    total = 0
    for i in range(hands):
        g = GameState([stack, stack], i % 2, random.Random(seed * 1_000_003 + i))
        g.start_hand(1, 2)
        lbr = (i // 2) % 2
        dead = set(g.hole[lbr])
        combos = [(a, b) for a in range(52) for b in range(a + 1, 52) if a not in dead and b not in dead]
        rng_hands = rng.sample(combos, n_range)
        steps = []
        bot.new_hand(rng)
        while not g.hand_over:
            if g.to_act == lbr:
                weights = np.broadcast_to(model.bot.likelihood(steps, 1 - lbr), (n_range,)) if steps else np.ones(n_range)
                g.apply_action(*lbr_action(g, model, rng_hands, np.array(weights, float), rng))
            else:
                X, M = model.spot(g, rng_hands)
                action, amount = bot.act(g, rng)
                aw = np.zeros(model.n)
                for a, share in model.translate(g, action, amount):
                    aw[a] += share
                steps.append((X, M, aw))
                g.apply_action(action, amount)
        total += g.net[lbr]
    return 100 * total / 2 / hands


def _job(args):
    return lbr_hands(*args)


def main():
    ap = argparse.ArgumentParser(description="Local best response: how much a simple exploiter that knows the bot's "
                                             "whole strategy wins from it, in bb/100. Lower means harder to exploit.")
    ap.add_argument("bots", nargs="+", help="model files, or v1")
    ap.add_argument("--hands", type=int, default=2000)
    ap.add_argument("--stack", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    chunk = 50
    with make_pool(args.workers) as pool:
        for spec in args.bots:
            t0 = time.time()
            res = np.array(pool.map(_job, [(spec, chunk, args.stack, 31 + k) for k in range(args.hands // chunk)]))
            print(f"LBR vs {spec} at {args.stack // 2}bb: LBR wins {res.mean():+.1f} +/- "
                  f"{res.std(ddof=1) / np.sqrt(len(res)):.1f} bb/100 over {len(res) * chunk} hands "
                  f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
