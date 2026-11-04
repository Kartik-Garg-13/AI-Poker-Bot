import argparse
import os
import random
import time
from multiprocessing import Pool

import numpy as np

from abstraction import CALL, features, legal_mask, regret_matching, to_engine
from deep_cfr import BotPolicy, sample_action
from engine import GameState
from legacy import V1Bot
from nn import MLP


class PolicyBot:
    def __init__(self, fn):
        self.fn = fn

    def new_hand(self, rng):
        pass

    def act(self, g, rng):
        return to_engine(g, sample_action(self.fn(g), rng))


def _random(g):
    m = legal_mask(g)
    return m / m.sum()


def _station(g):
    m = np.zeros(len(legal_mask(g)), np.float32)
    m[CALL] = 1
    return m


class SearchBot:
    def __init__(self, bot, street):
        self.bot, self.street = bot, street

    def new_hand(self, rng):
        self.bot.new_hand(rng)

    def act(self, g, rng):
        from search import replay_decisions, resolve
        if g.street < self.street:
            return self.bot.act(g, rng)
        probs, _ = resolve(g, replay_decisions(g), self.bot, rng=rng)
        return to_engine(g, sample_action(probs, rng))


def iterate_bot(nets):
    adv = [MLP.from_arrays(nets, f"adv{i}_") for i in (0, 1)]
    return PolicyBot(lambda g: regret_matching(adv[g.to_act].predict(features(g, g.to_act)[None])[0], legal_mask(g)))


_cache = {}


def make_bot(spec, fresh=False):
    if isinstance(spec, dict):
        return iterate_bot(spec)
    if fresh or spec not in _cache:
        if spec.startswith(("river:", "turn:")):
            kind, path = spec.split(":", 1)
            bot = SearchBot(make_bot(path, fresh=True), 3 if kind == "river" else 2)
            if not fresh:
                _cache[spec] = bot
            return bot
        bot = {"v1": V1Bot, "random": lambda: PolicyBot(_random), "station": lambda: PolicyBot(_station)}.get(
            spec, lambda: BotPolicy(spec))()
        if fresh:
            return bot
        _cache[spec] = bot
    return _cache[spec]


def duel(a, b, hands, stack=200, seed=0):
    rng = random.Random(seed)
    total = 0
    for i in range(hands // 2):
        for seat_a in (0, 1):
            g = GameState([stack, stack], i % 2, random.Random(seed * 1_000_003 + i))
            g.start_hand(1, 2)
            bots = [a, b] if seat_a == 0 else [b, a]
            for bot in bots:
                bot.new_hand(rng)
            while not g.hand_over:
                g.apply_action(*bots[g.to_act].act(g, rng))
            total += g.net[seat_a]
    return 100 * total / 2 / max(2 * (hands // 2), 1)


def _job(args):
    spec_a, spec_b, hands, stack, seed = args
    return duel(make_bot(spec_a), make_bot(spec_b, fresh=spec_a == spec_b), hands, stack, seed)


def make_pool(workers=None, init=None, initargs=()):
    saved = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    os.environ.update({k: "1" for k in saved})
    pool = Pool(workers or max(1, os.cpu_count() - 2), init, initargs)
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k)
        else:
            os.environ[k] = v
    return pool


def match(pool, a, b, hands, stack, seed=0):
    chunk = max(50, min(500, hands // 20) // 2 * 2)
    jobs = [(a, b, chunk, stack, seed * 7919 + k) for k in range(max(2, hands // chunk))]
    res = np.array(pool.map(_job, jobs))
    return float(res.mean()), float(res.std(ddof=1) / np.sqrt(len(res)))


def match_depths(pool, a, b, hands, stacks=(20, 60, 200), seed=0):
    return {s: match(pool, a, b, hands, s, seed=seed + s) for s in stacks}


def fmt(results):
    return ", ".join(f"{s // 2}bb {m:+.1f} +/- {se:.1f}" for s, (m, se) in results.items())


def main():
    ap = argparse.ArgumentParser(description="Head to head in bb/100 for bot A, with standard errors. "
                                             "A bot is a model file, river:<model> or turn:<model> to add live search, or one of: v1, random, station.")
    ap.add_argument("a")
    ap.add_argument("b", nargs="+")
    ap.add_argument("--hands", type=int, default=20_000, help="duplicate hands per stack depth")
    ap.add_argument("--stacks", default="20,60,200", help="stack depths in chips (blinds 1/2)")
    ap.add_argument("--workers", type=int)
    args = ap.parse_args()
    stacks = tuple(int(s) for s in args.stacks.split(","))
    with make_pool(args.workers) as pool:
        for b in args.b:
            t0 = time.time()
            r = match_depths(pool, args.a, b, args.hands, stacks)
            avg = np.mean([m for m, _ in r.values()])
            print(f"{args.a} vs {b}: {fmt(r)}  (average {avg:+.1f} bb/100, {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
