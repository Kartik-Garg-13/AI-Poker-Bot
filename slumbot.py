import argparse
import itertools
import json
import random
import re
import time
import urllib.request
from pathlib import Path

import numpy as np

from abstraction import evaluate_batch
from cards import ranks, suits
from engine import GameState
from evaluate import make_bot, make_pool

API = "https://slumbot.com/api/"
SCALE, STACK = 50, 20_000
BOARD_AT = [0, 3, 4, 5]


def post(endpoint, data, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(API + endpoint, json.dumps(data).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                out = json.loads(r.read())
            break
        except (OSError, ValueError):
            if k == tries - 1:
                raise
            time.sleep(2 ** k)
    if "error_msg" in out:
        raise RuntimeError(out["error_msg"])
    return out


def card(s):
    return ranks.index(s[0].upper()) * 4 + suits.index(s[1].upper())


def parse(action):
    moves, contrib, bet, last, pos = [], [100, 50], [100, 50], 100, 1
    for k, street in enumerate(action.split("/")):
        if k:
            bet, last, pos = [0, 0], 100, 0
        i = 0
        while i < len(street):
            c, to = street[i], None
            i += 1
            if c == "b":
                j = i
                while i < len(street) and street[i].isdigit():
                    i += 1
                to = int(street[j:i])
                last = max(last, to - max(bet))
                contrib[pos] += to - bet[pos]
                bet[pos] = to
            elif c == "c":
                d = max(bet) - bet[pos]
                contrib[pos] += d
                bet[pos] += d
            moves.append((c, to, contrib[pos] == STACK))
            pos = 1 - pos
    return moves, contrib, bet, last, pos


def rebuild(moves, hole, board, me, rng):
    known = set(hole) | set(board)
    rest = [c for c in range(52) if c not in known]
    rng.shuffle(rest)
    holes = [None, None]
    holes[me], holes[1 - me] = hole, rest[:2]
    dealt = [holes[0][0], holes[1][0], holes[0][1], holes[1][1], *board]
    g = GameState([STACK // SCALE] * 2, 1, rng)
    g.deck = rest[2:] + dealt[::-1]
    g.start_hand(1, 2)
    for c, to, allin in moves:
        acts = g.legal_actions()
        if c == "f":
            g.apply_action("fold")
        elif c in "kc":
            g.apply_action("call" if "call" in acts else "check")
        elif allin:
            g.apply_action("raise", g.max_raise_to())
        else:
            g.apply_action("raise", min(max(round(to / SCALE), g.min_raise_to()), g.max_raise_to()))
    return g


def decide(r, bot, rng):
    me = r["client_pos"]
    moves, contrib, bet, last, pos = parse(r["action"])
    g = rebuild(moves, [card(c) for c in r["hole_cards"]], [card(c) for c in r.get("board", [])], me, rng)
    assert pos == me and g.to_act == me and not g.hand_over, f"out of sync on {r['action']!r}"
    action, amount = bot.act(g, rng)
    owed = max(bet) - bet[me]
    if action == "fold":
        return "f" if owed > 0 else "k"
    if action in ("check", "call"):
        return "c" if owed > 0 else "k"
    hi = bet[me] + STACK - contrib[me]
    if amount == g.max_raise_to():
        return f"b{hi}"
    lo = min(max(bet) + last, hi)
    return f"b{min(max(amount * SCALE, lo), hi)}"


def allin_equity(hole, opp, board, rng, samples=20_000):
    rest = [c for c in range(52) if c not in {*hole, *opp, *board}]
    need = 5 - len(board)
    runs = list(itertools.combinations(rest, need)) if need <= 2 else [rng.sample(rest, need) for _ in range(samples)]
    full = [list(board) + list(run) for run in runs]
    s = evaluate_batch(np.array([list(hole) + f for f in full] + [list(opp) + f for f in full]))
    n = len(full)
    return float(np.mean((s[:n] > s[n:]) + 0.5 * (s[:n] == s[n:])))


def luck_adjusted(r, rng):
    won = r["winnings"] / 100
    if not r.get("bot_hole_cards") or parse(r["action"])[1] != [STACK, STACK]:
        return won
    k = max(i for i, s in enumerate(r["action"].split("/")) if s)
    eq = allin_equity([card(c) for c in r["hole_cards"]], [card(c) for c in r["bot_hole_cards"]],
                      [card(c) for c in r["board"][:BOARD_AT[k]]], rng)
    return (2 * eq - 1) * STACK / 100


_bot = _token = None


def _init(spec):
    global _bot
    _bot = make_bot(spec, fresh=True)


def _hands(job):
    global _token
    n, seed = job
    rng, out = random.Random(seed), []
    for _ in range(n):
        try:
            r = post("new_hand", {"token": _token} if _token else {})
            _token = r.get("token", _token)
            _bot.new_hand(rng)
            while "winnings" not in r:
                r = post("act", {"token": _token, "incr": decide(r, _bot, rng)})
                _token = r.get("token", _token)
        except (OSError, ValueError, RuntimeError, AssertionError) as e:
            out.append({"error": str(e)})
            continue
        out.append({"action": r["action"], "hole": r["hole_cards"], "board": r.get("board", []),
                    "slumbot": r.get("bot_hole_cards"), "bb": r["winnings"] / 100, "adj": luck_adjusted(r, rng)})
    return out


def report(rows, errors, t0):
    n = len(rows)
    if n < 2:
        return
    stat = lambda xs: (100 * np.mean(xs), 100 * np.std(xs, ddof=1) / np.sqrt(n))
    (m, se), (am, ase) = stat([d["bb"] for d in rows]), stat([d["adj"] for d in rows])
    print(f"{n} hands vs Slumbot: {m:+.1f} +/- {se:.1f} bb/100, all-in luck removed {am:+.1f} +/- {ase:.1f}"
          f" ({errors} dropped, {time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Play the bot against Slumbot (slumbot.com), 200bb deep, and report bb/100. "
                                             "Every hand goes to a log, and reruns add to it.")
    ap.add_argument("bot", help="model file, v1, or turn:/river:<model> for live search")
    ap.add_argument("--hands", type=int, default=1000, help="new hands to play this run")
    ap.add_argument("--sessions", type=int, default=4, help="games played at the same time")
    ap.add_argument("--log", help="JSONL hand log (default models/slumbot_<bot>.jsonl)")
    args = ap.parse_args()
    log = Path(args.log or f"models/slumbot_{re.sub(r'[^A-Za-z0-9]+', '_', args.bot)}.jsonl")
    seen = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    rows, errors, t0 = [d for d in seen if "bb" in d], sum("error" in d for d in seen), time.time()
    chunk = 20
    jobs = [(chunk, random.randrange(1 << 60)) for _ in range(-(-args.hands // chunk))]
    with make_pool(args.sessions, _init, (args.bot,)) as pool, open(log, "a") as f:
        for recs in pool.imap_unordered(_hands, jobs):
            f.writelines(json.dumps(d) + "\n" for d in recs)
            f.flush()
            rows += [d for d in recs if "bb" in d]
            errors += sum("error" in d for d in recs)
            report(rows, errors, t0)
    if not jobs:
        report(rows, errors, t0)


if __name__ == "__main__":
    main()
