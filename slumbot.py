import argparse
import json
import random
import time
import urllib.request

import numpy as np

from cards import ranks, suits
from engine import GameState
from evaluate import make_bot

API = "https://slumbot.com/api/"
SCALE, STACK = 50, 20_000


def post(endpoint, data):
    req = urllib.request.Request(API + endpoint, json.dumps(data).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        out = json.loads(r.read())
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


def main():
    ap = argparse.ArgumentParser(description="Play the bot against Slumbot (slumbot.com), 200bb deep, and report bb/100.")
    ap.add_argument("bot", help="model file, or v1")
    ap.add_argument("--hands", type=int, default=1000)
    ap.add_argument("--log", help="append each hand's result to this JSONL file")
    args = ap.parse_args()
    bot, rng = make_bot(args.bot), random.Random()
    token, won, t0 = None, [], time.time()
    for h in range(1, args.hands + 1):
        r = post("new_hand", {"token": token} if token else {})
        token = r.get("token", token)
        bot.new_hand(rng)
        while "winnings" not in r:
            r = post("act", {"token": token, "incr": decide(r, bot, rng)})
            token = r.get("token", token)
        won.append(r["winnings"] / 100)
        if args.log:
            with open(args.log, "a") as f:
                f.write(json.dumps({"bot": args.bot, "action": r["action"], "hole": r["hole_cards"],
                                    "board": r.get("board", []), "slumbot": r.get("bot_hole_cards"),
                                    "bb": won[-1]}) + "\n")
        if h % 50 == 0 or h == args.hands:
            se = 100 * np.std(won, ddof=1) / np.sqrt(len(won))
            print(f"{h} hands vs Slumbot: {100 * np.mean(won):+.1f} +/- {se:.1f} bb/100 ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
