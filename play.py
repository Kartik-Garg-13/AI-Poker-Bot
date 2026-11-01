import argparse
import json
import random
import re
import sys
import threading
import time
from pathlib import Path

from abstraction import (ACTION_NAMES, ALLIN, HALF, N_ACTIONS, POT, describe, equity, legal_mask,
                         nearest_abstract, raise_to, to_engine)
from cards import card_str, category_name
from deep_cfr import BotPolicy, sample_action
from engine import STREETS, GameState
from opponent import Opponent
from search import resolve_river
from webui import serve

HUMAN, BOT = 0, 1
HISTORY = Path(__file__).parent / "models" / "hand_history.jsonl"


def signed(x):
    return f"+{x}" if x > 0 else str(x)


class Session:
    def __init__(self, bot, search):
        self.bot, self.search = bot, search
        self.stack = bot.stack
        self.rng = random.Random()
        self.lock = threading.RLock()
        self.hands, self.net, self.g = 0, 0, None
        self.stacks, self.buyins = [self.stack, self.stack], [1, 1]
        self.all_hands, self.all_net = 0, 0
        if HISTORY.exists():
            for line in HISTORY.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.all_hands += 1
                    self.all_net += json.loads(line)["net"]
        self.opp, self.adapt = Opponent(), True
        self.new_hand()

    def new_hand(self, _=None):
        with self.lock:
            if self.g is not None and not self.g.hand_over:
                return self._view()
            assert self.stacks[HUMAN] > 0, "You are out of chips. Buy in to keep playing."
            if self.stacks[BOT] == 0:
                self.stacks[BOT] = self.stack
                self.buyins[BOT] += 1
            self.g = GameState(self.stacks[:], self.hands % 2, self.rng)
            self.g.start_hand(self.bot.sb, self.bot.bb)
            self.start = {"board": [], "pot": self.g.pot(), "stacks": self._behind(), "bets": self.g.street_bet[:]}
            self.events, self.decisions, self.bot_log, self.moves = [], [], [], []
            self.result = None
            self.vpip = self.river_bet = False
            self._bot_moves()
            return self._view()

    def rebuy(self, _=None):
        with self.lock:
            assert self.g.hand_over and self.stacks[HUMAN] == 0, "You still have chips."
            self.stacks[HUMAN] = self.stack
            self.buyins[HUMAN] += 1
            return self.new_hand()

    def set_adapt(self, body):
        with self.lock:
            self.adapt = bool(body["on"])
            return self._view()

    def forget(self, _=None):
        with self.lock:
            self.opp.forget()
            return self._view()

    def act(self, body):
        with self.lock:
            g = self.g
            assert not g.hand_over and g.to_act == HUMAN, "not your turn"
            action = body["action"]
            amount = int(body["amount"]) if action == "raise" else None
            self._apply(HUMAN, action, amount)
            self._bot_moves()
            return self._view()

    def state(self, _=None):
        with self.lock:
            return self._view()

    def _behind(self):
        return [self.g.start_stacks[i] - self.g.contributed[i] for i in (0, 1)]

    def _apply(self, p, action, amount):
        g = self.g
        snap, seen = g.clone(), len(g.board)
        abstract = nearest_abstract(g, action, amount)
        text = describe(g, action, amount)
        if p == HUMAN:
            self.opp.on_action(g.owed() > 0, action)
            self.vpip |= g.street == 0 and action in ("call", "raise")
            self.river_bet |= g.street == 3 and action == "raise"
        g.apply_action(action, amount)
        self.moves.append([action, amount])
        self.decisions.append((snap, p, abstract))
        self.events.append({"actor": p, "text": text, "board": [card_str(c) for c in g.board[:seen]],
                            "pot": g.pot(), "stacks": self._behind(),
                            "bets": g.street_bet[:] if len(g.board) == seen else [0, 0]})
        if len(g.board) > seen:
            self.events.append({"deal": "All-in: run it out" if g.hand_over else STREETS[g.street].title(),
                                "board": [card_str(c) for c in g.board], "pot": g.pot(),
                                "stacks": self._behind(), "bets": [0, 0]})
        if g.hand_over:
            self._finish()

    def _bot_moves(self):
        g = self.g
        while not g.hand_over and g.to_act == BOT:
            info = None
            if self.search and g.street == 3:
                probs, info = resolve_river(g, self.decisions, self.bot, rng=self.rng)
            else:
                probs = self.bot.probs(g)
            mask = legal_mask(g)
            eq = equity(g.hole[BOT], g.board)
            notes = []
            if self.adapt:
                probs, notes = self.opp.adjust(probs, mask, g.owed() > 0, eq)
            a = sample_action(probs, self.rng)
            action, amount = to_engine(g, a)
            self.bot_log.append({"street": STREETS[g.street], "text": describe(g, action, amount),
                                 "equity": round(eq, 3), "chosen": ACTION_NAMES[a], "adapt": notes,
                                 "probs": {ACTION_NAMES[i]: round(float(probs[i]), 3)
                                           for i in range(N_ACTIONS) if mask[i]},
                                 "search": info})
            self._apply(BOT, action, amount)

    def _finish(self):
        g = self.g
        self.hands += 1
        self.net += g.net[HUMAN]
        self.stacks = g.stacks[:]
        showdown = not any(g.folded)
        self.opp.end_hand(self.vpip, equity(g.hole[HUMAN], g.board) if self.river_bet and showdown else None)
        who = lambda i: "You" if i == HUMAN else "Bot"
        if any(g.folded):
            self.result = f"{who(g.winners[0])} won {g.pot()} chips - {who(1 - g.winners[0])} folded"
        elif len(g.winners) == 2:
            self.result = f"Split pot - both have {category_name(g.scores[0])}"
        else:
            w = g.winners[0]
            self.result = f"{who(w)} won {g.pot()} chips with {category_name(g.scores[w])}"
        if self.stacks[HUMAN] == 0:
            self.result += ". You are out of chips."
        elif self.stacks[BOT] == 0:
            self.result += f". The bot is out of chips and buys back in for {self.stack}."
        self.all_hands += 1
        self.all_net += g.net[HUMAN]
        record = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "button": "you" if g.button == HUMAN else "bot",
                  "you": [card_str(c) for c in g.hole[HUMAN]], "bot": [card_str(c) for c in g.hole[BOT]],
                  "board": [card_str(c) for c in g.board],
                  "actions": [("You " + re.sub(r"^(\w+)s\b", r"\1", e["text"]) if e["actor"] == HUMAN else "Bot " + e["text"])
                              for e in self.events if "actor" in e],
                  "result": self.result, "net": g.net[HUMAN], "stacks_after": g.stacks[:],
                  "start_stacks": g.start_stacks[:], "button_seat": g.button, "moves": self.moves,
                  "deal": {"hole": [list(h) for h in g.hole], "board": list(g.board)},
                  "bot_decisions": [{"street": b["street"], "chose": b["chosen"], "probs": b["probs"]} for b in self.bot_log]}
        HISTORY.parent.mkdir(exist_ok=True)
        with open(HISTORY, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _view(self):
        g = self.g
        legal = None
        if not g.hand_over and g.to_act == HUMAN:
            m = legal_mask(g)
            legal = {"actions": g.legal_actions(), "call": min(g.owed(), g.stacks[HUMAN]),
                     "min": g.min_raise_to(), "max": g.max_raise_to(),
                     "presets": {name: raise_to(g, a) for name, a in
                                 (("1/2 pot", HALF), ("pot", POT), ("all-in", ALLIN)) if m[a]}}
        bb = self.bot.bb
        hand_no = self.hands + (0 if g.hand_over else 1)
        rate = round(100 * self.net / bb / self.hands, 1) if self.hands else 0
        seat = "you are on the button (small blind)" if g.button == HUMAN else "bot is on the button"
        return {
            "title": "Heads-up No-Limit Hold'em vs the bot",
            "status": f"Hand #{hand_no} · blinds 1/2 · {seat}",
            "about": "Revealed when the hand ends, since showing it live would give its hand away. Equity is its own "
                     "estimate of beating a random hand. On the river it re-solves the spot in real time against "
                     "the range of hands your betting suggests.",
            "stats": {"hands played": self.hands, "your result": f"{signed(self.net)} chips",
                      "your win rate": f"{signed(rate)} bb/100", "your stack": f"{self.stacks[HUMAN]} chips",
                      "bot stack": f"{self.stacks[BOT]} chips",
                      "buy-ins": f"you {self.buyins[HUMAN]}, bot {self.buyins[BOT]}",
                      "all-time": f"{self.all_hands} hands, {signed(self.all_net)} chips",
                      "river search": "on" if self.search else "off", "bot trained": f"{self.bot.t} iterations"},
            "can_rebuy": g.hand_over and self.stacks[HUMAN] == 0, "buyin": self.stack,
            "profile": self.opp.summary(), "adapt": self.adapt,
            "button": g.button,
            "hole": [[card_str(c) for c in g.hole[HUMAN]],
                     [card_str(c) for c in g.hole[BOT]] if g.hand_over else None],
            "board": [card_str(c) for c in g.board], "pot": g.pot(),
            "stacks": g.stacks if g.hand_over else self._behind(), "bets": g.street_bet,
            "to_act": None if g.hand_over else g.to_act, "over": g.hand_over, "result": self.result,
            "legal": legal, "events": self.events, "start": self.start,
            "bot_log": self.bot_log if g.hand_over else [{"street": b["street"], "text": b["text"]}
                                                         for b in self.bot_log],
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/deepcfr_holdem.npz")
    ap.add_argument("--no-search", action="store_true", help="turn off the live river re-solve")
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    if not Path(args.model).exists():
        sys.exit("No trained Hold'em bot yet. Train it first (about 25 minutes):  python train_deep_cfr.py")

    bot = BotPolicy(args.model)
    if not bot.has_strat:
        print("This checkpoint has no average-strategy net yet, so the bot uses its last iteration's strategy.")
    session = Session(bot, search=not args.no_search)
    serve({("GET", "/api/state"): session.state, ("POST", "/api/new"): session.new_hand,
           ("POST", "/api/act"): session.act, ("POST", "/api/adapt"): session.set_adapt,
           ("POST", "/api/forget"): session.forget, ("POST", "/api/rebuy"): session.rebuy}, args.port, "play.html", not args.no_browser)
    print(f"Hold'em bot, {bot.t} Deep CFR iterations, river search {'on' if session.search else 'off'}. Ctrl+C to quit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
