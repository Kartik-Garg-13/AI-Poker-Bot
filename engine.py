import random
from cards import make_deck, evaluate_fast, card_str

STREETS = ["preflop", "flop", "turn", "river"]


class GameState:
    def __init__(self, stacks, button, rng=random):
        assert len(stacks) == 2, "heads-up only: to_act/button flip with 1 - p"
        self.start_stacks = list(stacks)
        self.stacks = list(stacks)
        self.button = button
        self.board = []
        self.contributed = [0, 0]
        self.street_bet = [0, 0]
        self.folded = [False, False]
        self.all_in = [False, False]
        self.hole = [[], []]
        self.street = 0
        self.to_act = None
        self.acted = set()
        self.bb = 0
        self.last_raise = 0
        self.num_raises = 0
        self.history = []
        self.hand_over = False
        self.net = None
        self.deck = make_deck()
        rng.shuffle(self.deck)

    def clone(self):
        g = object.__new__(GameState)
        g.__dict__ = self.__dict__.copy()
        for k in ("stacks", "contributed", "street_bet", "folded", "all_in", "board", "deck", "history"):
            setattr(g, k, list(getattr(self, k)))
        g.acted = set(self.acted)
        return g

    def post(self, player, amount):
        amount = min(amount, self.stacks[player])
        self.stacks[player] -= amount
        self.contributed[player] += amount
        self.street_bet[player] += amount
        if self.stacks[player] == 0:
            self.all_in[player] = True

    def start_hand(self, sb=1, bb=2):
        for _ in range(2):
            for i in range(2):
                self.hole[i].append(self.deck.pop())
        self.bb = bb
        self.last_raise = bb
        self.post(self.button, sb)
        self.post(1 - self.button, bb)
        self.to_act = self.button
        if self.street_over():
            self._end_street()

    def owed(self, p=None):
        p = self.to_act if p is None else p
        return max(self.street_bet) - self.street_bet[p]

    def pot(self):
        return sum(self.contributed)

    def legal_actions(self):
        p = self.to_act
        owed = self.owed(p)
        actions = ["fold", "call"] if owed > 0 else ["check"]
        if self.stacks[p] > owed and not self.all_in[1 - p]:
            actions.append("raise")
        return actions

    def min_raise_to(self):
        return min(max(self.street_bet) + self.last_raise, self.max_raise_to())

    def max_raise_to(self):
        return self.street_bet[self.to_act] + self.stacks[self.to_act]

    def street_over(self):
        active = [i for i in range(2) if not self.folded[i] and not self.all_in[i]]
        current = max(self.street_bet)
        matched = all(self.street_bet[i] == current for i in active)
        if len(active) < 2:
            return matched
        return matched and all(i in self.acted for i in active)

    def apply_action(self, action, amount=None):
        assert not self.hand_over, "hand is over"
        assert action in self.legal_actions(), f"{action} not legal: {self.legal_actions()}"
        p = self.to_act
        current = max(self.street_bet)
        if action == "fold":
            self.folded[p] = True
        elif action in ("check", "call"):
            self.post(p, current - self.street_bet[p])
            self.acted.add(p)
        else:
            assert amount is not None and self.min_raise_to() <= amount <= self.max_raise_to(), \
                f"raise-to {amount} outside [{self.min_raise_to()}, {self.max_raise_to()}]"
            self.last_raise = max(self.last_raise, amount - current)
            self.post(p, amount - self.street_bet[p])
            self.num_raises += 1
            self.acted = {p}
        self.history.append((self.street, p, action, amount if action == "raise" else None))

        if self.folded[p]:
            self._award([1 - p])
        elif self.street_over():
            self._end_street()
        else:
            self.to_act = 1 - p

    def _end_street(self):
        can_act = sum(1 for i in range(2) if not self.folded[i] and not self.all_in[i])
        if self.street == 3 or can_act < 2:
            while self.street < 3:
                self.next_street()
            self._showdown()
        else:
            self.next_street()

    def next_street(self):
        self.street += 1
        for _ in range(3 if self.street == 1 else 1):
            self.board.append(self.deck.pop())
        self.street_bet = [0, 0]
        self.acted = set()
        self.last_raise = self.bb
        self.num_raises = 0
        self.to_act = 1 - self.button

    def _showdown(self):
        s = [evaluate_fast(self.hole[i] + self.board) for i in range(2)]
        self.scores = s
        best = max(s)
        self._award([i for i in range(2) if s[i] == best])

    def _award(self, winners):
        matched = min(self.contributed)
        payout = [c - matched for c in self.contributed]
        pot = 2 * matched
        for w in winners:
            payout[w] += pot // len(winners)
        for i in range(2):
            self.stacks[i] += payout[i]
        self.net = [self.stacks[i] - self.start_stacks[i] for i in range(2)]
        self.hand_over = True
        self.winners = winners

    def __repr__(self):
        b = " ".join(card_str(c) for c in self.board) or "-"
        return (f"<{STREETS[self.street]} board={b} pot={self.pot()} stacks={self.stacks} "
                f"bets={self.street_bet} to_act={self.to_act}>")


def random_policy(g, rng=random):
    a = rng.choice(g.legal_actions())
    return (a, rng.randint(g.min_raise_to(), g.max_raise_to())) if a == "raise" else (a, None)


def play_hand(policies, stacks=(200, 200), button=0, rng=random):
    g = GameState(stacks, button, rng)
    g.start_hand()
    while not g.hand_over:
        action, amount = policies[g.to_act](g)
        g.apply_action(action, amount)
    return g
