import json
import math
from pathlib import Path

import numpy as np

from abstraction import ALLIN, BETS, CALL, FOLD, N_ACTIONS

PROFILE = Path(__file__).parent / "models" / "opponent_profile.json"
SIZED = [a for a in BETS if a != ALLIN]
MAX_SHIFT = 0.3

TRAITS = {
    "fold_to_bet": ("folds when bet into", 0.40, 12),
    "aggression": ("bets or raises when acting", 0.30, 20),
    "vpip": ("plays the hand preflop", 0.65, 15),
    "river_bluff": ("river bets that were bluffs", 0.30, 6),
}


class Opponent:
    def __init__(self, path=PROFILE):
        self.path = Path(path)
        self.c = json.loads(self.path.read_text()) if self.path.exists() else {}

    def _bump(self, trait, hit):
        self.c[f"{trait}_n"] = self.c.get(f"{trait}_n", 0) + 1
        self.c[f"{trait}_hit"] = self.c.get(f"{trait}_hit", 0) + int(hit)

    def rate(self, trait):
        _, prior, k = TRAITS[trait]
        n, hit = self.c.get(f"{trait}_n", 0), self.c.get(f"{trait}_hit", 0)
        return (hit + prior * k) / (n + k), n / (n + k)

    def on_action(self, facing, action):
        if facing:
            self._bump("fold_to_bet", action == "fold")
        if action != "fold":
            self._bump("aggression", action == "raise")

    def end_hand(self, vpip, river_bet_strength):
        self.c["hands"] = self.c.get("hands", 0) + 1
        self._bump("vpip", vpip)
        if river_bet_strength is not None:
            self._bump("river_bluff", river_bet_strength < 0.5)
        self.path.parent.mkdir(exist_ok=True)
        self.path.write_text(json.dumps(self.c))

    def forget(self):
        self.c = {}
        self.path.unlink(missing_ok=True)

    def adjust(self, probs, mask, facing, eq):
        w = np.ones(len(probs))
        notes = []
        fold, fold_conf = self.rate("fold_to_bet")
        if not facing:
            push = (fold - TRAITS["fold_to_bet"][1]) * fold_conf
            if eq < 0.4 and abs(push) > 0.02:
                w[SIZED if push > 0 else BETS] *= math.exp(4 * push)
                notes.append(f"{'bluffing more' if push > 0 else 'bluffing less'}: you fold {fold:.0%} when bet into")
            elif eq > 0.65 and push < -0.02:
                w[BETS] *= math.exp(-4 * push)
                notes.append(f"value-betting more: you only fold {fold:.0%} when bet into")
        else:
            aggr, aggr_conf = self.rate("aggression")
            bluff, bluff_conf = self.rate("river_bluff")
            push = (aggr - TRAITS["aggression"][1]) * aggr_conf + (bluff - TRAITS["river_bluff"][1]) * bluff_conf
            if 0.3 < eq < 0.75 and abs(push) > 0.02:
                w[FOLD] *= math.exp(-4 * push)
                w[CALL] *= math.exp(2 * push)
                notes.append(f"{'calling lighter' if push > 0 else 'folding more'}: you bet or raise {aggr:.0%} of the time")
        w = np.clip(w, 0.25, 4.0)
        p = probs * w * mask
        if p.sum() <= 0:
            return probs, notes
        p = p / p.sum()
        shift = 0.5 * np.abs(p - probs).sum()
        cap = MAX_SHIFT * max(fold_conf, self.rate("aggression")[1])
        if shift > cap:
            p = probs + (p - probs) * cap / shift
        return p, notes

    def summary(self):
        out = []
        for trait, (label, prior, _) in TRAITS.items():
            value, conf = self.rate(trait)
            out.append({"label": label, "value": round(value, 3), "typical": prior,
                        "seen": self.c.get(f"{trait}_n", 0), "confidence": round(conf, 2)})
        return {"hands": self.c.get("hands", 0), "traits": out}


if __name__ == "__main__":
    import tempfile
    probs = np.array([0, 0.7] + [0.3 / len(BETS)] * len(BETS))
    mask = np.ones(N_ACTIONS)
    with tempfile.TemporaryDirectory() as tmp:
        folder = Opponent(Path(tmp) / "a.json")
        station = Opponent(Path(tmp) / "b.json")
        for _ in range(60):
            folder.on_action(True, "fold")
            station.on_action(True, "call")
        bluff_vs_folder = folder.adjust(probs, mask, False, 0.2)[0][BETS].sum()
        bluff_vs_station = station.adjust(probs, mask, False, 0.2)[0][BETS].sum()
        print(f"weak-hand bet frequency: {probs[BETS].sum():.2f} baseline, "
              f"{bluff_vs_folder:.2f} vs a folder, {bluff_vs_station:.2f} vs a calling station")
        assert bluff_vs_folder > probs[BETS].sum() > bluff_vs_station
        shifted = folder.adjust(probs, mask, False, 0.2)[0]
        assert 0.5 * np.abs(shifted - probs).sum() <= MAX_SHIFT + 1e-9, "exploiting must stay near the blueprint"
        assert shifted[ALLIN] <= probs[ALLIN] + 1e-9, "never bluff-shove more"
        folder.end_hand(False, None)
        assert Opponent(Path(tmp) / "a.json").c["fold_to_bet_n"] == 60, "profile must persist"
    print("opponent self-check passed")
