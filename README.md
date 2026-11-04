# AI Poker Bot — Heads-up No-Limit Texas Hold'em

A Hold'em bot that learns from self-play with Deep CFR (the algorithm family behind Libratus and Pluribus),
with optional real-time re-solving of the turn and river. Only dependency: `numpy`.

## Run it

| Step | Command | What you get |
|---|---|---|
| 1. Train | `python train_deep_cfr.py` | dashboard at http://127.0.0.1:8766: the bot playing itself + the live neural network |
| 2. Play | `python play.py` | http://127.0.0.1:8767: you vs the bot, its cards and reasoning revealed after each hand |
| 3. Explore | `python view_network.py` | http://127.0.0.1:8768: the trained networks, fully interactive |
| Showcase | open `showcase/index.html` | a one-page tour of the project; its river section runs the trained network in the browser |

The shipped bot is `models/v2_holdem.npz` and `play.py` loads it every time. It took 1000 iterations
(`python train_deep_cfr.py --fresh --iters 1000 --memory 2000000`, ~6 h on 12 cores). The default 60 iterations
take ~15 min. Running `train_deep_cfr.py` again continues from the saved bot (`--fresh` starts over); self-play runs
on all but 4 cores (`--workers`). To stop early and still get a finished bot, create `models/v2_holdem.stop`.

`play.py --search turn|river` turns on real-time search from that street (~5 s per decision on the turn). It is off
by default: search guesses the opponent's cards from what the bot itself would have done, so it beats a copy of the
bot but loses to anyone who plays differently (see below).

The training page has a **Reset stacks after every hand** checkbox. Untick it (or start with `--finite`) and chips carry over
between self-play hands, a broke bot buys back in, and going broke counts as an extra loss (`--bust-penalty`), so the
bot learns to protect its stack. Ticked, every hand deals fresh stacks of 10-250 big blinds (`--stacks`).

While you play, the bot profiles you (how often you fold to bets, how aggressive you are, how often your river
bets are bluffs) and shifts its strategy to exploit it, capped so it never drifts more than 30% from its own strategy.
The profile is saved in `models/opponent_profile.json`; the play page has a switch to turn adapting off and a "Forget me" button.

## How strong is it

v2 is the snapshot ensemble (SD-CFR): one saved advantage net per iteration, one picked per hand, weighted by iteration.
Training also builds an average-strategy net and keeps the latest iteration, then keeps whichever candidate does
best at its *worst* stack depth against v1.

| Test (bb/100, ± 1 standard error) | v1 | v2 |
|---|---|---|
| v2 vs v1, 20,000 duplicate hands per depth | — | 10bb +15.9 ± 2.9 · 30bb +1.5 ± 6.1 · 100bb +0.9 ± 13.7 |
| Local best response wins, 3000 hands at 100bb (lower = harder to exploit) | -104 ± 55 | +73 ± 77 |
| vs [Slumbot](https://slumbot.com), 1000 hands at 200bb | +0.7 ± 77.2 | +20.3 ± 67.2 |
| Turn search vs the same bot without search, 4000 duplicate hands at 100bb | — | +106.7 ± 27.9 |
| v2 + turn search vs v1, same | — | -70.5 ± 32.7 |
| v2 + river search vs v1, same | — | -34.3 ± 30.3 |
| v2 + turn search vs Slumbot, 1980 hands, all-in luck removed | — | -169 ± 51 |

Search wins only against the network it takes its range estimates from. Making it safe against other opponents
(e.g. safe subgame solving) is open work.

Of the candidates, the latest iteration alone beat v1 by +59 at 10bb, but local best response won +728 ± 73 from it,
so it is not shipped. The LBR here is simple (it calls preflop) and even loses to v1, so treat it as a leak detector,
not an exploitability bound.

```
python evaluate.py models/v2_holdem.npz v1 --hands 20000          # head to head at 10/30/100bb
python evaluate.py turn:models/v2_holdem.npz v1                   # does search help (also river:, turn:v1)
python lbr.py models/v2_holdem.npz v1 --hands 3000                # local best response
python slumbot.py models/v2_holdem.npz --hands 1000               # vs slumbot.com; logs every hand, reruns add to it
```

## Self-checks

`python test_cards.py` · `python test_engine.py` · `python abstraction.py` · `python nn.py` · `python search.py` · `python opponent.py` · `python legacy.py`

## How it's built

| Piece | File |
|---|---|
| Hand evaluator + no-limit engine | `cards.py`, `engine.py` |
| Card & bet abstraction (bets 1/3, 1/2, 3/4, 1x, 2x pot, all-in; 32 features) | `abstraction.py` |
| Neural net + Deep CFR + SD-CFR, parallel self-play | `nn.py`, `deep_cfr.py`, `train_deep_cfr.py` |
| Real-time turn/river search | `search.py` |
| Learning your play style | `opponent.py` |
| Evaluation: head to head, LBR, Slumbot | `evaluate.py`, `lbr.py`, `slumbot.py` |
| Frozen v1 bot, the baseline | `legacy.py`, `models/v1_holdem.npz` |
| Web UI | `webui.py`, `play.py`, `view_network.py`, `web/` |
