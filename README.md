# AI Poker Bot — Heads-up No-Limit Texas Hold'em

A Hold'em bot that learns from self-play with Deep CFR (the algorithm family behind Libratus and Pluribus).
Only dependency: `numpy`.

## Run it

| Step | Command | What you get |
|---|---|---|
| 1. Train | `python train_deep_cfr.py` | dashboard at http://127.0.0.1:8766: the bot playing itself + the live neural network (~25 min) |
| 2. Play | `python play.py` | http://127.0.0.1:8767: you vs the bot, its cards and reasoning revealed after each hand |
| 3. Explore | `python view_network.py` | http://127.0.0.1:8768: the trained networks, fully interactive |
| Showcase | open `showcase/index.html` | a one-page tour of the project; its river section runs the trained network in the browser |

Train once: the bot is saved to `models/deepcfr_holdem.npz` and `play.py` loads it every time. Running
`train_deep_cfr.py` again continues from the saved bot and makes it stronger (`--fresh` starts over).
Ctrl+C during training still saves a playable bot.

The training page has a **Reset stacks after every hand** checkbox. Untick it (or start with `--finite`) and chips carry over
between self-play hands, a broke bot buys back in, and going broke counts as an extra loss (`--bust-penalty`), so the
bot learns to protect its stack. Ticked, every hand deals fresh stacks of 10-250 big blinds (`--stacks`).

While you play, the bot profiles you (how often you fold to bets, how aggressive you are, how often your river
bets are bluffs) and shifts its strategy to exploit it. The profile is saved in `models/opponent_profile.json`;
the play page has a switch to turn adapting off and a "Forget me" button.

## Self-checks

`python test_cards.py` · `python test_engine.py` · `python abstraction.py` · `python nn.py` · `python search.py` · `python opponent.py`

## How it's built

| Piece | File |
|---|---|
| Hand evaluator + no-limit engine | `cards.py`, `engine.py` |
| Card & bet abstraction | `abstraction.py` |
| Neural net + Deep CFR | `nn.py`, `deep_cfr.py`, `train_deep_cfr.py` |
| Real-time river search | `search.py` |
| Learning your play style | `opponent.py` |
| Web UI | `webui.py`, `play.py`, `view_network.py`, `web/` |
