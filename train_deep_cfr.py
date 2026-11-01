import argparse
import random
import time
from pathlib import Path

import numpy as np

from abstraction import equity, features, legal_mask, normalize_policy
from deep_cfr import (DeepCFR, calling_station, network_view, play_match, random_bot, record_hand,
                      BotPolicy)
from nn import MLP
from webui import Dashboard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=60, help="CFR iterations to run (on top of any saved bot)")
    ap.add_argument("--fresh", action="store_true", help="ignore the saved bot and start from scratch")
    ap.add_argument("--traversals", type=int, default=300, help="sampled hands per player per iteration")
    ap.add_argument("--train-steps", type=int, default=300, help="SGD steps per advantage net per iteration")
    ap.add_argument("--strategy-steps", type=int, default=4000, help="SGD steps for the final average-strategy net")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--hidden", default="64,64,64", help="hidden layer sizes")
    ap.add_argument("--memory", type=int, default=400_000, help="reservoir size per memory")
    ap.add_argument("--eval-hands", type=int, default=400, help="hands vs each baseline per eval")
    ap.add_argument("--stack", type=int, default=200, help="buy-in in chips (blinds 1/2); also the stack used for testing")
    ap.add_argument("--stacks", default="20,500", help="range of stacks dealt in self-play, in chips (20 = 10 big blinds)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--out", default="models/deepcfr_holdem.npz")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dash = Dashboard("Hold'em - Deep CFR self-play", args.port, not args.no_browser)
    dash.set(info=f"Heads-up no-limit; self-play deals stacks of {args.stacks.split(',')[0]}-{args.stacks.split(',')[1]} chips so the bot learns short and deep play. Charts: lower loss = nets fit "
                  f"the regrets; win rate vs baselines should climb as it learns.")
    equity([0, 4], [])
    d = DeepCFR(stack=args.stack, stack_range=tuple(int(v) for v in args.stacks.split(",")), hidden=tuple(int(h) for h in args.hidden.split(",")),
                memory=args.memory, seed=args.seed)
    start, old_strat = 0, None
    if out.exists() and not args.fresh:
        data = dict(np.load(out))
        d.adv = [MLP.from_arrays(data, f"adv{p}_") for p in (0, 1)]
        d.strat = MLP.from_arrays(data, "strat_") if "strat_W0" in data else MLP(d.adv[0].sizes, args.seed + 3)
        old_strat = MLP.from_arrays(data, "strat_") if "strat_W0" in data else None
        d.trained = [True, True]
        start = int(data["meta"][3])
        dash.log(f"Continuing the saved bot from iteration {start}. Use --fresh to start over.")
    end = start + args.iters
    rng = random.Random(args.seed + 99)
    hand_id = [0]
    last_hand = [0.0]

    def maybe_show_hand(probs_fn, force=False):
        if force or time.time() - last_hand[0] > 4:
            last_hand[0] = time.time()
            hand_id[0] += 1
            dash.set(hand=record_hand(probs_fn, hand_id[0], rng=rng, stack=args.stack))

    def show_net(net, name, mode="advantage"):
        dash.set(network=network_view(net, name, mode))

    def on_train(p):
        def cb(step, loss):
            dash.point("Advantage-net loss", f"P{p} net", d.t + step / args.train_steps, loss, log=True)
            show_net(d.adv[p], f"Advantage net P{p} - iteration {d.t}, step {step}/{args.train_steps}")
            dash.set(status=f"iteration {d.t}/{end}: training P{p}'s advantage net "
                            f"(step {step}/{args.train_steps})")
        return cb

    dash.log(f"Deep CFR: {args.iters} iterations x {args.traversals} traversals/player, "
             f"net {[d.adv[0].sizes]}")
    t0 = time.time()
    total_trav = 0
    try:
        for it in range(start + 1, end + 1):
            d.t = it
            for p in (0, 1):
                for k in range(args.traversals):
                    d.traverse(d.new_game(), p)
                    total_trav += 1
                    if k % 20 == 0:
                        dash.set(status=f"iteration {it}/{end}: self-play traversals as P{p} "
                                        f"({k}/{args.traversals})",
                                 progress=(it - start - 1 + (p + k / args.traversals) / 2) / args.iters)
                        maybe_show_hand(d.current_probs)
                d.train(d.adv[p], d.adv_mem[p], args.train_steps, args.batch, every=25, callback=on_train(p))
                d.trained[p] = True
            vs_rand = play_match(d.current_probs, random_bot, args.eval_hands, args.stack, seed=it)
            vs_call = play_match(d.current_probs, calling_station, args.eval_hands, args.stack, seed=it)
            dash.point("Win rate vs baselines (bb/100)", "vs random bot", it, round(vs_rand, 1))
            dash.point("Win rate vs baselines (bb/100)", "vs calling station", it, round(vs_call, 1))
            el = time.time() - t0
            dash.set(stats={"iteration": f"{it}/{end}", "traversals": total_trav,
                            "traversals/sec": f"{total_trav / el:.1f}",
                            "advantage samples": f"{d.adv_mem[0].seen + d.adv_mem[1].seen:,}",
                            "strategy samples": f"{d.strat_mem.seen:,}",
                            "vs random": f"{vs_rand:+.0f} bb/100", "vs calling station": f"{vs_call:+.0f} bb/100",
                            "ETA": f"{el / (it - start) * (end - it) / 60:.0f} min"})
            dash.log(f"iter {it}: vs random {vs_rand:+.0f} bb/100, vs calling station {vs_call:+.0f} bb/100")
            d.save(out, include_strategy=old_strat is not None)
    except KeyboardInterrupt:
        dash.log("Interrupted - training the average-strategy net on what we have (Ctrl+C again to quit)")

    if d.strat_mem.size == 0:
        dash.log("Nothing learned yet - not saving.")
        return
    done = d.t - start
    steps = args.strategy_steps if start == 0 else max(300, round(args.strategy_steps * done / d.t))
    if old_strat is not None:
        n, share = d.strat_mem.size, start * (start + 1) / (d.t * (d.t + 1))
        X, M = d.strat_mem.X[:n], d.strat_mem.M[:n]
        p = np.maximum(old_strat.predict(X), 0) * M
        total = p.sum(1, keepdims=True)
        p = np.where(total > 0, p / np.where(total > 0, total, 1), M / M.sum(1, keepdims=True))
        d.strat_mem.Y[:n] = share * p + (1 - share) * d.strat_mem.Y[:n]
        dash.log(f"Blending in the saved average strategy ({share:.0%} weight for iterations 1-{start}).")
    dash.log(f"Training the average-strategy net ({steps} steps). This is the policy that plays.")

    def on_strat(step, loss):
        dash.point("Average-strategy-net loss", "strategy net", step, loss, log=True)
        show_net(d.strat, f"Average-strategy net - step {step}/{steps}", "strategy")
        dash.set(status=f"final: training average-strategy net ({step}/{steps})", progress=step / steps)
    d.train(d.strat, d.strat_mem, steps, args.batch, every=25, callback=on_strat)

    def score(net):
        probs = lambda g: normalize_policy(net.predict(features(g, g.to_act)[None])[0], legal_mask(g))
        rand = {c: play_match(probs, random_bot, 1000, c, seed=12345) for c in (20, 60, 200)}
        call = {c: play_match(probs, calling_station, 1000, c, seed=12345) for c in (20, 60, 200)}
        return rand[200], call[200], sum(rand.values()) + sum(call.values())

    dash.set(status="final: measuring the bot at 10, 30 and 100 big blinds deep")
    vs_rand, vs_call, total = score(d.strat)
    if old_strat is not None:
        old_rand, old_call, old_total = score(old_strat)
        dash.log(f"Same 6,000 hands at 10/30/100bb: new bot {total:+.0f}, previous bot {old_total:+.0f} "
                 f"(bb/100 summed over random + calling station).")
        if old_total > total:
            d.strat, vs_rand, vs_call = old_strat, old_rand, old_call
            dash.log("The previous average strategy scored higher on the same hands, so it stays.")
    d.save(out)
    bot = BotPolicy(out)
    dash.log(f"Saved {out}. Final average strategy: vs random {vs_rand:+.0f} bb/100, "
             f"vs calling station {vs_call:+.0f} bb/100 (100bb deep)")
    dash.log("Now play it:  python play.py")
    dash.set(status="done - watching the final bot play itself (Ctrl+C to quit)", progress=1,
             stats={"vs random": f"{vs_rand:+.0f} bb/100", "vs calling station": f"{vs_call:+.0f} bb/100",
                    "measured on": "1,000 duplicate hands each, 100bb deep", "iterations trained": d.t,
                    "saved to": str(out)})
    try:
        while True:
            maybe_show_hand(bot.probs, force=True)
            time.sleep(5)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
