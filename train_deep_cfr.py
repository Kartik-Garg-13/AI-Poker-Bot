import argparse
import csv
import json
import os
import random
import shutil
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import numpy as np

from abstraction import equity
from deep_cfr import BotPolicy, DeepCFR, network_view, pack_ensemble, record_hand, worker_init, worker_traverse
from evaluate import fmt, make_pool, match_depths
from nn import MLP
from webui import Dashboard


def nets_of(d):
    return {**d.adv[0].arrays("adv0_"), **d.adv[1].arrays("adv1_")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=60, help="CFR iterations to run (on top of any saved bot)")
    ap.add_argument("--fresh", action="store_true", help="ignore the saved bot and start from scratch")
    ap.add_argument("--traversals", type=int, default=1500, help="sampled hands per player per iteration")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 4), help="processes running self-play")
    ap.add_argument("--train-steps", type=int, default=400, help="SGD steps per advantage net per iteration")
    ap.add_argument("--strategy-steps", type=int, default=8000, help="SGD steps for the average-strategy net")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--hidden", default="128,128,128", help="hidden layer sizes")
    ap.add_argument("--memory", type=int, default=1_000_000, help="reservoir size per memory")
    ap.add_argument("--eval-every", type=int, default=10, help="iterations between checks against the v1 bot")
    ap.add_argument("--eval-hands", type=int, default=2000, help="hands per stack depth in those checks")
    ap.add_argument("--final-hands", type=int, default=20_000, help="hands per stack depth when picking the final bot")
    ap.add_argument("--stack", type=int, default=200, help="buy-in in chips (blinds 1/2)")
    ap.add_argument("--stacks", default="20,500", help="range of stacks dealt in self-play, in chips (20 = 10 big blinds)")
    ap.add_argument("--finite", action="store_true", help="start with chips carried over between hands (the page checkbox can switch it)")
    ap.add_argument("--bust-penalty", type=float, default=0.5, help="extra loss for going broke in finite mode, in buy-ins")
    ap.add_argument("--human-share", type=float, default=0.2, help="share of self-play hands started from spots in hands people played on the play page")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--out", default="models/v2_holdem.npz")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mem_path = out.with_name(out.stem + "_memory.npz")
    hist_path = out.with_name(out.stem + "_history.csv")
    snap_dir = out.parent / "snapshots" / out.stem
    stop_path = out.with_name(out.stem + ".stop")
    human_path = out.with_name("hand_history.jsonl")
    stop_path.unlink(missing_ok=True)

    dash = Dashboard("Hold'em - Deep CFR self-play", args.port, not args.no_browser, options={"reset_stacks": not args.finite})
    lo, hi = (int(v) for v in args.stacks.split(","))
    dash.set(info=f"Heads-up no-limit; self-play deals stacks of {lo}-{hi} chips on {args.workers} processes. "
                  f"Every iteration's networks are kept; the bot that plays samples one of them per hand (SD-CFR). "
                  f"The chart tracks the latest iteration against the frozen v1 bot.")
    equity([0, 4], [])
    hidden = tuple(int(h) for h in args.hidden.split(","))
    d = DeepCFR(stack=args.stack, stack_range=(lo, hi), hidden=hidden, memory=args.memory, seed=args.seed)
    d.bust_penalty, d.human_share = args.bust_penalty, args.human_share
    start, old_strat, restored, previous = 0, None, False, None

    def load_human():
        if args.human_share <= 0 or not human_path.exists():
            return []
        rows = (json.loads(line) for line in human_path.read_text(encoding="utf-8").splitlines() if line.strip())
        return [r for r in rows if "moves" in r]

    if args.fresh:
        hist_path.unlink(missing_ok=True)
        if snap_dir.exists():
            shutil.rmtree(snap_dir)
    if out.exists() and not args.fresh:
        data = dict(np.load(out))
        d.adv = [MLP.from_arrays(data, f"adv{p}_") for p in (0, 1)]
        if "strat_W0" in data:
            old_strat = MLP.from_arrays(data, "strat_")
            d.strat = MLP.from_arrays(data, "strat_")
        else:
            d.strat = MLP(d.adv[0].sizes, args.seed + 3)
        d.trained = [True, True]
        d.t = start = int(data["meta"][3])
        previous = out.with_name(out.stem + "_previous.npz")
        shutil.copy(out, previous)
        dash.log(f"Continuing the saved bot from iteration {start}. Use --fresh to start over.")
        if mem_path.exists():
            mem_t = d.load_memories(mem_path)
            restored = True
            dash.log(f"Loaded {d.adv_mem[0].size + d.adv_mem[1].size + d.strat_mem.size:,} training memories "
                     f"saved at iteration {mem_t}.")
        if hist_path.exists():
            with open(hist_path, newline="") as f:
                for row in csv.DictReader(f):
                    i = int(row["iteration"])
                    if row["vs_v1"]:
                        dash.point("Latest iteration vs v1 bot (bb/100)", "average of 10/30/100bb", i, float(row["vs_v1"]))
                    for p in (0, 1):
                        if row[f"loss_p{p}"]:
                            dash.point("Advantage-net loss", f"P{p} net", i, float(row[f"loss_p{p}"]), log=True)
    end = start + args.iters
    rng = random.Random(args.seed + 99)
    hand_id, last_hand, last_loss = [0], [0.0], [None, None]

    def maybe_show_hand(probs_fn, force=False):
        if force or time.time() - last_hand[0] > 4:
            last_hand[0] = time.time()
            hand_id[0] += 1
            dash.set(hand=record_hand(probs_fn, hand_id[0], rng=rng, stack=args.stack))

    def on_train(p):
        def cb(step, loss):
            last_loss[p] = loss
            dash.point("Advantage-net loss", f"P{p} net", d.t + step / args.train_steps, loss, log=True)
            dash.set(network=network_view(d.adv[p], f"Advantage net P{p} - iteration {d.t}, step {step}/{args.train_steps}"),
                     status=f"iteration {d.t}/{end}: training P{p}'s advantage net (step {step}/{args.train_steps})")
        return cb

    cfg = {"stack": args.stack, "stack_range": (lo, hi), "hidden": hidden, "seed": args.seed, "bust_penalty": args.bust_penalty}
    pool = make_pool(args.workers, worker_init, (cfg,))
    per_job = max(1, -(-args.traversals // (3 * args.workers)))
    dash.log(f"Deep CFR: {args.iters} iterations x {args.traversals} traversals/player on {args.workers} processes, "
             f"net {d.adv[0].sizes}, {args.memory:,} samples per memory")
    t0, total_trav, vs_v1 = time.time(), 0, None
    try:
        for it in range(start + 1, end + 1):
            if stop_path.exists():
                dash.log("Stop file found - finishing up.")
                break
            d.t = it
            hands = load_human()
            if len(hands) != len(d.human_hands):
                dash.log(f"Practising from {len(hands)} hands played against people "
                         f"({args.human_share:.0%} of self-play starts from those spots).")
            d.human_hands = hands
            if d.finite == dash.options["reset_stacks"]:
                d.finite = not dash.options["reset_stacks"]
                dash.log("Chips now carry over between hands; going broke costs extra." if d.finite
                         else "Stacks now reset after every hand.")
            for p in (0, 1):
                job = {"p": p, "t": it, "nets": nets_of(d), "trained": d.trained[:], "finite": d.finite,
                       "human": d.human_hands, "human_share": args.human_share}
                sizes = [per_job] * (args.traversals // per_job) + ([args.traversals % per_job] if args.traversals % per_job else [])
                for k, (adv_rows, strat_rows, session, busts) in enumerate(
                        pool.imap_unordered(worker_traverse, [{**job, "n": n} for n in sizes])):
                    d.adv_mem[p].add_many(adv_rows)
                    d.strat_mem.add_many(strat_rows)
                    d.session, d.busts = session, busts
                    total_trav += sizes[k]
                    dash.set(status=f"iteration {it}/{end}: self-play as P{p} ({k + 1}/{len(sizes)} batches)",
                             progress=(it - start - 1 + (p + (k + 1) / len(sizes)) / 2) / args.iters)
                    maybe_show_hand(d.current_probs)
                d.train(d.adv[p], d.adv_mem[p], args.train_steps, args.batch, every=50, callback=on_train(p))
                d.trained[p] = True
            d.save_snapshot(snap_dir)
            d.save(out, include_strategy=False)
            if it % args.eval_every == 0 or it == end:
                dash.set(status=f"iteration {it}/{end}: playing the latest iteration against the v1 bot")
                r = match_depths(pool, nets_of(d), "v1", args.eval_hands, seed=it)
                vs_v1 = float(np.mean([m for m, _ in r.values()]))
                dash.point("Latest iteration vs v1 bot (bb/100)", "average of 10/30/100bb", it, round(vs_v1, 1))
                dash.log(f"iter {it}: latest iteration vs v1 bot {fmt(r)} bb/100")
            el = time.time() - t0
            dash.set(stats={"iteration": f"{it}/{end}", "traversals": f"{total_trav:,}",
                            "traversals/sec": f"{total_trav / el:.0f}",
                            "advantage samples": f"{d.adv_mem[0].seen + d.adv_mem[1].seen:,}",
                            "strategy samples": f"{d.strat_mem.seen:,}", "hands vs people": len(d.human_hands),
                            "latest vs v1": "-" if vs_v1 is None else f"{vs_v1:+.0f} bb/100",
                            "ETA": f"{el / (it - start) * (end - it) / 60:.0f} min",
                            "stacks": "carried over" if d.finite else "reset after every hand"})
            new_file = not hist_path.exists()
            with open(hist_path, "a", newline="") as f:
                w = csv.writer(f)
                if new_file:
                    w.writerow(["iteration", "time", "vs_v1", "loss_p0", "loss_p1", "stacks", "traversals_per_sec"])
                w.writerow([it, time.strftime("%Y-%m-%d %H:%M:%S"),
                            round(vs_v1, 1) if it % args.eval_every == 0 and vs_v1 is not None else "",
                            f"{last_loss[0]:.5f}" if last_loss[0] is not None else "",
                            f"{last_loss[1]:.5f}" if last_loss[1] is not None else "",
                            "carried over" if d.finite else "reset", round(total_trav / el, 1)])
            if it % 100 == 0:
                dash.set(status=f"iteration {it}/{end}: saving training memories")
                d.save_memories(mem_path)
    except KeyboardInterrupt:
        dash.log("Interrupted - picking the final bot from what we have.")

    if d.strat_mem.size == 0:
        dash.log("Nothing learned yet - not saving.")
        pool.terminate()
        return
    dash.set(status="saving training memories")
    d.save_memories(mem_path)
    first = int(d.strat_mem.T[:d.strat_mem.size].min())
    share = (first - 1) * first / (d.t * (d.t + 1))
    if old_strat is not None and share > 0:
        n = d.strat_mem.size
        X, M = d.strat_mem.X[:n], d.strat_mem.M[:n]
        p = np.maximum(old_strat.predict(X), 0) * M
        total = p.sum(1, keepdims=True)
        p = np.where(total > 0, p / np.where(total > 0, total, 1), M / M.sum(1, keepdims=True))
        d.strat_mem.Y[:n] = share * p + (1 - share) * d.strat_mem.Y[:n]
        dash.log(f"Blending in the saved average strategy ({share:.0%} weight for iterations 1-{first - 1}).")
    steps = args.strategy_steps

    def on_strat(step, loss):
        dash.point("Average-strategy-net loss", "strategy net", step, loss, log=True)
        dash.set(network=network_view(d.strat, f"Average-strategy net - step {step}/{steps}", "strategy"),
                 status=f"final: training the average-strategy net ({step}/{steps})", progress=step / steps)
    d.train(d.strat, d.strat_mem, steps, args.batch, every=100, callback=on_strat)

    meta = [d.stack, d.sb, d.bb, d.t]
    cands = {"snapshot ensemble (SD-CFR)": out.with_name(out.stem + "_ensemble.npz"),
             "average-strategy net": out.with_name(out.stem + "_strategy.npz"),
             "latest iteration": out.with_name(out.stem + "_latest.npz")}
    pack_ensemble(snap_dir, cands["snapshot ensemble (SD-CFR)"], meta)
    d.save(cands["average-strategy net"])
    d.save(cands["latest iteration"], include_strategy=False)
    if previous is not None:
        cands["previous bot"] = previous
    scores = {}
    for name, path in cands.items():
        dash.set(status=f"final: {name} vs the v1 bot, {args.final_hands:,} hands at each of 10/30/100bb")
        r = match_depths(pool, str(path), "v1", args.final_hands, seed=4242)
        scores[name] = (float(min(m for m, _ in r.values())), r)
        dash.log(f"{name} vs v1 bot: {fmt(r)} bb/100")
    best = max(scores, key=lambda k: scores[k][0])
    shutil.copy(cands[best], out)
    for path in cands.values():
        path.unlink(missing_ok=True)
    out.with_name(out.stem + "_selection.json").write_text(json.dumps(
        {"iterations": d.t, "chosen": best, "hands_per_depth": args.final_hands,
         "vs_v1": {k: {f"{s // 2}bb": [round(m, 1), round(se, 1)] for s, (m, se) in v[1].items()} for k, v in scores.items()}},
        indent=2))
    pool.close()
    dash.log(f"Kept the {best} ({scores[best][0]:+.1f} bb/100 vs v1 at its worst depth). Saved {out}.")
    dash.log("Now play it:  python play.py")
    bot = BotPolicy(out)
    dash.set(status="done - watching the final bot play itself (Ctrl+C to quit)", progress=1,
             stats={"chosen": best, "vs v1": f"{scores[best][0]:+.1f} bb/100", "iterations trained": d.t,
                    "saved to": str(out)})
    try:
        while True:
            bot.new_hand(rng)
            maybe_show_hand(bot.probs, force=True)
            time.sleep(5)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
