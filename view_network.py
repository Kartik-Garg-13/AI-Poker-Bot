import argparse
import sys
import time
from pathlib import Path

import numpy as np

from deep_cfr import network_view
from nn import MLP
from webui import serve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/deepcfr_holdem.npz")
    ap.add_argument("--port", type=int, default=8768)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    if not Path(args.model).exists():
        sys.exit(f"No trained network at {args.model}. Train one first:  python train_deep_cfr.py")

    data = dict(np.load(args.model))
    iters = int(data["meta"][3])
    nets = []
    if "strat_W0" in data:
        nets.append(network_view(MLP.from_arrays(data, "strat_"), f"Average-strategy net (the one that plays) - {iters} iterations", "strategy"))
    for p in (0, 1):
        nets.append(network_view(MLP.from_arrays(data, f"adv{p}_"), f"Advantage net, player {p} - {iters} iterations"))

    serve({("GET", "/api/nets"): lambda _: {"nets": nets}}, args.port, "network.html", not args.no_browser)
    print("Ctrl+C to quit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
