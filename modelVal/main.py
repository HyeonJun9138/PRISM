"""PRISM modelVal -- the live validation console (the only top-level file).

What this is: a pilot-in-the-loop test bench for trained PRISM predictors.
  * dataGen's simulator runs live (same physics the dataset came from);
  * YOU fly the eVTOL ownship with the keyboard, through traffic that keeps
    coming (scenarios auto-rotate every 10 minutes);
  * a trained model (Logs/Models) predicts every visible track at 2 Hz through
    the SAME feature pipeline it was trained on;
  * left pane: the world. right pane: the doc [6] bird's-eye risk map built
    from the model's K-branch ellipse fans;
  * bottom: live scoreboard -- inference latency, rolling error and coverage
    at 5/10/15 s, separation state.

Usage:
  python main.py                 # console GUI (default), port 8810
  python main.py gui [--port N] [--no-browser]
  python main.py smoke           # headless self-check (~60 s, needs a model)
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bridge  # noqa: E402,F401  (must run before any dataGen/modelTraining import)


def cmd_gui(args) -> None:
    # package is named 'console', NOT 'gui': the bridge puts dataGen ahead of
    # this root on sys.path and dataGen owns a 'gui' package already.
    from console.server import serve
    host = "0.0.0.0" if args.public else None
    serve(port=args.port, open_browser=not args.no_browser, host=host)


def cmd_smoke(args) -> None:
    """Headless end-to-end: session + model + scripted pilot + scoreboard."""
    import time

    from runtime.predictor import LivePredictor, list_models
    from runtime.session import LiveSession

    models = list_models()
    if not models:
        sys.exit("no models under $PRISM_ROOT/Logs/Models - train one first")
    pred = LivePredictor()
    info = pred.load(models[0]["path"])
    print(f"model: {info['path']} ({info['params']:,} params, {info['device']})")

    sess = LiveSession(pred, preset="test_ood", seed=1, speed=8.0)
    sess.paused = False
    t0 = time.time()
    phase = 0
    while time.time() - t0 < 60:
        el = time.time() - t0
        if el > 20 and phase == 0:
            sess.keys.set(0.6, 0.3, False)   # bank right, accelerate
            phase = 1
        if el > 40 and phase == 1:
            sess.keys.set(-0.4, 0.0, True)   # left, brake
            phase = 2
        time.sleep(1.0)
    sess.paused = True
    frames = sess.latest_frames(0)
    assert frames, "no frames produced"
    m = frames[-1]["metrics"]
    print(f"frames={frames[-1]['seq']}  sim_t={frames[-1]['t']}s  "
          f"tracks={m['tracks_predicted']}  latency={m['latency_ms']}ms")
    print(f"scored={m['n_scored']}  err5s={m.get('err_5s')}m  "
          f"err15s={m.get('err_15s')}m  cov90_15s={m.get('cov90_15s')}")
    assert m["n_scored"] > 0, "scoreboard never settled a prediction"
    assert m["latency_ms"] and m["latency_ms"] < 100, "latency implausible"
    own = frames[-1]["own"]
    assert own["v"] >= 29.0, "ownship violated envelope floor"
    sess.stop()
    print("SMOKE OK")


def main() -> None:
    ap = argparse.ArgumentParser(prog="main.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command")
    g = sub.add_parser("gui", help="serve the console (default)")
    g.add_argument("--port", type=int, default=None)
    g.add_argument("--no-browser", action="store_true")
    g.add_argument("--public", action="store_true",
                   help="bind 0.0.0.0 for external demo viewing (no auth -- "
                        "close the firewall port again after the demo)")
    sub.add_parser("smoke", help="headless end-to-end self-check")
    args = ap.parse_args()
    cmd = args.command or "gui"
    if args.command is None:
        args.port = None
        args.no_browser = False
    {"gui": cmd_gui, "smoke": cmd_smoke}[cmd](args)


if __name__ == "__main__":
    main()
