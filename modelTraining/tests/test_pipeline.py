"""Self-tests for the training system. Run via `python main.py selftest`.

These are the checks that protect the paper:
  1. feature builder: shapes, mask discipline, no NaN, frame invariance
     (rotating the whole world must not change the features)
  2. bivariate NLL matches a hand-computed reference
  3. MC-HPD coverage of a KNOWN Gaussian ~= nominal level
  4. end-to-end overfit: training on synthetic data reduces loss
  5. checkpoint save/load round-trip preserves weights exactly
"""
from __future__ import annotations

import math
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.constants import F_DIM, HIST_STEPS, PRED_STEPS      # noqa: E402
from data.dataset import (SampleIndex, build_sample, build_sample_index,  # noqa: E402
                          split_train_val)
from data.synthetic import make_synthetic_run                    # noqa: E402
from model.losses import compute_losses, gaussian_nll_steps      # noqa: E402
from model.metrics import batch_metrics                          # noqa: E402
from model.network import build_model                            # noqa: E402

DATA_CFG = {"max_agents": 8, "t0_stride": 6, "min_obs": 3, "min_future_valid": 27}
MODEL_CFG = {"d_model": 64, "n_modes": 3, "temporal_layers": 1, "agent_layers": 1,
             "n_heads": 4, "d_ff": 128, "dropout": 0.0, "cumulative_mu": True}


# ------------------------------------------------------------------ helpers

def _tmp_run(tmp: str, seed: int = 7) -> str:
    return make_synthetic_run(os.path.join(tmp, "run"), n_episodes=6, seed=seed)


def _first_sample(run_dir: str):
    idx = build_sample_index([run_dir], DATA_CFG)
    assert len(idx) > 50, f"synthetic index too small: {len(idx)}"
    s = build_sample(idx.files[idx.file_idx[0]], int(idx.t0[0]),
                     int(idx.agent[0]), DATA_CFG)
    return idx, s


# -------------------------------------------------------------------- tests

def test_feature_builder(tmp: str) -> None:
    run = _tmp_run(tmp)
    idx, s = _first_sample(run)
    assert s["x"].shape == (8, HIST_STEPS, F_DIM)
    assert s["y"].shape == (PRED_STEPS, 2)
    assert not np.isnan(s["x"]).any(), "NaN leaked into features"
    assert not np.isnan(s["y"]).any(), "NaN leaked into labels"
    assert s["agent_mask"][0] == 1.0 and s["agent_mask"][1] == 1.0
    # undetected steps must carry zeroed kinematics
    det = s["x"][:, :, 2]
    assert np.all(np.abs(s["x"][:, :, 0][det == 0]) < 1e-9)
    # target slot flag
    assert np.all(s["x"][0, :, 12] == 1.0) and np.all(s["x"][1:, :, 12] == 0.0)
    # target's own last detected step sits at the frame origin
    t_det = np.nonzero(det[0])[0]
    last = t_det[-1]
    assert abs(s["x"][0, last, 0]) < 1e-6 and abs(s["x"][0, last, 1]) < 1e-6, \
        "frame origin is not the target's last observed position"


def test_frame_invariance(tmp: str) -> None:
    """Rotating + translating the whole world must leave features unchanged."""
    run = _tmp_run(tmp)
    idx = build_sample_index([run], DATA_CFG)
    path = idx.files[idx.file_idx[0]]
    t0, agent = int(idx.t0[0]), int(idx.agent[0])
    s0 = build_sample(path, t0, agent, DATA_CFG)

    ang = 0.7
    c, s_ = math.cos(ang), math.sin(ang)
    R = np.array([[c, -s_], [s_, c]], dtype=np.float64)
    shift = np.array([5000.0, -3000.0])

    with np.load(path) as z:
        ep = {k: z[k].copy() for k in z.files}
    for key in ("gt_pos", "obs_pos"):
        shp = ep[key].shape
        ep[key] = ((ep[key].reshape(-1, 2) @ R.T) + shift).reshape(shp).astype(np.float32)
    ep["own_pos"] = ((ep["own_pos"] @ R.T) + shift).astype(np.float32)
    ep["obs_R"] = np.einsum("ij,tnjk,lk->tnil", R, np.nan_to_num(ep["obs_R"]), R).astype(np.float32)
    # restore NaN pattern on rotated obs arrays
    nan_mask = ~ep["obs_detected"].astype(bool)
    ep["obs_pos"][nan_mask] = np.nan
    ep["obs_R"][nan_mask] = np.nan

    rot_path = os.path.join(tmp, "rotated.npz")
    np.savez_compressed(rot_path, **ep)
    s1 = build_sample(rot_path, t0, agent, DATA_CFG)

    err_x = np.max(np.abs(s0["x"] - s1["x"]))
    err_y = np.max(np.abs(s0["y"] - s1["y"]))
    assert err_x < 1e-3, f"features not frame-invariant: max err {err_x}"
    assert err_y < 1e-3, f"labels not frame-invariant: max err {err_y}"


def test_split_no_leak(tmp: str) -> None:
    run = _tmp_run(tmp)
    idx = build_sample_index([run], DATA_CFG)
    tr, va = split_train_val(idx, 0.34, seed=1)
    tr_files = set(tr.file_idx.tolist())
    va_files = set(va.file_idx.tolist())
    assert tr_files.isdisjoint(va_files), "episode leaked across train/val split"
    assert len(tr) + len(va) == len(idx)


def test_nll_reference() -> None:
    """gaussian_nll_steps vs a hand-computed bivariate normal density."""
    torch.manual_seed(0)
    B, K, T = 2, 3, PRED_STEPS
    out = {"mu": torch.randn(B, K, T, 2),
           "log_sigma": torch.randn(B, K, T, 2) * 0.3,
           "rho": torch.rand(B, K, T) * 1.6 - 0.8}
    y = torch.randn(B, T, 2)
    nll = gaussian_nll_steps(y, out)
    b, k, t = 1, 2, 5
    mu = out["mu"][b, k, t].numpy(); sig = np.exp(out["log_sigma"][b, k, t].numpy())
    rho = float(out["rho"][b, k, t]); yy = y[b, t].numpy()
    cov = np.array([[sig[0] ** 2, rho * sig[0] * sig[1]],
                    [rho * sig[0] * sig[1], sig[1] ** 2]])
    d = yy - mu
    ref = 0.5 * d @ np.linalg.inv(cov) @ d + 0.5 * math.log(np.linalg.det(cov)) \
        + math.log(2 * math.pi)
    assert abs(float(nll[b, k, t]) - ref) < 1e-4, \
        f"NLL mismatch: {float(nll[b, k, t])} vs {ref}"


def test_coverage_sanity() -> None:
    """GT drawn from the predicted Gaussian itself -> empirical coverage must
    match the nominal level. This validates the MC-HPD estimator the paper's
    calibration claims rest on."""
    torch.manual_seed(1)
    B, K, T = 4000, 1, PRED_STEPS
    mu = torch.zeros(B, K, T, 2)
    log_sigma = torch.full((B, K, T, 2), math.log(0.5))
    rho = torch.full((B, K, T), 0.3)
    out = {"mu": mu, "log_sigma": log_sigma, "rho": rho,
           "mode_logits": torch.zeros(B, K),
           "type_logits": torch.zeros(B, 7)}
    # sample GT from the same correlated Gaussian
    e1 = torch.randn(B, T); e2 = torch.randn(B, T)
    x1 = 0.5 * e1
    x2 = 0.5 * (0.3 * e1 + math.sqrt(1 - 0.09) * e2)
    y = torch.stack([x1, x2], dim=-1)
    batch = {"y": y, "y_valid": torch.ones(B, T),
             "type_label": torch.zeros(B, dtype=torch.long),
             "meta_intent": torch.zeros(B, dtype=torch.long)}
    m = batch_metrics(out, batch, mc_samples=256)
    for lvl in (50, 70, 90):
        v = m[f"cov{lvl}_15s"]
        assert abs(v - lvl / 100) < 0.06, f"coverage@{lvl} = {v:.3f}, expected ~{lvl/100}"


def test_overfit(tmp: str) -> None:
    """~200 gradient steps on synthetic data must clearly reduce the loss
    (>30%). Uses a dense sample index so the loader yields enough batches."""
    run = _tmp_run(tmp, seed=99)
    cfg = dict(DATA_CFG, t0_stride=3)
    idx = build_sample_index([run], cfg)
    from data.loader import make_loader
    dl = make_loader(idx, cfg, batch_size=64, shuffle=True, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(MODEL_CFG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    steps = 0
    while steps < 200:
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["x"], batch["agent_mask"])
            loss, comps = compute_losses(out, batch, {"w_mode": 1.0, "w_type": 0.2}, 0.0)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(comps["loss"])
            steps += 1
            if steps >= 200:
                break
    first = float(np.mean(losses[:10])); last = float(np.mean(losses[-10:]))
    assert last < first * 0.7, f"loss did not drop: {first:.3f} -> {last:.3f}"


def test_checkpoint_roundtrip(tmp: str) -> None:
    model = build_model(MODEL_CFG)
    path = os.path.join(tmp, "ck.pt")
    torch.save({"model": model.state_dict()}, path)
    model2 = build_model(MODEL_CFG)
    model2.load_state_dict(torch.load(path, weights_only=False)["model"])
    for (n1, p1), (_, p2) in zip(model.state_dict().items(), model2.state_dict().items()):
        assert torch.equal(p1, p2), f"weight mismatch after round-trip: {n1}"


# ------------------------------------------------------------------- runner

def run_all(verbose: bool = True) -> bool:
    tests = [("feature_builder", test_feature_builder, True),
             ("frame_invariance", test_frame_invariance, True),
             ("split_no_leak", test_split_no_leak, True),
             ("nll_reference", test_nll_reference, False),
             ("coverage_sanity", test_coverage_sanity, False),
             ("overfit", test_overfit, True),
             ("checkpoint_roundtrip", test_checkpoint_roundtrip, True)]
    ok = True
    for name, fn, needs_tmp in tests:
        tmp = tempfile.mkdtemp(prefix=f"prism_test_{name}_")
        try:
            fn(tmp) if needs_tmp else fn()
            if verbose:
                print(f"  PASS  {name}")
        except AssertionError as e:
            ok = False
            print(f"  FAIL  {name}: {e}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    print("ALL PASS" if ok else "FAILURES PRESENT")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if run_all() else 1)
