"""Generate R3_RESULTS.md from the raw aggregate JSONs (never by hand; D5.8).

    python make_r3_results.py            # writes R3_RESULTS.md next to this file

Inputs: Paper/rebuttal/r3/planner/*.json (experiments.aggregate_r3 output),
Logs/Analysis/eval_r3/*_report.json, Paper/rebuttal/r3/recal_*.json,
Paper/rebuttal/r3/rollout_*.json. Rates are per exposure hour; ratios are
a/b with paired episode-bootstrap 95 % CIs; p is the two-sided exact sign test.
"""
from __future__ import annotations

import os

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PL = f"{HERE}/planner"
EV = (os.environ["PRISM_ROOT"] + '/Logs/Analysis/eval_r3')


def J(p):
    return json.load(open(p, encoding="utf-8"))


def f(x, n=2):
    return "–" if x is None else (f"{x:.{n}f}" if isinstance(x, (int, float)) else str(x))


def ratio(e):
    r = e["rate_ratio"]
    return f"{r['est']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]"


out = []
w = out.append
w("# R3 결과 원장 (자동 생성, 손으로 고치지 않는다)\n")
w("생성: `Paper/rebuttal/r3/make_r3_results.py`. 원자료는 각 절에 적었다. 비율은 a/b, 95 % CI는 paired "
  "에피소드 bootstrap(B = 10,000), p는 양측 정확 부호검정. 플래너 지표는 20 Hz 분리거리, 새 스크린(현재 트랙 "
  "탐지 → 임계 진입 리드) 기준이다. 플래너 모델 = 현재 트랙 K=3 시드 1.\n")

# ------------------------------------------------------------- planner core
w("## 1. 폐루프 핵심 조건 (R1-3, R2-V, R3-M6, P2, P3)\n")
w("| 조건 | 노출 h/arm | 152 m 사건/h none·cv·PRISM | 45 m 미만/h none·cv·PRISM | 45 m 아래 s/h none·cv·PRISM | 최소이격 p5 [m] none·cv·PRISM |")
w("|---|---|---|---|---|---|")
core = {c: J(f"{PL}/core_{c}.json") for c in ("sparse", "ood40", "ood100", "id40", "id24")}
for c, d in core.items():
    A = d["per_arm"]
    ev = lambda a, k: A[a]["events"][k]["per_hour"]
    tb = lambda a: A[a]["time_below"]["union20_tb_45"]["per_hour_s"]
    w(f"| {c} | {A['model']['exposure_hours']:.2f} | "
      + " · ".join(f(ev(a, "union20_152_new")) for a in ("none", "cv", "model")) + " | "
      + " · ".join(f(ev(a, "union20_severe_new")) for a in ("none", "cv", "model")) + " | "
      + " · ".join(f(tb(a), 1) for a in ("none", "cv", "model")) + " | "
      + " · ".join(f(A[a]["min_sep_20hz"]["p5"], 1) for a in ("none", "cv", "model")) + " |")
w("")
w("| 조건 | 비교 | 152 m 비 [CI], p | 76 m 비 [CI], p | 45 m 미만 비 [CI], p | 45 m 아래 시간 W/L, p | 최소이격 W/L/T, p |")
w("|---|---|---|---|---|---|---|")
for c, d in core.items():
    for pr in ("cv|model", "none|model", "none|cv"):
        P = d["paired"][pr]
        e1, e2, e3 = (P["events"][k] for k in ("union20_152_new", "union20_76_new", "union20_severe_new"))
        t = P["time_below"]["union20_tb_45"]; m = P["min_sep_20hz"]
        w(f"| {c} | {pr} | {ratio(e1)}, {e1['sign_p']:.3f} | {ratio(e2)}, {e2['sign_p']:.3f} | {ratio(e3)}, {e3['sign_p']:.3f} | "
          f"{t['wins']}/{t['losses']}, {t['sign_p']:.3f} | {m['wins']}/{m['losses']}/{m['ties']}, {m['sign_p']:.4f} |")
w("\n비율 a/b > 1이면 b(오른쪽 arm)의 사건이 적다. W는 a가 나은 에피소드 수.\n")

# ------------------------------------------------------------------ P1 strata
w("## 2. P1 층화: CV가 구조적으로 틀리는 조우 (cv|model, pairwise 152 m, 새 스크린)\n")
w("| 조건 | 층 | cv 사건 | PRISM 사건 | 차이(/h) cv − PRISM [CI] |")
w("|---|---|---|---|---|")
for c in ("sparse", "ood40", "ood100", "id40", "id24"):
    S = core[c]["paired"]["cv|model"]["strata"]["pairwise_152"]
    for k in ("class=HARD", "class=EASY", "turning_at_track_start=yes", "turning_at_track_start=no",
              "reaction_active=yes", "reaction_active=no"):
        if k in S:
            s = S[k]["new"]; rd = s["rate_diff"]
            w(f"| {c} | {k} | {s['total_a']} | {s['total_b']} | {rd['est']:+.2f} [{rd['ci'][0]:+.2f}, {rd['ci'][1]:+.2f}] |")
w("")
w("같은 층 구분의 45 m 미만(pairwise_45):\n")
w("| 조건 | 층 | cv | PRISM | 차이(/h) [CI] |")
w("|---|---|---|---|---|")
for c in ("sparse", "ood40", "ood100", "id40", "id24"):
    S = core[c]["paired"]["cv|model"]["strata"].get("pairwise_45", {})
    for k in ("class=HARD", "class=EASY", "turning_at_track_start=yes", "turning_at_track_start=no"):
        if k in S:
            s = S[k]["new"]; rd = s["rate_diff"]
            w(f"| {c} | {k} | {s['total_a']} | {s['total_b']} | {rd['est']:+.2f} [{rd['ci'][0]:+.2f}, {rd['ci'][1]:+.2f}] |")
w("")

# -------------------------------------------------------------- P4 coverage
w("## 3. P4 폐루프 예측 보정 (R3-m5): HPD 커버리지 50/70/90 %\n")
w("층: interv = 예측 지평 안 ownship 기동(측방 변화 > 30 m 또는 속도 변화 > 0.1·v_des), react = 침입기 반응 층 "
  "latched 또는 REACTIVE_AVOID 의도.\n")
w("| 조건 | arm | 층 | n(15 s) | 5 s | 10 s | 15 s |")
w("|---|---|---|---|---|---|---|")
for c in ("ood40", "id40"):
    for arm in ("cv", "model"):
        C = core[c]["coverage"][arm]["cells"]
        for cell in ("all", "interv0_react0", "interv0_react1", "interv1_react0", "interv1_react1"):
            x = C[cell]
            w(f"| {c} | {arm} | {cell} | {x['15s']['n']} | " + " | ".join(
                f"{x[h]['cov50']:.3f}/{x[h]['cov70']:.3f}/{x[h]['cov90']:.3f}" for h in ("5s", "10s", "15s")) + " |")
w("")

# -------------------------------------------------------------- M4 planner
w("## 4. R3-M4 플래너 결정 규칙: feasible 비율과 비상 모드\n")
w("| 조건 | arm | feasible 평균 | 비상 비율 | 충돌(정적) | 충돌(예측) | 경계만 | 경계+동역학 | 동역학만 | 침투 깊이 평균/최대 [m] |")
w("|---|---|---|---|---|---|---|---|---|---|")
for c in ("ood40", "id40"):
    for arm in ("none", "cv", "model"):
        pl = core[c]["per_arm"][arm]["planner"]; ca = pl["causes"]
        w(f"| {c} | {arm} | {pl['feasible_frac_mean']:.3f} | {pl['emergency_frac']:.4f} | {ca['emerg_collision_static']} | "
          f"{ca['emerg_collision_forecast']} | {ca['emerg_boundary_only']} | {ca['emerg_boundary_dynamic']} | {ca['emerg_dynamic_only']} | "
          f"{f(pl['pen_depth_mean_m'], 1)}/{f(pl['pen_depth_max_m'], 1)} |")
w("")
sh = J(f"{PL}/shared40_300.json")
w("공통 hard 제약(모든 arm이 t0 관측 위치 디스크로 hard 검사, soft는 arm별), 인덱스 0~299:\n")
w("| 비교 | 152 m 비 [CI], p | 45 m 미만 비 [CI], p | 45 m 아래 시간 W/L, p |")
w("|---|---|---|---|")
for pr in ("cv_sh|model_sh", "none_sh|model_sh", "none_sh|cv_sh", "cv_arm|cv_sh", "model_arm|model_sh", "none_arm|none_sh"):
    if pr in sh["paired"]:
        P = sh["paired"][pr]; e1 = P["events"]["union20_152_new"]; e3 = P["events"]["union20_severe_new"]; t = P["time_below"]["union20_tb_45"]
        w(f"| {pr} | {ratio(e1)}, {e1['sign_p']:.3f} | {ratio(e3)}, {e3['sign_p']:.3f} | {t['wins']}/{t['losses']}, {t['sign_p']:.3f} |")
w("")
w("| arm(공통 hard) | feasible 평균 | 비상 비율 |")
w("|---|---|---|")
for a in ("none_sh", "cv_sh", "model_sh"):
    pl = sh["per_arm"][a]["planner"]; w(f"| {a} | {pl['feasible_frac_mean']:.3f} | {pl['emergency_frac']:.4f} |")
w("")

# -------------------------------------------------------------- M5 floor + sens
w("## 5. R3-M5 σ 하한과 1요인 민감도 (인덱스 0~299, 기준 = ood40 기본값)\n")
fl = core["ood40"]["per_arm"]["model"]["floor"]
w(f"σ 하한(10 m) 활성 비율(예측 디스크 σ, 0.5~15 s): PRISM {f(fl['floor_active_frac'], 4)}. 갈래 σ가 하한 아래인 비율: "
  f"5 s {f(fl['branch_sigma_below_floor_5s'], 4)}, 10 s {f(fl['branch_sigma_below_floor_10s'], 4)}, 15 s {f(fl['branch_sigma_below_floor_15s'], 5)}. "
  f"CV는 {f(core['ood40']['per_arm']['cv']['floor']['floor_active_frac'], 4)}.\n")
w("| 설정 | cv 152/h | PRISM 152/h | cv/PRISM 152 비 [CI], p | cv 심각/h | PRISM 심각/h | cv/PRISM 심각 비 [CI], p |")
w("|---|---|---|---|---|---|---|")
base = J(f"{PL}/sens300_floor1.json")
Pb = base["paired"]["cv_base|model_base"]; Ab = base["per_arm"]
w(f"| 기준(10 m, ×1, w 12, pool; 인덱스 0~299) | {f(Ab['cv_base']['events']['union20_152_new']['per_hour'])} | {f(Ab['model_base']['events']['union20_152_new']['per_hour'])} | "
  f"{ratio(Pb['events']['union20_152_new'])}, {Pb['events']['union20_152_new']['sign_p']:.3f} | {f(Ab['cv_base']['events']['union20_severe_new']['per_hour'])} | "
  f"{f(Ab['model_base']['events']['union20_severe_new']['per_hour'])} | {ratio(Pb['events']['union20_severe_new'])}, {Pb['events']['union20_severe_new']['sign_p']:.3f} |")
for c, label in (("floor1", "σ 하한 1 m"), ("floor20", "σ 하한 20 m"), ("plat05", "plateau ×0.5"), ("plat2", "plateau ×2"),
                 ("wsafe6", "w_safety 6"), ("wsafe24", "w_safety 24"), ("normabs", "absolute 정규화 (S_REF 23.61)")):
    d = J(f"{PL}/sens300_{c}.json"); A = d["per_arm"]; P = d["paired"][f"cv_{c}|model_{c}"]
    w(f"| {label} | {f(A['cv_'+c]['events']['union20_152_new']['per_hour'])} | {f(A['model_'+c]['events']['union20_152_new']['per_hour'])} | "
      f"{ratio(P['events']['union20_152_new'])}, {P['events']['union20_152_new']['sign_p']:.3f} | {f(A['cv_'+c]['events']['union20_severe_new']['per_hour'])} | "
      f"{f(A['model_'+c]['events']['union20_severe_new']['per_hour'])} | {ratio(P['events']['union20_severe_new'])}, {P['events']['union20_severe_new']['sign_p']:.3f} |")
w("\nPRISM에만 적용되는 설정(기준 = 같은 인덱스의 기본 PRISM, CV는 기본값):\n")
w("| 설정 | PRISM 152/h · 심각/h (기준 → 설정) | PRISM 기준/설정 152 m 비, p | PRISM 기준/설정 심각 비, p | 최소이격 W/L/T, p (W = 기준이 큼) | CV/PRISM(설정) 152 m 비 | CV/PRISM(설정) 심각 비 |")
w("|---|---|---|---|---|---|---|")
for c, label in (("pimin20", "π_min 0.20"), ("isomax", "등방화 max(σx, σy)"), ("isomin", "등방화 min(σx, σy)")):
    p = f"{PL}/sens300_{c}.json"
    if not os.path.exists(p):
        w(f"| {label} | PENDING | | | | | |"); continue
    d = J(p); P = d["paired"][f"model_base|model_{c}"]; Q = d["paired"][f"cv_base|model_{c}"]; A = d["per_arm"]
    e1, e3, m = P["events"]["union20_152_new"], P["events"]["union20_severe_new"], P["min_sep_20hz"]
    w(f"| {label} | {f(A['model_base']['events']['union20_152_new']['per_hour'])} · {f(A['model_base']['events']['union20_severe_new']['per_hour'])} → "
      f"{f(A['model_'+c]['events']['union20_152_new']['per_hour'])} · {f(A['model_'+c]['events']['union20_severe_new']['per_hour'])} | "
      f"{ratio(e1)}, {e1['sign_p']:.3f} | {ratio(e3)}, {e3['sign_p']:.3f} | {m['wins']}/{m['losses']}/{m['ties']}, {m['sign_p']:.4f} | "
      f"{ratio(Q['events']['union20_152_new'])} | {ratio(Q['events']['union20_severe_new'])} |")
w("")

# -------------------------------------------------------------- M8 cal + decomp
w("## 6. R3-M8 / R2-IV 보정 σ(T = 1.695)와 중심·폭 분해 (인덱스 0~299)\n")
w("| 비교 | 152 m 비 [CI], p | 45 m 미만 비 [CI], p | 45 m 아래 시간 W/L, p | 최소이격 W/L/T, p |")
w("|---|---|---|---|---|")
for fn, prs in (("cal40_300", ("model_raw|model_cal", "cv|model_cal", "none|model_cal")),
                ("decomp40_300", ("cv|mc_cvsig", "cv|cvc_msig", "model|mc_cvsig", "model|cvc_msig", "mc_cvsig|cvc_msig"))):
    d = J(f"{PL}/{fn}.json")
    for pr in prs:
        if pr not in d["paired"]:
            continue
        P = d["paired"][pr]; e1 = P["events"]["union20_152_new"]; e3 = P["events"]["union20_severe_new"]; t = P["time_below"]["union20_tb_45"]; m = P["min_sep_20hz"]
        w(f"| {pr} | {ratio(e1)}, {e1['sign_p']:.3f} | {ratio(e3)}, {e3['sign_p']:.3f} | {t['wins']}/{t['losses']}, {t['sign_p']:.3f} | "
          f"{m['wins']}/{m['losses']}/{m['ties']}, {m['sign_p']:.4f} |")
w("")
for fn in ("cal40_300", "decomp40_300"):
    d = J(f"{PL}/{fn}.json")
    w(f"{fn} arm별: " + "; ".join(f"{a} 152 {f(v['events']['union20_152_new']['per_hour'])}/h, 심각 {f(v['events']['union20_severe_new']['per_hour'])}/h, "
                                  f"p5 {f(v['min_sep_20hz']['p5'], 1)} m" for a, v in d["per_arm"].items()) + "\n")

# -------------------------------------------------------------- timing
w("## 7. R3-m2 계산 시간 (공유 CPU/GPU, 실시간 벤치마크 아님)\n")
w("| 조건 | arm | 사이클 p50 중앙값 [ms] | p99 최대 [ms] | 최대 [ms] | 0.5 s 마감 미스 / 사이클 |")
w("|---|---|---|---|---|---|")
tot = {a: [0, 0] for a in ("none", "cv", "model")}
for c in ("sparse", "ood40", "ood100", "id40", "id24"):
    for a in ("none", "cv", "model"):
        t = core[c]["per_arm"][a]["timing"]
        tot[a][0] += t["deadline_miss"]; tot[a][1] += t["cycles"]
        w(f"| {c} | {a} | {f(t['cycle_ms_p50_median'], 1)} | {f(t['cycle_ms_p99_max'], 1)} | {f(t['cycle_ms_max'], 1)} | {t['deadline_miss']} / {t['cycles']} |")
w("| **전체** | " + " · ".join(f"{a} {m}/{n}" for a, (m, n) in tot.items()) + " | | | | |")
w("\n러너는 다른 샤드·학습과 CPU/GPU를 공유했다. 마감 미스는 모두 PRISM arm이며 id24(30건)와 ood40(1건)에서만 났다.\n")

# -------------------------------------------------------------- R3-m4 economy
ep = f"{PL}/economy_r3.json"
if os.path.exists(ep):
    ec = J(ep)["conditions"]
    w("## 9. R3-m4 기동 경제: 선회율 통계의 정확한 정의와 산포 (`planner/economy_r3.json`, `analysis_economy.py`)\n")
    w("ω95 = **에피소드별 |ω| 95백분위(2 Hz ownship 기록)의 에피소드 평균** [°/s]. 제출본 Table 11의 값이 이 통계이며 "
      "풀링한 95백분위가 아니다. 산포는 에피소드 간 SD와 중앙값 [IQR]. 짝비교는 같은 인덱스 에피소드의 평균 차 a − b, "
      "paired bootstrap 95 % CI(B = 10,000), 정확 부호검정(W = a가 큼).\n")
    w("| 조건 | arm | n | ω95 평균 ± SD | ω95 중앙값 [IQR] | 평균 |ω| | 경로 연장 [%] 평균 ± SD | σ_v [m/s] 평균 ± SD |")
    w("|---|---|---|---|---|---|---|---|")
    for c, d in ec.items():
        for a in ("none", "cv", "model"):
            s = d["per_arm"][a]; o = s["omega_p95_ep"]
            w(f"| {c} | {a} | {o['n']} | {o['mean']:.2f} ± {o['sd']:.2f} | {o['median']:.2f} [{o['q25']:.2f}, {o['q75']:.2f}] | "
              f"{s['omega_mean_ep']['mean']:.2f} | {s['elong_pct']['mean']:.3f} ± {s['elong_pct']['sd']:.3f} | "
              f"{s['v_std']['mean']:.2f} ± {s['v_std']['sd']:.2f} |")
    w("")
    w("| 조건 | 비교 | ω95 차 [CI] (상대 %), W/L, p | 경로 연장 차 [%p] [CI], p | σ_v 차 [CI], p |")
    w("|---|---|---|---|---|")
    for c, d in ec.items():
        for pr in ("none|model", "cv|model", "none|cv"):
            P = d["paired"][pr]; o, e, v = P["omega_p95_ep"], P["elong_pct"], P["v_std"]
            w(f"| {c} | {pr} | {o['diff_mean']:+.3f} [{o['ci'][0]:+.3f}, {o['ci'][1]:+.3f}] ({o['rel_pct']:+.1f} %), "
              f"{o['wins']}/{o['losses']}, {o['sign_p']:.3f} | {e['diff_mean']:+.4f} [{e['ci'][0]:+.4f}, {e['ci'][1]:+.4f}], {e['sign_p']:.3f} | "
              f"{v['diff_mean']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}], {v['sign_p']:.3f} |")
    w("\n차 a − b < 0이면 b(오른쪽 arm)가 더 많이 선회/연장/속도 변동한다.\n")
    w("사건 발생 에피소드 비율(제출본 Table 11의 열; 사건 ≥ 1건인 에피소드 비율, 새 스크린, 원장과 같은 "
      "`aggregate_r3.episode_endpoints` 집계). 짝비교: 차 a − b와 paired bootstrap CI, 정확 McNemar(불일치 에피소드).\n")
    w("| 조건 | 152 m none·cv·PRISM | 76 m none·cv·PRISM | 45 m 미만 none·cv·PRISM | cv − PRISM 152 m 차 [CI], cv만/PRISM만, p |")
    w("|---|---|---|---|---|")
    for c, d in ec.items():
        F = d["event_fraction"]; pa = F["per_arm"]; q = F["paired"]["cv|model"]["union20_152_new"]
        w(f"| {c} | " + " · ".join(f"{pa[a]['union20_152_new']['fraction']:.3f}" for a in ("none", "cv", "model")) + " | "
          + " · ".join(f"{pa[a]['union20_76_new']['fraction']:.3f}" for a in ("none", "cv", "model")) + " | "
          + " · ".join(f"{pa[a]['union20_severe_new']['fraction']:.3f}" for a in ("none", "cv", "model")) + " | "
          f"{q['diff']:+.3f} [{q['ci'][0]:+.3f}, {q['ci'][1]:+.3f}], {q['only_a']}/{q['only_b']}, {q['mcnemar_exact_p']:.3f} |")
    w("")
else:
    w("## 9. R3-m4 기동 경제\n\nPENDING: `python analysis_economy.py`\n")

# -------------------------------------------------------------- Table 11 (revised), manuscript format
if os.path.exists(ep):
    SET = {"sparse": "Sparse, OOD", "ood40": "Nominal load, OOD", "ood100": "Saturated, OOD",
           "id40": "Nominal density, ID", "id24": "Matched load, ID"}
    ARM_NAME = {"none": "A: static observation", "cv": "B: CV", "model": "C: PRISM"}
    w("## 10. 표 11 개정안 (원고 형식, 영문; 위 1·9절과 같은 원자료)\n")
    w("Table 11 (revised). Paired closed-loop evaluation. Encounters are counted on the union of intruders at 20 Hz, after "
      "the kinematic-avoidability screen based on the currently open track. ω95 is the mean over episodes of the "
      "per-episode 95th percentile of |ω| (± between-episode SD).\n")
    w("| Setting (agents; episodes; exposure h per arm) | Risk input | NMAC-radius encounters (/h) | Fraction of episodes with ≥ 1 | "
      "Severe < 45 m (/h) [count] | Time below 45 m (s/h) | 5th pct. of episode min. separation (m) | ω95 (°/s) |")
    w("|---|---|---|---|---|---|---|---|")
    for c in ("sparse", "ood40", "ood100", "id40", "id24"):
        A = core[c]["per_arm"]; e = ec[c]
        na = e["n_agents"]["model"]["mean"]
        lab = f"{SET[c]} ({na:.0f}; {A['model']['episodes'] if isinstance(A['model'].get('episodes'), int) else e['per_arm']['model']['omega_p95_ep']['n']}; {A['model']['exposure_hours']:.1f})"
        for a in ("none", "cv", "model"):
            ev = A[a]["events"]; o = e["per_arm"][a]["omega_p95_ep"]
            w(f"| {lab} | {ARM_NAME[a]} | {ev['union20_152_new']['per_hour']:.1f} | "
              f"{e['event_fraction']['per_arm'][a]['union20_152_new']['fraction']:.3f} | "
              f"{ev['union20_severe_new']['per_hour']:.2f} [{ev['union20_severe_new']['total']}] | "
              f"{A[a]['time_below']['union20_tb_45']['per_hour_s']:.1f} | {A[a]['min_sep_20hz']['p5']:.1f} | {o['mean']:.2f} ± {o['sd']:.2f} |")
    w("")
    w("Table 11b (revised). Paired comparisons (pairs share the scenario seed and initial conditions; the traffic then evolves stochastically and diverges between arms, DECISIONS D6.19; B/C and A/C rate ratios, > 1 = fewer encounters with C; "
      "95 % paired episode-bootstrap CI, B = 10,000; p = exact two-sided sign test; better/worse = paired episodes in which C "
      "is better/worse, ties excluded).\n")
    w("| Setting | Comparison | NMAC-radius ratio [CI], p | Severe ratio [CI], p | Time below 45 m: C better/worse, p | "
      "Episode min. separation: C better/worse/tie, p | Δ 5th pct. min. sep. C − other (m) [CI] | Δω95 C − other (°/s) [CI] |")
    w("|---|---|---|---|---|---|---|---|")
    for c in ("sparse", "ood40", "ood100", "id40", "id24"):
        for pr, lab in (("cv|model", "C vs B"), ("none|model", "C vs A")):
            P = core[c]["paired"][pr]; e1, e3 = P["events"]["union20_152_new"], P["events"]["union20_severe_new"]
            t = P["time_below"]["union20_tb_45"]; m = P["min_sep_20hz"]; o = ec[c]["paired"][pr]["omega_p95_ep"]
            w(f"| {SET[c]} | {lab} | {ratio(e1)}, {e1['sign_p']:.3f} | {ratio(e3)}, {e3['sign_p']:.3f} | "
              f"{t['losses']}/{t['wins']}, {t['sign_p']:.3f} | {m['losses']}/{m['wins']}/{m['ties']}, {m['sign_p']:.4f} | "
              f"{-m['p5_diff']['est']:+.1f} [{-m['p5_diff']['ci'][1]:+.1f}, {-m['p5_diff']['ci'][0]:+.1f}] | "
              f"{-o['diff_mean']:+.2f} [{-o['ci'][1]:+.2f}, {-o['ci'][0]:+.2f}] |")
    w("\n표 11b의 better/worse는 C 기준이다. 원장 1절의 W/L(W = 왼쪽 arm이 나음)을 뒤집어 옮겼다(C better = 원장 L).\n")

# -------------------------------------------------------------- prediction
w("## 8. 예측 (시드 1, 현재 트랙 셋) — 원자료 `Logs/Analysis/eval_r3/`\n")
for name in ("r3_k3_s1_testid_report.json", "r3_k3_s1_testood_report.json", "s4_testid_indexcurrent_report.json"):
    p = f"{EV}/{name}"
    if os.path.exists(p):
        o = J(p)["overall"]
        w(f"- `{name}`: n {J(p)['n_samples']}, minADE {f(o.get('minADE'))}, NLL {f(o.get('NLL'), 4)}, trajNLL {f(o.get('trajNLL'), 4)}, "
          f"top1ADE {f(o.get('top1ADE'))}, ES {f(o.get('ES'), 1)}, cov90@5/10/15 s {f(o.get('cov90_5s'), 4)}/{f(o.get('cov90_10s'), 4)}/{f(o.get('cov90_15s'), 4)}")
w("")
open(f"{HERE}/R3_RESULTS.md", "w", encoding="utf-8").write("\n".join(out) + "\n")
print("written", f"{HERE}/R3_RESULTS.md", len(out), "lines")
