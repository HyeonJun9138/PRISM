"""Figure 12 review: distinguish selection among feasible candidates.

Dark solid blue is the exact selected 13 s plan. The long, lighter blue
dashed curve is the executed flight under subsequent replanning. They are
separate recorded objects, never joined into one invented selected plan.
No simulation, model, candidate, forecast record, or manuscript is modified.
The partial maps must not be represented as the full planner risk field.
"""

import os
from pathlib import Path
import sys
import os
import json
import pickle
import hashlib

sys.dont_write_bytecode = True
BASE = Path(os.environ['PRISM_ROOT']) / 'Paper'
WORK = BASE / 'figures_r3'
WORK.mkdir(parents=True, exist_ok=True)
OUT = BASE / 'figures_r3'
os.environ['MPLCONFIGDIR'] = str(WORK / 'plot_cache')
SOURCE = Path((os.environ["PRISM_ROOT"] + '/Paper/figures/make_fig_explain_r3_sel.py'))
DATA = Path((os.environ["PRISM_ROOT"] + '/Paper/figures_r3/data'))
STEM = 'fig12_original_overlay_v4'
CACHE = WORK / 'fig12_original_scene_cache.pkl'
OMITTED_SLOTS = {0, 4, 5, 9}
SUBSET_CACHE = WORK / 'fig12_upper_traffic_omitted_fields.npz'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


protected = [DATA / 'scene_319.json', DATA / 'case_319.json', SOURCE]
before = {str(p): digest(p) for p in protected}
source_text = SOURCE.read_text(encoding='utf-8')
cache_signature = {str(p): digest(p) for p in
                   [SOURCE, DATA / 'scene_319.json', DATA / 'case_319.json',
                    Path((os.environ["PRISM_ROOT"] + '/Paper/figures/make_fig_hdr.py'))]}
g = {'__name__': 'fig12_original_readonly', '__file__': str(SOURCE)}
if CACHE.exists():
    cached = pickle.loads(CACHE.read_bytes())
    assert cached['signature'] == cache_signature, 'Stale source cache'
    preamble = source_text[:source_text.index('# ------------------------------------------------------------------ load')]
    exec(compile(preamble, str(SOURCE), 'exec'), g)
    g.update(cached['data'])
    drawing = source_text[source_text.index('def own_marker():'):
                          source_text.index('files: list[str] = []')]
    exec(compile(drawing, str(SOURCE), 'exec'), g)
    print('Loaded verified original scene cache', flush=True)
else:
    prefix = source_text[:source_text.index('files: list[str] = []')]
    exec(compile(prefix, str(SOURCE), 'exec'), g)
    keys = ['EXT', 'TICK', 'L', 'GRID', 'R', 'VMAX', 'IN_VIEW', 'FC',
            'TRACKS', 'CRIT', 'OWN', 'PSI', 'OWN_HIST', 't_own', 'T_FIG',
            'CANDS', 'SEL', 'ok', 'bad', 'EMERGENCY', 'horizon', 'case',
            'D', 'S', 'VAL', 'CRIT_FC', 'tr_c']
    CACHE.write_bytes(pickle.dumps({'signature': cache_signature,
                                   'data': {k: g[k] for k in keys}}, protocol=5))
    print('Cached original fields after original validation checks', flush=True)

D = g['D']

# Upper-scene sources: an off-frame bird (0), fixed-wing UAS (4),
# multirotor (5), and the upper-left general-aviation aircraft (9).
# They are omitted from these explanatory maps, NOT from planning.
assert g['CRIT'] not in OMITTED_SLOTS
assert all(g['TRACKS'][i]['obs_t0'][1] > 350 for i in OMITTED_SLOTS)
full_hpd = g['L'].copy()
full_risk = g['R'].copy()
displayed_slots = [i for i in g['IN_VIEW'] if i not in OMITTED_SLOTS]
subset_signature = json.dumps({'sources': cache_signature,
    'omitted_slots': sorted(OMITTED_SLOTS)}, sort_keys=True)
if SUBSET_CACHE.exists():
    with np.load(SUBSET_CACHE, allow_pickle=False) as z:
        assert z['signature'].item() == subset_signature
        display_hpd = z['L'].copy()
        display_risk = z['R'].copy()
else:
    hdr = g['HDR']
    display_hpd = np.ones_like(g['L'])
    for slot in displayed_slots:
        f = g['FC'][slot]
        nodes = hdr.prism_nodes(f, np.zeros(2))
        schedule = hdr.display_schedule(nodes, f['origin'], True)
        node_tables = [hdr.StepHPD(cp) for cp in nodes]
        tables = [node_tables[j] if j is not None else hdr.StepHPD(cp)
                  for _, cp, j in schedule]
        assert min(t.mass for t in tables) >= hdr.MASS_MIN
        level = hdr.swept_levels([cp for _, cp, _ in schedule], tables, g['GRID'])
        display_hpd = np.minimum(display_hpd, level)
        print(f'Retained HPD source {slot}', flush=True)
    rg = hdr.make_grid(g['EXT'], g['R'].shape[0])
    disks = np.asarray(D['disks'], float)
    disk_slots = np.array([x[0] for x in D['disk_slot_map']])
    disks = disks[:, ~np.isin(disk_slots, list(OMITTED_SLOTS))]
    vr = D['planner']['vehicle_radius']
    st = D['planner']['soft_standoff']
    reach = D['risk']['hard_radius'] + vr + st
    ext = g['EXT']
    display_risk = np.zeros_like(g['R'])
    for frame in disks:
        field = np.zeros_like(display_risk)
        for cx, cy, hr, sigma, gain in frame:
            if (cx < ext[0] - reach - 6*sigma or cx > ext[1] + reach + 6*sigma
                    or cy < ext[2] - reach - 6*sigma or cy > ext[3] + reach + 6*sigma):
                continue
            gap = np.maximum(0, np.hypot(rg['X']-cx, rg['Y']-cy)-hr-vr-st)
            field += gain * np.exp(-gap**2/(2*sigma**2))
        display_risk = np.maximum(display_risk, field/D['risk']['base_gain'])
    np.savez_compressed(SUBSET_CACHE, L=display_hpd, R=display_risk,
                        signature=np.array(subset_signature))
assert np.all(display_hpd >= full_hpd-1e-10)
assert np.all(display_risk <= full_risk+1e-9)
assert np.any(display_risk < full_risk-.5)
g['IN_VIEW'] = displayed_slots
g['L'] = display_hpd
g['R'] = display_risk

arm = g['case']['arms']['model']
t0 = float(g['T_FIG'])
old = next(c for c in arm['cycles'] if c['t'] == 16.0)
now = D['candidates'][D['selected']]
assert t0 == 17.5 and now['T'] == 13.0
assert len(g['ok']) == 98 and len(g['bad']) == 177
assert sum(bool(c['collision']) for c in D['candidates']) == 0


def recorded_plan_times(T):
    full = np.arange(0.0, T, 0.5)
    if abs(full[-1] - T) > 1e-8:
        full = np.r_[full, T]
    ii = list(range(0, len(full), 2))
    if ii[-1] != len(full) - 1:
        ii.append(len(full) - 1)
    return full[ii]


old_time = recorded_plan_times(old['sel']['T']) + old['t']
old_xy = np.asarray(old['plan'], float)
assert len(old_xy) == len(old_time)
q = np.r_[t0, old_time[old_time > t0]]
original = np.c_[np.interp(q, old_time, old_xy[:, 0]),
                  np.interp(q, old_time, old_xy[:, 1])]
flight_time = np.asarray(arm['t20'], float)
flight_xy = np.asarray(arm['own20'], float)
m = (flight_time >= t0) & (flight_time <= q[-1] + 1e-8)
flight = flight_xy[m]
assert np.linalg.norm(flight[0] - g['OWN']) < 0.01
assert flight_time[m][-1] == 44.3

# Validation of the exact trajectory sources. Never splice the selected
# plan's endpoint to the later executed route: they are different objects.
assert np.array_equal(flight, flight_xy[m])
curr_cycle = next(c for c in arm['cycles'] if c['t'] == t0)
ix = list(range(0, len(now['t']), 2))
if ix[-1] != len(now['t']) - 1:
    ix.append(len(now['t']) - 1)
assert np.allclose(curr_cycle['plan'],
                   np.round(np.c_[now['x'], now['y']][ix], 1), atol=1e-8)

OLD = '#A02B72'
BLUE = '#0069BC'
FLOWN = '#5598BD'
INK = '#222222'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.2,
    'axes.titlesize': 8.8, 'axes.labelsize': 7.4,
    'xtick.labelsize': 6.8, 'ytick.labelsize': 6.8,
    'pdf.fonttype': 42, 'svg.fonttype': 'none'})

# Same extent, orientation, aspect ratio and recorded routes. Only the
# upper traffic contributions are omitted from the two displayed maps.
W, H = 7.62, 4.46
side, gap, x0, bottom = 3.30, 0.26, 0.54, .94
fig = plt.figure(figsize=(W, H), facecolor='white')
axa = fig.add_axes([x0/W, bottom/H, side/W, side/H])
axb = fig.add_axes([(x0+side+gap)/W, bottom/H, side/W, side/H])
g['setup'](axa, '(a) Forecasts and original plan')
g['setup'](axb, '(b) Selection among feasible candidates')
axa.set_ylabel('y [m]', labelpad=1.5)
axb.tick_params(axis='y', labelleft=False)
av_a = g['draw_forecasts'](axa)

# Display the reduced-source field, with only the feasible candidates.
# Their classifications and selection were evaluated using ALL traffic.
rgba = g['RISK_CMAP'](Normalize(0.0, g['VMAX'])(g['R']))
rgba[..., 3] = (g['R'] >= g['RISK_BLANK']).astype(float) * 0.92
axb.imshow(rgba, extent=g['EXT'], origin='lower', interpolation='nearest', zorder=1)
for i in g['ok']:
    if i == g['SEL']:
        continue
    c = g['CANDS'][i]
    axb.plot(c['x'], c['y'], color='#999999', lw=.4, alpha=.66, zorder=4)
for slot in sorted(set(g['IN_VIEW']) | {g['CRIT']}):
    p = g['arr'](g['TRACKS'][slot]['obs_t0'])
    if np.all(np.isfinite(p)):
        axb.plot(*p, 'o', ms=2.6, mfc=g['HDR'].OBS, mec='white', mew=.4, zorder=6)
g['draw_ownship'](axb)

halo = [pe.Stroke(linewidth=3.9, foreground='white'), pe.Normal()]


def along_arrow(ax, xy, fraction, color, lw=1.4):
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    s = fraction * distance[-1]
    q2 = [max(0, s-50), min(distance[-1], s+20)]
    pp = np.c_[np.interp(q2, distance, xy[:, 0]),
                np.interp(q2, distance, xy[:, 1])]
    ax.annotate('', xy=pp[1], xytext=pp[0],
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                                mutation_scale=8.5, shrinkA=0, shrinkB=0), zorder=13)


def note(ax, text, point, at, color=INK, align='left', size=7.5):
    return ax.annotate(text, xy=point, xytext=at, ha=align, va='center',
        fontsize=size, color=color, zorder=22,
        bbox=dict(facecolor='white', edgecolor='none', alpha=.92, pad=1.3),
        arrowprops=dict(arrowstyle='-', color=color, lw=.65,
                        shrinkA=3, shrinkB=3))


for ax in [axa, axb]:
    ax.plot(*original.T, color=OLD, lw=1.8, ls=(0, (4.8, 2.4)),
            path_effects=halo, zorder=10)
    along_arrow(ax, original, .48, OLD)
    along_arrow(ax, original, .86, OLD)
    ax.plot(*g['OWN'], marker=g['own_marker'](), ms=8.4, mfc=INK,
            mec='white', mew=.8, ls='none', zorder=20)
    note(ax, 'Ownship', g['OWN'], (1830, -285), size=7.0)

selected_xy = np.c_[now['x'], now['y']]
assert now['valid'] and g['SEL'] in g['ok']
axb.plot(*flight.T, color=FLOWN, lw=1.4, ls=(0,(3.6,2.0)),
         path_effects=[pe.Stroke(linewidth=2.5,foreground='white'),pe.Normal()],
         zorder=12)
along_arrow(axb, flight, .83, FLOWN, lw=1.1)
axb.plot(*selected_xy.T, color=BLUE, lw=2.1, path_effects=halo,
         solid_capstyle='round', zorder=16)
along_arrow(axb, selected_xy, .58, BLUE, lw=1.5)
axb.plot(*selected_xy[-1], marker='o', ms=4.3, mfc='white', mec=BLUE,
         mew=1.2, zorder=19)

# Only call this spatial proximity, not a same-time predicted collision.
# Select a location from the actual logged earlier plan nearest to the
# recorded bird forecast means; do not move either curve to create a hit.
bird_fc = g['FC'][g['CRIT']]['mu_w'].reshape(-1, 2)
near_idx = np.argmin(np.min(np.linalg.norm(
    original[:, None, :] - bird_fc[None, :, :], axis=2), axis=1))
near = original[near_idx]
ring = Circle(near, 80, fill=False, ec=OLD, lw=1.0, ls=(0, (2, 1.8)), zorder=15)
axa.add_patch(ring)
note(axa, 'Original plan', original[12], (1475, -245), OLD, size=7.7)
note(axa, 'Near bird\nforecast region', near, (1460, 220), OLD, size=7.7)
note(axb, 'Original plan', original[17], (1470, -175), OLD, size=7.7)
note(axb, 'Selected plan', selected_xy[-6],
     (1670, 20), BLUE, size=7.7)
note(axb, 'Subsequent flight', flight[int(len(flight)*.68)],
     (1475, 275), FLOWN, size=7.2)

# Preserve the original traffic labels and branch probabilities, while
# passing the new trajectories as label-avoidance geometry.
lab_a = g['add_intruder_labels'](axa, np.vstack([av_a, original, g['OWN'][None]]))
g['add_intruder_labels'](axb, np.vstack([original, flight, selected_xy, g['OWN'][None]]))
g['label_pi'](fig, axa, lab_a)

# Put display omissions and timing caveats in the separate caption rather
# than in grey sentences above and below the figure.
handles = [Line2D([], [], color=OLD, lw=1.8, ls=(0, (4.8, 2.4))),
           Line2D([], [], color=BLUE, lw=2.1,marker='o',ms=3.2,mfc='white'),
           Line2D([], [], color=FLOWN, lw=1.4,ls=(0,(3.6,2.0)))]
fig.legend(handles, ['Original plan (16.0 s)', 'Selected plan (13 s)',
                     'Executed flight with replanning'],
           loc='center', bbox_to_anchor=(.54, .50/H), ncol=3,
           frameon=False, fontsize=7.0, columnspacing=1.6,
           handlelength=2.6, handletextpad=.6)

left_handles = [Line2D([], [], color=g['HDR'].OBS, marker='o', ls='none', ms=2.7),
                Line2D([], [], color=g['HDR'].TRUE_FUT, lw=.9, ls=g['HDR'].DASH),
                Line2D([], [], color=g['HDR'].INK, lw=1.0),
                Line2D([], [], color=g['HDR'].INK, lw=1.0, ls=(0,(4,2)))]
fig.legend(left_handles, ['Observed', 'True future', r'Mean path (width $\propto \pi$)',
                          'Ownship past flight'],
           loc='center', bbox_to_anchor=((x0+side/2)/W, .19/H), ncol=2,
           frameon=False, fontsize=6.3, columnspacing=1.05,
           handlelength=2.0, handletextpad=.5)
right_handles = [Line2D([], [], color='#999999', lw=.9),
                 Line2D([], [], color=g['HDR'].OBS, marker='o', ls='none', ms=2.7)]
fig.legend(right_handles, ['Other feasible candidates (97)',
                           'Detections at 17.5 s'],
           loc='center', bbox_to_anchor=((x0+side+gap+side/2)/W, .19/H), ncol=1,
           frameon=False, fontsize=6.3, columnspacing=1.0,
           handlelength=2.0, handletextpad=.5)

for ax, norm, cmap, title, ticks, labels in [
    (axa, g['HDR'].NORM, g['HDR'].CMAP, 'HPD (%)',
     [0,.5,.9], ['0','50','90']),
    (axb, Normalize(0,g['VMAX']), g['RISK_CMAP'], 'Risk',
     [0,g['VMAX']], ['Low','High'])]:
    cbax = ax.inset_axes([.895,.690,.016,.210])
    cb = fig.colorbar(ScalarMappable(norm=norm,cmap=cmap), cax=cbax,
                      orientation='vertical')
    cb.set_ticks(ticks)
    cb.set_ticklabels(labels)
    cbax.yaxis.set_ticks_position('right')
    cbax.tick_params(labelsize=5.8, length=1.5, pad=2.0, width=.35)
    cb.outline.set_linewidth(.35)
    cb.outline.set_edgecolor('#888888')
    cbax.set_title(title, fontsize=6.7, pad=5, loc='center')
    if ax is axa:
        # A small HPD level indicates higher predictive density, not a
        # lower planning risk. Keep its numerical key distinct from Risk.
        cbax.invert_yaxis()

fig.canvas.draw()
for ext in ['png','pdf','svg']:
    target = OUT / f'{STEM}.{ext}'
    fig.savefig(target, dpi=360)
preview = OUT / f'{STEM}_preview.png'
fig.savefig(preview, dpi=240)
plt.close(fig)

after = {str(p): digest(p) for p in protected}
assert before == after, 'An original input changed'
validation = {
    'originals_unchanged': before == after,
    'original_sha256': before,
    'same_original_world_extent': list(g['EXT']),
    'snapshot_s': t0,
    'earlier_plan_issued_s': 16.0,
    'earlier_plan_display_absolute_s': [float(q[0]),float(q[-1])],
    'dark_solid_blue_is': 'Exact selected candidate at t=17.5 s, horizon=13 s',
    'selected_candidate_id':g['SEL'],
    'selected_candidate_is_feasible':bool(now['valid']),
    'selected_xy_matches_log_exactly':bool(np.array_equal(selected_xy,np.c_[now['x'],now['y']])),
    'lighter_dashed_blue_is':'Actual subsequently executed flight with repeated replanning',
    'executed_interval_s': [float(flight_time[m][0]),float(flight_time[m][-1])],
    'executed_xy_matches_log_exactly':bool(np.array_equal(flight,flight_xy[m])),
    'displayed_other_candidate_ids':[i for i in g['ok'] if i!=g['SEL']],
    'displayed_rejected_candidate_count':0,
    'single_selected_plan_horizon_s': float(now['T']),
    'candidate_count': len(g['CANDS']), 'admissible':len(g['ok']),
    'rejected':len(g['bad']),
    'colliding_candidates_at_snapshot':sum(bool(c['collision']) for c in g['CANDS']),
    'spatial_proximity_annotation_world_m':near.tolist(),
    'temporal_caveat': 'A time-swept spatial overlap does not prove a same-time conflict.',
    'display_only_omitted_slots':sorted(OMITTED_SLOTS),
    'displayed_hpd_sources':displayed_slots,
    'planning_and_candidate_classification_used_all_traffic':True,
    'color_normalization_unchanged':float(g['VMAX']),
    'colorbars':'Compact vertical keys inside each panel',
    'figure_inches':[W,H],
    'subset_risk_no_greater_than_full_risk':bool(np.all(display_risk<=full_risk+1e-9)),
    'original_full_scene_pipeline_validation':g['VAL'],
    'outputs':[str(OUT/f'{STEM}.{e}') for e in ['png','pdf','svg']]+[str(preview)],
}
(OUT/f'{STEM}_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/f'{STEM}_caption_ko.txt').write_text(
    '그림 12 수정 검토안 v4. 실행 가능한 후보 가운데 실제로 선택한 계획을 구별하였다. '
    '진한 파란 실선과 끝의 빈 원은 17.5초에 실제로 선택한 13초 계획이다. 나머지 실행 '
    '가능 후보 97개는 회색으로 표시하며, 실행 불가능한 후보는 그림에서 생략하였다. '
    '옅은 파란 점선은 이후 반복 재계획을 거친 실제 비행 기록으로, 선택한 단일 후보와 '
    '구분하였다. 위아래 회색 설명 문장은 그림에서 삭제하고 해석에 필요한 설명을 이 '
    '캡션으로 옮겼다. 색상 막대는 각 패널 오른쪽 위에 작게 세우고 제목을 막대 바로 '
    '위에 가운데 정렬하였다. 조류 주변의 경로 비교를 명확히 하기 위해 상단 교통의 표시만 '
    '생략하였다. 제외한 표시 대상은 상단의 조류(slot 0), 고정익 무인기(slot 4), 멀티로터'
    '(slot 5), 경항공기(slot 9)이며, 핵심 조류(slot 2)와 하단 교통은 유지하였다. 이 대상들의 '
    '예측 궤적, 관측 표식, 이름, HPD 및 위험 기여분을 두 패널에서 일관되게 제외하였다. '
    '위험장은 남은 객체들의 기록된 커널을 합산하고 예측 시점별 최대값을 다시 계산한 '
    '부분 위험장이며, 색상 눈금과 좌표 범위는 변경하지 않았다. 원래 계획 계산, 후보 분류, '
    '선택 기록, 비행 궤적에는 전체 교통이 포함되어 있으며 재계산하거나 변경하지 않았다. '
    '따라서 이 그림을 전체 위험이 제거되었거나 파란 경로의 총위험이 낮다는 증거로 '
    '해석하면 안 된다. 양쪽의 자주색 점선은 16.0초에 생성한 이전 계획의 '
    '17.5초 이후 잔여 구간이다. 왼쪽은 이 경로가 조류의 예측 영역과 공간적으로 가까워지는 '
    '부분을 표시한다. 이는 여러 예측 시점을 투영한 지도에서의 공간적 관계이며, 같은 시각의 '
    '충돌을 입증하는 표시는 아니다. 오른쪽의 긴 옅은 파란 점선은 17.5초부터 44.3초까지 '
    '반복 재계획을 거치며 실제 비행한 기록이다. 진한 파란 실선은 별도로 표시한 13초 '
    '선택 계획이며, 두 경로를 임의로 이어 붙이지 않았다. 예측과 '
    '위험 배경은 17.5초에 계산한 0–15초 투영이므로 긴 후속 비행 전체가 처음 계산된 위험장을 '
    '최적화한 하나의 계획이라는 뜻도 아니다. 후보 275개 중 98개가 실행 가능했고, 나머지 '
    '177개는 경계 또는 동역학 제약으로 제외되었다. 원고와 원본 그림은 변경하지 않았다.\n',
    encoding='utf-8')
print(json.dumps({k:validation[k] for k in ['originals_unchanged',
    'same_original_world_extent','executed_interval_s','executed_xy_matches_log_exactly',
    'selected_candidate_is_feasible','displayed_rejected_candidate_count',
    'spatial_proximity_annotation_world_m','outputs']},ensure_ascii=False,indent=2),flush=True)
