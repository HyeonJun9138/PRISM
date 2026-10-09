"""Verify released corrected rows and reproduce all arm and paired statistics.

This uses released corrected rows, not a new simulation or raw-state replay.
All 10,000 bootstrap draws and the original seed are retained.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('round2_aggregate', ROOT/'modelVal/experiments/aggregate_r3.py')
AGG = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AGG)

def validate_row(r):
    if r.get('screen_version')!='open_asof_v2':raise ValueError('교정 선별 버전 누락')
    precision=r.get('endpoint_precision_version')
    certified=precision=='displayed_minimum_threshold_certified_v1'
    if precision!='raw_minimum_v2' and not certified:raise ValueError('원시 최소거리 또는 인증 버전 누락')
    if certified:
        proof=r.get('recovery_provenance',{})
        if (proof.get('method')!='native_grid_invariance_proof' or
            proof.get('raw_state_recovered') is not False or
            proof.get('endpoint_membership_certified') is not True or
            proof.get('screen_decision_certified') is not True):raise ValueError('비재생 행의 불변성 증명 누락')
    tversion=r.get('time_integration_version',r.get('time_integration'))
    if tversion!='left_tick_interval_v2':raise ValueError('구간 시간 적분 버전 누락')
    x=r.get('exposure_s')
    if x is None or not math.isfinite(float(x)) or float(x)<=0:raise ValueError('유효 scoring exposure 누락')
    for field in ('scoring_start_s','scoring_end_s'):
        if field not in r or not math.isfinite(float(r[field])):raise ValueError('scoring 시간 누락')
    if not math.isclose(float(r['scoring_end_s'])-float(r['scoring_start_s']),float(x),abs_tol=1e-8):
        raise ValueError('노출 및 scoring 구간 불일치')
    if 'pairwise' not in r or 'union20_event_list' not in r:raise ValueError('사건 원장 누락')
    for events in [r['pairwise']]+list(r['union20_event_list'].values()):
        for e in events:
            if e.get('min_sep_raw_m') is None:
                if not certified:raise ValueError('원시 사건 최소거리 누락')
                minimum=float(e['min_sep'])
                if not math.isfinite(minimum) or any(abs(minimum-t)<=.05000001 for t in (45.,76.2,152.4)):
                    raise ValueError('반올림 최소거리의 임계값 교차 가능성')
                if e.get('unavoidable_new') is not True:raise ValueError('비재생 행의 선별 포함은 증명 경로 밖')
            elif not math.isfinite(float(e['min_sep_raw_m'])):raise ValueError('유한 원시 최소거리 누락')
            if not isinstance(e.get('unavoidable_new'),bool):raise ValueError('교정 선별 boolean 누락')
    for event in r['pairwise']:
        for thr in ('152','76','45'):
            v=event.get(f'tb_{thr}_s')
            if v is None or not math.isfinite(float(v)) or v < -1e-8 or v > float(x)+1e-8:
                raise ValueError('각 pairwise 사건의 체류시간은 노출시간 이내이어야 함')
    for thr in ('152','76','45'):
        v=r.get(f'union20_time_below_{thr}_s')
        if v is None or not math.isfinite(float(v)) or v < -1e-8 or v > float(x)+1e-8:
            raise ValueError('union 체류시간은 노출 내 유한 비음수이어야 함')


def validate_strata_proof(r):
    """Count/exposure invariance is not a proof of turning strata invariance."""
    p=r.get('recovery_provenance',{})
    if p.get('method')=='native_grid_invariance_proof' and (r.get('pairwise') or r.get('events') or r.get('events_new_screen') or any(r.get('union20_event_list',{}).values())):
        raise ValueError('최종 proof 경로는 zero-event 행으로 제한')
    if r.get('pairwise') and p.get('strata_membership_certified') is not True:
        raise ValueError('pairwise turning-at-track-start strata의 독립 복원/증명 누락')


def unrounded_rate_stats(ca,cb,ha,hb,bidx):
    ca=np.asarray(ca,dtype=float);cb=np.asarray(cb,dtype=float)
    ha=np.asarray(ha,dtype=float);hb=np.asarray(hb,dtype=float)
    def val(x):return float(x) if np.isfinite(x) else None
    def ci(x):
        x=x[np.isfinite(x)]
        return [float(v) for v in np.percentile(x,[2.5,97.5])] if len(x) else None
    with np.errstate(divide='ignore',invalid='ignore'):
        a=ca.sum()/ha.sum();b=cb.sum()/hb.sum()
        ra=ca[bidx].sum(axis=1)/ha[bidx].sum(axis=1)
        rb=cb[bidx].sum(axis=1)/hb[bidx].sum(axis=1)
        ratio=np.where(rb>0,ra/rb,np.nan)
    undef=int(np.count_nonzero(~np.isfinite(ratio)))
    return {'rate_a':val(a),'rate_b':val(b),
       'rate_diff':{'est':val(a-b),'ci':ci(ra-rb)},
       'rate_ratio':{'est':val(a/b) if b>0 else None,'ci':ci(ratio),
          'undefined_draws':undef,'defined_draws':int(len(ratio)-undef),
          'undefined_frac':undef/len(ratio)},
       'sign_p':AGG.sign_test(int((ca<cb).sum()),int((ca>cb).sum()))}


def add_unrounded(paired,ra,rb,boot):
    indices=sorted(i for i in set(ra)&set(rb) if AGG.same_traffic(ra[i],rb[i]))
    if not indices:return
    a=[ra[i] for i in indices];b=[rb[i] for i in indices]
    bi=AGG.boot_indices(len(indices),boot,20260926)
    ha=[AGG.episode_exposure_s(r)/3600 for r in a]
    hb=[AGG.episode_exposure_s(r)/3600 for r in b]
    for fam,fn in [('events',AGG.episode_endpoints),('time_below',AGG.episode_times)]:
        ea=[fn(r) for r in a];eb=[fn(r) for r in b]
        for k,v in paired[fam].items():
            if v is None:continue
            v['unrounded']=unrounded_rate_stats([r[k] for r in ea],[r[k] for r in eb],ha,hb,bi)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=ROOT/'outputs/reproduction')
    ap.add_argument('--only',help='One aggregate name, otherwise all frozen comparisons')
    args=ap.parse_args()
    manifest=json.loads((ROOT/'results/round2/numbers_manifest_round2.json').read_text(encoding='utf-8'))
    cache={}
    for run, info in manifest['coverage']['inputs'].items():
        path=args.root/'round2/rescored'/(run+'.jsonl')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=info['sha256']:raise ValueError('Row checksum mismatch: '+run)
        rows={}
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():continue
            row=json.loads(line);idx=int(row['index'])
            if idx in rows:raise ValueError('Duplicate episode: '+run)
            validate_row(row);validate_strata_proof(row);rows[idx]=row
        if len(rows)!=info['expected']:raise ValueError('Episode count mismatch: '+run)
        cache[run]=rows
    print('Verified',sum(len(v) for v in cache.values()),'corrected rows',flush=True)
    count=0
    for name, reference in manifest['aggregates'].items():
        if args.only and name!=args.only:continue
        path=ROOT/'results/round2/aggregates'/(name+'.json')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=reference['sha256']:raise ValueError('Aggregate checksum mismatch: '+name)
        expected=json.loads(path.read_text(encoding='utf-8'));conds={}
        for label, info in expected['inputs'].items():
            run=info['paths'][0].replace('\\','/').split('/rescored/',1)[1][:-6]
            lo,hi=info['indices'];rows={i:v for i,v in cache[run].items() if lo<=i<=hi}
            if set(rows)!=set(range(lo,hi+1)):raise ValueError('Index range mismatch')
            if AGG.summarize(label,rows)!=expected['per_arm'][label]:raise ValueError(name+'/'+label+' arm statistics mismatch')
            conds[label]=rows
        for pair, want in expected['paired'].items():
            a,b=pair.split('|');ra,rb=conds[a],conds[b]
            if set(ra)!=set(rb) or any(not AGG.same_traffic(ra[i],rb[i]) for i in ra):raise ValueError('Unpaired traffic')
            got=AGG.compare(ra,rb,expected['B'],expected['seed']);add_unrounded(got,ra,rb,expected['B'])
            if got!=want:raise ValueError(name+'/'+pair+' paired statistics mismatch')
        count+=1;print(name+': exact arm and paired statistics match',flush=True)
    if not count:raise ValueError('No matching analysis')
    print('PASS:',count,'analyses; no reference files overwritten',flush=True)

if __name__=='__main__':main()
