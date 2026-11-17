"""Compute the final MERI scores directly from the fifteen-pen numerical timelines."""
from pathlib import Path
import argparse
import json
import sys

import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import calculate_sow_meri as m
from scripts.calculate_sow_meri_kdpa import summarize

START,END='2025-07-13','2025-08-02'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pens',nargs='+',default=[f'ch{i:02d}' for i in range(1,16)])
    p.add_argument('--output',type=Path,default=ROOT/'reproduced/meri')
    a=p.parse_args()
    if not a.pens or len(set(a.pens))!=len(a.pens) or any(pen not in {f'ch{i:02d}' for i in range(1,16)} for pen in a.pens):
        p.error('--pens must contain unique pen IDs ch01 through ch15')
    threadpool_limits(limits=1)
    base=ROOT/'data/MERI_15pens'
    cal=json.loads((base/'calibration_fixed.json').read_text(encoding='utf-8-sig'))
    a.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for pen in a.pens:
        bins=pd.read_csv(base/pen/'meri_timeline_1s.csv.gz',low_memory=False,float_precision='round_trip')
        widths=bins.bin_end_sec-bins.bin_start_sec
        if not (widths.gt(0).all() and widths.le(1.+1e-9).all()):
            raise ValueError(f'{pen}: invalid numerical time bins')
        events=m.score_events(bins,cal,pre_sec=20.,pre_burden_sec=120.,post_sec=60.,
            pre_burden_weight=.1,post_burden_weight=.15,utc_offset_hours=8.)
        events=m.physiological_nms(events,1200.)
        attempts=m.nursing_attempts(bins,events,pre_association_sec=240.,post_association_sec=180.,
            merge_gap_sec=30.,minimum_duration_sec=60.,minimum_probability=.5,utc_offset_hours=8.)
        events.to_csv(a.output/(pen+'_events.csv'),index=False)
        attempts.to_csv(a.output/(pen+'_attempts.csv'),index=False)
        e=events.loc[events.date.astype(str).between(START,END)]
        t=attempts.loc[attempts.date.astype(str).between(START,END)]
        result=summarize(e,t)
        rows.append(dict(pen=pen,**result))
        dates=sorted(set(e.date)|set(t.date))
        pd.DataFrame([dict(date=day,**summarize(e[e.date.eq(day)],t[t.date.eq(day)])) for day in dates]).to_csv(
            a.output/(pen+'_daily.csv'),index=False)
        print(f'{pen}: MERI={result["MERI_KDPA"]:.12f}',flush=True)
        del bins,events,attempts
    scores=pd.DataFrame(rows)
    scores.to_csv(a.output/'pen_scores.csv',index=False)

if __name__=='__main__':main()
