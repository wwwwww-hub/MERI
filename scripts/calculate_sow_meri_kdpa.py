"""Compute the final equal-weight K, Dme, P and A index."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import calculate_sow_meri as m

def summarize(events, attempts, threshold=55.0):
    eligible = m.equal4_eligible_events(events, threshold)
    pre = pd.to_numeric(eligible.pre_excess_burden_score, errors="coerce")
    missing_pre = int(pre.isna().sum())
    # A missing pre-release burden leaves the K component undefined.
    bp = float(pre.mean()) if len(pre) and missing_pre == 0 else None
    post = float(eligible.post_excess_burden_score.mean()) if len(eligible) else None
    d = float(eligible.release_drop_score.median()) if len(eligible) else None
    f = m.attempt_failure_fraction(attempts)
    components = {"K": None if bp is None else 100 - bp, "Dme": d,
                  "P": None if post is None else 100 - post,
                  "A": None if f is None else 100 * (1 - f)}
    value = None if any(v is None for v in components.values()) else sum(components.values()) / 4
    statuses = attempts.get("attempt_status", pd.Series(dtype=str))
    return {"MERI_KDPA": value, **components, "Bpre": bp, "Bpost": post,
            "Fua": None if f is None else 100 * f,
            "eligible_events": len(eligible), "retained_events": int(events.physiology_filter.eq("kept").sum()),
            "release_attempts": int(statuses.eq("release").sum()),
            "unassociated_attempts": int(statuses.eq("unassociated_collective").sum()),
            "missing_pre_burden_events": missing_pre,
            "median_event_confidence": float(eligible.meri_confidence.median()) if len(eligible) else None,
            "mean_release_duration_sec": float(eligible.duration_sec.mean()) if len(eligible) else None}

