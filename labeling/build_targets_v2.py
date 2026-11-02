"""Build T:\\Group_Project\\data\\targets_v2.parquet from a v2 label run.

    python labeling/build_targets_v2.py --labels labeling/runs/full_v21_qwen3_14b.jsonl

Same schema as T:\\Kaggle_Knee_Competition\\work\\targets.parquet (which stays
untouched). Every non-label column -- folds, scanner group, gold labels g_*, is_gold,
vendor, field, sex, counts -- is copied from v1 so the splits in PROJECT_PLAN.md stay
identical. Only s_* (stance), t_* (soft target) and w_* (weight) are replaced, using
the v1 conventions: t = clip(p, 0.02, 0.98); w = 1.0 if addressed, 0.5 if 'na'.
The postprocess guardrails are applied first. A stance-transition report vs v1 is printed.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from postprocess import apply_all  # noqa: E402

V1_TARGETS = Path(r"T:\Kaggle_Knee_Competition\work\targets.parquet")
OUT = Path(r"T:\Group_Project\data\targets_v2.parquet")
K = ['acl', 'mcl', 'mm', 'lm', 'moa', 'loa', 'pfoa', 'eff', 'syn', 'bak', 'con', 'frx']


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--w-na", type=float, default=0.5)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    lab = {}
    for line in open(args.labels, encoding="utf-8"):
        d = json.loads(line)
        if d.get("ok"):
            lab[d["StudyInstanceUID"]] = d          # last successful record wins
    guard = apply_all(lab)
    print("[targets_v2] %d labelled studies; guardrail demoted %d cells in %d studies"
          % (len(lab), sum(map(len, guard.values())), len(guard)))

    v1 = pl.read_parquet(V1_TARGETS)
    missing = set(v1["study_dir"]) - set(lab)
    assert not missing, "%d studies have no v2 label, rerun extraction to resume" % len(missing)

    recs = []
    for sid in v1["study_dir"]:
        d, row = lab[sid], {"study_dir": sid}
        for k in K:
            v, p = d[k + "_v"], d[k + "_p"]
            row["s_" + k] = v
            row["t_" + k] = float(np.clip(p, 0.02, 0.98))
            row["w_" + k] = args.w_na if v == "na" else 1.0
        recs.append(row)
    new = pl.DataFrame(recs)

    label_cols = [c for c in v1.columns if c[:2] in ("s_", "t_", "w_")]
    keep = v1.drop(label_cols)
    out = keep.join(new, on="study_dir", how="left").select(v1.columns)   # same column order
    assert out.shape == v1.shape and out["study_dir"].to_list() == v1["study_dir"].to_list()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(args.out)
    print("[targets_v2] wrote %s  %s" % (args.out, out.shape))

    print("\nstance transitions v1 -> v2 (non-gold studies)")
    ng = ~v1["is_gold"].to_numpy()
    print("%-5s %8s %8s %8s %8s %8s %8s | %s" % ("", "pos>neg", "pos>na", "neg>pos", "na>pos",
                                                 "na>neg", "neg>na", "pos v1 -> v2"))
    for k in K:
        a, b = v1["s_" + k].to_numpy()[ng], out["s_" + k].to_numpy()[ng]
        c = Counter(zip(a, b))
        print("%-5s %8d %8d %8d %8d %8d %8d | %d -> %d"
              % (k, c[("pos", "neg")], c[("pos", "na")], c[("neg", "pos")], c[("na", "pos")],
                 c[("na", "neg")], c[("neg", "na")], (a == "pos").sum(), (b == "pos").sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
