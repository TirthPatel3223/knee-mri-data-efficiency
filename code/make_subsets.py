"""T1.2: nested training subsets. For each (rotation r, seed s) the pool is permuted
with default_rng(1000*r + s) and every size N takes a prefix, so 65 c 130 c ... c 2600.

    python code/make_subsets.py

Writes manifests/subsets_r{r}_s{s}.json and results/tables/subset_label_counts.csv.
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl

from common import KEYS, LABELS, MANIFESTS, RESULTS, ROTATIONS, SEEDS, SIZES, load_targets


def main() -> int:
    tg = load_targets()
    rot = json.loads((MANIFESTS / "rotations.json").read_text())
    stance = {k: dict(zip(tg["study_dir"], tg["s_" + k])) for k in KEYS}
    rows = []
    for r in ROTATIONS:
        pool = rot[str(r)]["pool"]                    # sorted, so the permutation is stable
        for s in SEEDS:
            perm = np.random.default_rng(1000 * r + s).permutation(len(pool))
            sub = {str(n): [pool[i] for i in perm[:min(n, len(pool))]] for n in SIZES}
            for a, b in zip(SIZES, SIZES[1:]):
                assert set(sub[str(a)]) <= set(sub[str(b)])
            assert set(sub[str(SIZES[-1])]) <= set(pool)
            (MANIFESTS / ("subsets_r%d_s%d.json" % (r, s))).write_text(json.dumps(sub), encoding="utf-8")
            for n in SIZES:
                ids = sub[str(n)]
                for k, lab in zip(KEYS, LABELS):
                    st = [stance[k][i] for i in ids]
                    rows.append({"rotation": r, "seed": s, "n": n, "finding": lab,
                                 "pos": st.count("pos"), "neg": st.count("neg"), "na": st.count("na")})
    df = pl.DataFrame(rows)
    df.write_csv(RESULTS / "tables" / "subset_label_counts.csv")
    few = df.filter(pl.col("pos") < 5)
    print("wrote %d subset manifests; %d (r,s,N,finding) cells have < 5 training positives"
          % (len(ROTATIONS) * len(SEEDS), len(few)))
    print(df.filter((pl.col("rotation") == 0) & (pl.col("seed") == 0))
          .pivot(on="finding", index="n", values="pos"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
