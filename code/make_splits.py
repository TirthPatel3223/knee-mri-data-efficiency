"""T1.1: rotation manifests. test = fold r, val = fold (r+1)%5, pool = the other three.

    python code/make_splits.py

Writes manifests/rotations.json and results/tables/rotation_summary.csv. The 58 gold
studies (fold -1) are in no split; every final model scores them separately.
"""
from __future__ import annotations

import json

import polars as pl

from common import GP, KEYS, MANIFESTS, RESULTS, ROTATIONS, load_targets


def main() -> int:
    tg = load_targets()
    ng = tg.filter(pl.col("fold") >= 0)
    rot, rows = {}, []
    for r in ROTATIONS:
        te, va = r, (r + 1) % 5
        sp = {
            "test": sorted(ng.filter(pl.col("fold") == te)["study_dir"].to_list()),
            "val": sorted(ng.filter(pl.col("fold") == va)["study_dir"].to_list()),
            "pool": sorted(ng.filter(~pl.col("fold").is_in([te, va]))["study_dir"].to_list()),
        }
        a, b, c = (set(sp[k]) for k in ("test", "val", "pool"))
        assert not (a & b or a & c or b & c), "splits overlap"
        assert a | b | c == set(ng["study_dir"]), "splits do not cover the non-gold studies"
        assert not (a | b | c) & set(tg.filter(pl.col("is_gold"))["study_dir"]), "gold leaked"
        assert 2609 <= len(c) <= 2610, len(c)
        rot[str(r)] = sp
        for name, ids in sp.items():
            d = tg.filter(pl.col("study_dir").is_in(ids))
            row = {"rotation": r, "split": name, "n": len(d)}
            for v, cnt in d["vendor"].value_counts().iter_rows():
                row["vendor_" + str(v)] = cnt
            for f, cnt in d["field"].cast(pl.Utf8).value_counts().iter_rows():
                row["field_" + str(f)] = cnt
            for k in KEYS:
                for st in ("pos", "neg", "na"):
                    row["%s_%s" % (k, st)] = int((d["s_" + k] == st).sum())
            rows.append(row)
        print("rotation %d  test %d  val %d  pool %d" % (r, len(a), len(b), len(c)))

    MANIFESTS.mkdir(parents=True, exist_ok=True)
    (MANIFESTS / "rotations.json").write_text(json.dumps(rot), encoding="utf-8")
    (RESULTS / "tables").mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).fill_null(0).write_csv(RESULTS / "tables" / "rotation_summary.csv")
    print("wrote", MANIFESTS / "rotations.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
