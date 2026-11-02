"""Score label files against the two rulers we have.

    python labeling/eval_labels.py --make-ids                 # writes labeling/dev_ids.txt
    python labeling/eval_labels.py labeling/runs/dev_qwen3_14b.jsonl [more.jsonl ...]

TEST  - gold 58: expert image labels. Macro ROC-AUC of p. Never used to tune the prompt.
DEV   - the 168 hand-audited cells from label_audit/ (84 'na' + 84 'pos' v1 labels).
        Expected stance: audited-correct v1 cells keep their v1 stance; cells judged
        error / over-call / arguable-negative should become 'neg'. Arguable 'pos'
        cells are reported but not scored. The prompt was written FROM these cells,
        so DEV is optimistic by construction; gold-58 is the honest number.
The v1 labels (T:\\Kaggle_Knee_Competition\\work\\labels_qwen3_14b.jsonl) are always
scored alongside, restricted to the same studies, so every comparison is paired.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from postprocess import apply_all  # noqa: E402

DATA = Path(r"T:\Kaggle_Knee_Competition\rsna-knee-abnormality-detection")
V1 = Path(r"T:\Kaggle_Knee_Competition\work\labels_qwen3_14b.jsonl")
AUDIT = Path(r"T:\Group_Project\label_audit")
HERE = Path(__file__).resolve().parent

K = ['acl', 'mcl', 'mm', 'lm', 'moa', 'loa', 'pfoa', 'eff', 'syn', 'bak', 'con', 'frx']
NAME = dict(zip(K, ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
                    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
                    "Contusion", "Fracture"]))


def load_labels(path) -> dict:
    out = {}
    for line in open(path, encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("ok"):
            out[d["StudyInstanceUID"]] = d
    return out


def load_train():
    csv.field_size_limit(10 ** 8)
    rows = list(csv.DictReader(open(DATA / "train.csv", encoding="utf-8")))
    gold = {r["StudyInstanceUID"]: r for r in rows if r["ACL"].strip() != ""}
    return rows, gold


def audit_cells(v1: dict, gold: dict) -> list[tuple]:
    """Reconstruct the audited (study, finding, expected) cells.

    Re-runs the exact sampling of label_audit/sample_na.py (seed 7) and
    sample_pos.py (seed 11) so ids 01..84 map back to full StudyInstanceUIDs.
    """
    cells = []
    for kind, seed, csvname in (("na", 7, "na_audit.csv"), ("pos", 11, "pos_audit.csv")):
        rng = random.Random(seed)
        used, order = set(), []
        for k in K:
            cand = [s for s in v1 if v1[s][k + "_v"] == kind and s not in used and s not in gold]
            rng.shuffle(cand)
            for s in cand[:7]:
                used.add(s)
                order.append((k, s))
        verdicts = list(csv.DictReader(open(AUDIT / csvname, encoding="utf-8")))
        assert len(verdicts) == len(order) == 84
        for (k, s), row in zip(order, verdicts):
            assert NAME[k] == row["finding"], (k, row)
            verdict = row["verdict"]
            if kind == "na":
                exp = "na" if verdict == "correct" else "neg"
            else:
                exp = {"correct": "pos", "over-call": "neg", "error": "neg"}.get(verdict)
            cells.append((s, k, kind, verdict, exp))
    return cells


def gold_report(labels: dict, gold: dict, sids: list) -> tuple[float, dict]:
    per = {}
    for k in K:
        y = np.array([int(gold[s][NAME[k]].strip() == "1") for s in sids])
        p = np.array([float(labels[s][k + "_p"]) for s in sids])
        per[k] = roc_auc_score(y, p)
    return float(np.mean(list(per.values()))), per


def dev_report(labels: dict, cells: list) -> dict:
    res = {"na_keep": [0, 0], "na_fix": [0, 0], "pos_keep": [0, 0], "pos_fix": [0, 0],
           "na_became_pos": 0}
    for s, k, kind, verdict, exp in cells:
        if s not in labels or exp is None:
            continue
        got = labels[s][k + "_v"]
        bucket = ("na_" if kind == "na" else "pos_") + ("keep" if verdict == "correct" else "fix")
        ok = (got == exp) or (kind == "na" and verdict == "correct" and got == "neg")
        res[bucket][0] += int(ok)
        res[bucket][1] += 1
        if kind == "na" and verdict == "correct" and got == "pos":
            res["na_became_pos"] += 1
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--make-ids", action="store_true")
    ap.add_argument("--post", action="store_true", help="also score with postprocess guardrails")
    args = ap.parse_args()

    rows, gold = load_train()
    v1 = load_labels(V1)
    cells = audit_cells(v1, gold)

    if args.make_ids:
        ids = sorted(set(gold) | {c[0] for c in cells})
        (HERE / "dev_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
        print("wrote %d ids (%d gold + %d audited studies)"
              % (len(ids), len(gold), len({c[0] for c in cells})))
        return 0

    runs = [("v1 (qwen3:14b, old prompt)", v1)]
    for f in args.files:
        runs.append((Path(f).stem, load_labels(f)))
        if args.post:
            lab = load_labels(f)
            log = apply_all(lab)
            print("guardrail on %s: %d cells demoted" % (Path(f).stem, sum(map(len, log.values()))))
            runs.append((Path(f).stem + "+guard", lab))
    common_gold = sorted(set(gold).intersection(*[set(l) for _, l in runs]))
    print("gold studies scored by every run: %d / %d\n" % (len(common_gold), len(gold)))

    table = {}
    for name, lab in runs:
        macro, per = gold_report(lab, gold, common_gold)
        table[name] = (macro, per, dev_report(lab, cells))

    w = max(len(n) for n in table) + 2
    print("TEST: gold-58 ROC-AUC (expert image labels)")
    print("%-18s" % "finding" + "".join("%*s" % (w, n[:w - 2]) for n in table))
    for k in K:
        print("%-18s" % NAME[k] + "".join("%*.3f" % (w, table[n][1][k]) for n in table))
    print("%-18s" % "MACRO" + "".join("%*.4f" % (w, table[n][0]) for n in table))
    print("  (58 studies resolve ~0.02 macro; per-finding CIs are ~+-0.12)\n")

    print("DEV: hand-audited cells (agreement with audit verdict)")
    for n, (_, _, d) in table.items():
        f = lambda b: "%d/%d" % tuple(d[b])
        print("  %-*s na-keep %-6s na-fix %-6s pos-keep %-6s pos-fix %-6s  (true-silent na turned pos: %d)"
              % (w, n, f("na_keep"), f("na_fix"), f("pos_keep"), f("pos_fix"), d["na_became_pos"]))
    print("  na-keep : silent cells that should stay na (neg also accepted)")
    print("  na-fix  : na cells the audit said should be neg")
    print("  pos-keep: correct positives that must stay pos")
    print("  pos-fix : over-called / misread positives that should become neg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
