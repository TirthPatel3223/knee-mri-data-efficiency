"""Deterministic guardrails applied to v2 LLM labels after extraction.

The only one so far: COMPARTMENT LEAK. Even with an explicit warning in the prompt,
qwen3:14b sometimes grades a patellar / trochlear cartilage defect as Medial or Lateral
OA because the text says "medial patellar facet" or "lateral trochlea". The v2 prompt
makes the model quote its evidence, so the mistake is detectable: a Medial/Lateral OA
positive whose ONLY evidence is patellofemoral anatomy is demoted to negative.
"""
from __future__ import annotations

import re

# patellofemoral anatomy, multilingual (patella, trochlea and their variants)
PF = re.compile(
    r"patel|rótul|rotul|trócl|trocl|trochl|επιγονατ|τροχιλ|пател|трохле|трохл|patela|"
    r"trohle|gleitlager|retropatell|knieschijf|femoropatell|patelofemoral|patellofemoral",
    re.I)
# tibiofemoral anatomy: condyles, plateaus, femorotibial compartment
TF = re.compile(
    r"condyl|cóndil|condil|kondil|kondyl|κόνδυλ|κονδύλ|кондил|plateau|platillo|meseta|"
    r"plato|πλατ|κνημιαί|κνηµιαί|femorotib|tibiofem|femoro-tib|\bFT\b|tibia|tibial|"
    r"μηριαί|µηριαί|бедрен|фемур|femur|femoral cond|Tibiaplateau|Femurkondyl",
    re.I)


def compartment_guard(rec: dict) -> list[str]:
    """Demote moa/loa positives justified only by PF anatomy. Returns list of changes."""
    changed = []
    for k in ("moa", "loa"):
        if rec.get(k + "_v") != "pos":
            continue
        e = rec.get(k + "_e") or ""
        if PF.search(e) and not TF.search(e):
            rec[k + "_v"] = "neg"
            rec[k + "_p"] = min(float(rec[k + "_p"]), 0.15)
            rec[k + "_guard"] = "pf_only_evidence"
            changed.append(k)
    return changed


def apply_all(labels: dict) -> dict:
    """In-place over {study: record}; returns {study: [changed keys]}."""
    log = {}
    for sid, rec in labels.items():
        c = compartment_guard(rec)
        if c:
            log[sid] = c
    return log
