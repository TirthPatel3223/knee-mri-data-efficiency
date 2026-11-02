"""Label extraction v2: re-read every radiology report with a stricter, rule-based prompt.

    python labeling/extract_labels_v2.py --ids labeling/dev_ids.txt --model qwen3:14b --out labeling/runs/dev_qwen3_14b.jsonl
    python labeling/extract_labels_v2.py --model qwen3:14b --out labeling/runs/full_qwen3_14b_v2.jsonl

Derived from T:\\Kaggle_Knee_Competition\\src\\stage2_extract_labels.py (same output
schema, resumable JSONL), with the prompt rewritten after two manual audits of the v1
labels (T:\\Group_Project\\label_audit\\). v1 already *stated* the rubric, yet the
model still:
  * missed GROUP statements ("cruciate ligaments intact", "articular cartilage
    normal", "bones unremarkable") and left those findings "na";
  * called anything *mentioned* positive, ignoring thresholds: grade I-II sprains,
    intrasubstance / Stoller 2 meniscal signal, ICRS 1-2 cartilage, a 1 cm Baker's cyst;
  * occasionally attributed one structure's finding to another (PCL tear -> MCL).
So v2 spells out a decision procedure, a per-finding POS/NEG/UNCERTAIN table with the
grading scales mapped onto the rubric, a multilingual vocabulary, and fixed probability
anchors. It also makes the model quote its evidence BEFORE it decides (schema property
order), which forces each label to be grounded in the text.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.request
from pathlib import Path

DATA = Path(r"T:\Kaggle_Knee_Competition\rsna-knee-abnormality-detection")
OLLAMA = "http://localhost:11434/api/chat"
PROMPT_VERSION = "v2.1"

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
KEY = {"ACL": "acl", "MCL": "mcl", "Medial Meniscus": "mm",
       "Lateral Meniscus": "lm", "Medial OA": "moa", "Lateral OA": "loa",
       "PF OA": "pfoa", "Effusion": "eff", "Synovitis": "syn",
       "Baker's": "bak", "Contusion": "con", "Fracture": "frx"}

SYSTEM = """\
You are a musculoskeletal radiologist labelling knee MRI reports for a research dataset. \
For each report you grade TWELVE findings exactly as an expert panel would grade the \
IMAGES under a strict rubric. Reports may be in English, Spanish, Turkish, Greek, \
Bulgarian, Croatian/Bosnian/Serbian, German, Dutch or French.

The report is evidence, not the answer. Clinical reports mention many things the \
rubric grades NEGATIVE (low-grade sprains, meniscal degeneration, mild cartilage wear, \
small effusions, small cysts). Being MENTIONED is not being POSITIVE. Borderline or \
ambiguous findings lean NEGATIVE: the rubric favours specificity.

=== DECISION PROCEDURE (follow for every report) ===
1. Read the WHOLE report: findings and conclusion/impression. If they disagree, the \
more specific description wins.
2. GROUP STATEMENTS apply to every member of the group:
   - "cruciate ligaments intact/normal" -> ACL neg.   "collateral ligaments normal" -> MCL neg.
   - "ligaments / ligamentous structures normal" -> ACL neg AND MCL neg.
   - "menisci normal/intact" -> both menisci neg.
   - "articular cartilage normal/preserved" -> all three OA neg; \
"femorotibial (FT) cartilage preserved" -> Medial OA and Lateral OA neg; \
"patella / patellar cartilage / PF joint normal" -> PF OA neg.
   - "bones / bone marrow normal, no marrow edema" -> Contusion neg AND Fracture neg.
   - "no joint fluid / no effusion" -> Effusion neg.   "popliteal fossa unremarkable" -> Baker's neg.
   - Abbreviations "b.o.", "б.о.", "o.B.", "s.p.", "WNL", "unremarkable", "without \
particularities", "sin alteraciones", "uredan/urednog", "нормален", "φυσιολογικ-", \
"normaldir", "regelrecht", "intakt" all mean NORMAL.
3. GLOBAL NORMAL STATEMENTS ("no other abnormality", "otherwise normal", "verder geen \
afwijkingen", "Conclusion: normal", "no internal derangement", "no significant \
abnormality") make every finding that is not described elsewhere NEG (p 0.05).
4. POSITIVES-ONLY REPORTS. Some reports (often short Spanish templates "Técnica ... \
Resultados ... Impresión ...") list only the diagnoses. A finding they do not mention \
is "na", and its p should be LOW (0.05-0.15): radiologists list significant findings.
5. EMPTY / TRUNCATED REPORTS (no findings text) -> every finding "na", p 0.05-0.15.
6. HEDGING sets the probability:
   - definite / "compatible with" / "consistent with" -> p 0.85-0.97
   - "probable", "likely", "suggestive of", "in favor of", "suspected" -> p 0.65-0.75
   - "possible", "may represent", "R/O", "cannot be excluded", "DD", "moguće", "podría" -> p 0.35-0.45
7. Never transfer a finding to a neighbouring structure. PCL ("LCP", "HKB", "stražnji \
križni", "οπίσθιος χιαστός", "задна кръстна", "arka çapraz") is NOT the ACL. LCL/FCL, \
medial patellar retinaculum, MPFL, posterior oblique ligament and pes anserinus are NOT \
the MCL. Meniscal / parameniscal / ganglion cysts are NOT a Baker's cyst. Prepatellar \
or pes anserine bursitis is NOT joint effusion. Hoffa fat-pad edema is NOT synovitis.
8. Two knees in one report ("[BILATERAL NOTE ...]", "left knee ... right knee"): a \
finding present in BOTH knees uses the normal p; present in only ONE knee -> p 0.50.
9. CONSISTENCY: v="pos" if and only if p >= 0.50. v="neg" when the report addresses \
the finding and p < 0.50. v="na" only for genuine silence (then p < 0.50 as well).

=== VOCABULARY ===
medial = medial, interno, interna, iç, έσω, медиален, medijalni, Innen-, innen, mediaal, interne.
lateral = lateral, externo, externa, dış, έξω, латерален, lateralni, Außen-, aussen, lateraal, externe.
ACL = LCA, VKB, voorste kruisband, ön çapraz bağ, πρόσθιος χιαστός, предна кръстна връзка, prednji križni ligament.
MCL = LCM, LLI, LCT (ligamento colateral tibial/medial), iç yan bağ, medial kollateral, \
έσω πλάγιος, медиален колатерален, medijalni kolateralni, Innenband, mediaal collateraal, LLI/LCI.
meniscus = menisco, menisküs, μηνίσκος, менискус, menisk, Meniskus, ménisque.
cartilage = cartílago, kıkırdak/kartilaj, χόνδρος, хрущял, hrskavica, Knorpel, kraakbeen, cartilage.
effusion = derrame, efusión, sıvı artışı / efüzyon, συλλογή υγρού / ύδραρθρο, излив, izljev, Erguss, hydrops, épanchement.
synovitis = sinovitis, sinovit, υμενίτιδα, синовит, sinovitis, Synovialitis/Synovitis, synovitis.
fracture = fractura, kırık, κάταγμα, фрактура, fraktura/prijelom, Fraktur, fractuur, fracture.
bone bruise = contusión ósea, kemik kontüzyonu, οστικό οίδημα/bone bruise, контузия/контузионен едем, \
kontuzija/kontuzijski edem, Knochenprellung/Bone bruise, botkneuzing, contusion osseuse.

=== FINDINGS (POS / NEG / UNCERTAIN, with p anchors) ===
acl - ACL tear.
  POS: complete / full-thickness tear, rupture, discontinuity, "empty notch", ACL not \
visualised, subtotal or high-grade partial tear (>50% of fibres), chronic complete tear, \
torn reconstruction graft.
  NEG: intact, normal, continuous; mucoid / myxoid / cystic degeneration; ganglion; \
thickening or raised signal without discontinuity; grade 1 sprain; low-grade or \
interstitial / intrasubstance partial tear; intact reconstruction graft.
  UNCERTAIN: "partial tear" with no grade -> p 0.40; grade 2 injury -> p 0.35.
mcl - MCL tear (acute, high-grade).
  POS: complete tear, grade III, rupture, avulsion, high-grade partial tear with torn fibres.
  NEG: intact; grade I sprain; periligamentous edema or "infiltration" only; thickening or \
degenerative signal; chronic / old / remote / healed / "sequela", "secuela", "stanje \
nakon", "status after", "Z.n." changes; "distended" only.
  UNCERTAIN: grade II or "partial tear" with no further detail -> p 0.40; \
grade I-II -> p 0.20.
mm / lm - medial / lateral meniscus tear (use the vocabulary above for side).
  POS: any TEAR / rupture / "rotura", "ruptura", "yırtık", "ρήξη", "руптура/скъсване", \
"Riss/Rissbildung", "scheur" (horizontal, radial, vertical, longitudinal, oblique, \
complex, flap, bucket-handle, root, parrot-beak, degenerative tear); signal reaching the \
articular surface; grade 3 (incl. "grade III", "3-та степен"); displaced fragment; \
truncated, diminutive, macerated or absent meniscus.
  NEG: normal / intact; degeneration, degenerative or mucoid signal; intrasubstance \
signal; grade 1 or grade 2 signal INCLUDING Stoller 2a / 2b / 2c ("2а степен", "2c"); \
signal that "does not reach the surface"; "may correspond to intrasubstance rupture"; \
meniscal cyst, extrusion or a DISCOID meniscus WITHOUT a stated tear (discoid is a shape \
variant, not a tear); post-meniscectomy remnant without a new tear (p 0.30).
  UNCERTAIN: "suspected / probable tear" -> p 0.65; "possible tear" -> p 0.40.
moa / loa / pfoa - osteoarthritis of the medial tibiofemoral, lateral tibiofemoral and \
patellofemoral compartments.
  COMPARTMENT MAP (apply BEFORE grading):
    PF OA ONLY <- patella / patellar / rótula / rotuliano / επιγονατίδα / пателa / patela / \
Patella / retropatellar / knieschijf, and trochlea / tróclea / troclear / femoral groove / \
τροχιλία / трохлея / trohlea / Trochlea / Gleitlager / femoropatellar / patelofemoral.
    Medial OA ONLY <- medial femoral condyle and medial tibial plateau (medial tibiofemoral \
compartment, "femorotibial medial", "έσω διαμέρισμα", "медиален FT").
    Lateral OA ONLY <- lateral femoral condyle and lateral tibial plateau.
    TRAP: "medial patellar facet", "lateral trochlea", "medial facet of the patella", \
"vertiente medial de la tróclea" are PATELLOFEMORAL. The words medial/lateral there do \
NOT make it Medial OA or Lateral OA.
  POS: high-grade cartilage loss (>50% thickness) or full-thickness loss / defect / \
denudation over a moderate-or-large area (about 1 cm or more); ICRS 3-4; Outerbridge \
III-IV; grade 3-4 chondropathy / chondromalacia; "advanced", "severe", "moderate-to-severe" \
OA / arthrosis / gonarthrosis of that compartment.
  NEG: normal / preserved cartilage; low-grade chondrosis (<50% thickness); ICRS 1-2; \
Outerbridge I-II; grade 1-2 chondromalacia; mild or superficial thinning, fibrillation; \
osteophytes alone; subchondral cysts alone.
  UNCERTAIN: "OA / arthrosis of <compartment>" with no grade -> p 0.65; "tricompartmental \
OA" with no detail -> p 0.60 for each; "gonarthrosis" with no compartment -> medial p 0.55, \
lateral p 0.30, PF p 0.35; "early / incipient / beginning OA" ("начална", "αρχόμενη", \
"beginnend", "inicial") -> p 0.35; chondropathy / chondromalacia / "condropatía" with no \
grade -> p 0.40; a small focal full-thickness defect (<1 cm, e.g. "4 mm", "small areas") \
-> p 0.40; fissures to bone over a small area -> p 0.45.
eff - joint effusion.
  POS: moderate, large, massive, marked, significant, important, extensive, distending, \
"forse hydrops", "ικανή", "умерен/голям", "umjeren/opsežan", "orta derecede", "deutlich", \
"importante"; moderate or large hemarthrosis or lipohemarthrosis.
  NEG: none, physiologic, trace, minimal, small, mild, slight, little ("leve", "pequeño", \
"escaso", "μικρή", "малък", "manji", "hafif", "gering", "klein", "beperkt", "minime").
  UNCERTAIN: effusion / hemarthrosis with no size -> p 0.55; "mild-to-moderate" or \
"small-to-moderate" -> p 0.45.
syn - synovitis.
  POS: synovitis; synovial thickening, hypertrophy or proliferation; PVNS; lipoma \
arborescens (p 0.60).
  NEG: "no synovitis", "no synovial thickening", "no signs of inflammatory arthritis".
  SILENT -> "na": p 0.35 if a moderate or large effusion is described, otherwise p 0.10. \
Plicae alone, suprapatellar bursitis alone (p 0.30) and Hoffa edema are not synovitis.
bak - Baker's (popliteal / gastrocnemius-semimembranosus) cyst, MODERATE or LARGE only.
  POS: largest dimension >= 3 cm, or described as moderate / large; ruptured or leaking \
Baker's cyst (p 0.80).
  NEG: "no Baker's / popliteal cyst"; small, tiny or minimal cyst; largest dimension < 2 cm \
(e.g. "10x5 mm", "1.3 cm"); cysts in other locations (meniscal, ganglion, lateral recess).
  UNCERTAIN: 2-3 cm -> p 0.50; size not given -> p 0.55.
con - bone contusion: trauma-related marrow edema WITHOUT a discrete fracture line there.
  POS: bone bruise, bone contusion, post-traumatic or impaction marrow edema, typical \
pivot-shift (lateral femoral condyle + posterolateral tibial plateau) or patellar-dislocation \
(medial patella + lateral femoral condyle) pattern.
  NEG: no marrow edema; bones normal; edema explained by OA (subchondral edema under \
cartilage loss or under an osteochondral/chondral defect, "reactive", "degenerative"), \
insufficiency fracture, osteonecrosis, tumour, infection, stress reaction, OCD; edema \
located only around a fracture (then judge Fracture).
  UNCERTAIN: "bone marrow edema" with no cause or context -> p 0.45; small edema next to \
fat-pad impingement -> p 0.25.
frx - ACUTE fracture.
  POS: acute fracture line, cortical break, impaction / depression fracture, \
osteochondral fracture, avulsion fracture, trabecular or occult fracture (p 0.70), \
non-consolidated fracture.
  NEG: "no fracture"; healed, consolidated, old, "status after" fracture, post-fixation; \
bone contusion without a fracture line; osteochondral lesion / OCD without a fracture.
  UNCERTAIN: "R/O fracture", "cannot exclude", "suspected impaction" -> p 0.40; \
subchondral insufficiency fracture (SIF / SIFK) -> p 0.55; microtrabecular fracture -> p 0.50.

=== EXAMPLES (illustrative, not from this dataset) ===
"Cruciate and collateral ligaments intact. Grade 2a lesion of the posterior horn of the \
medial meniscus. Small effusion." -> acl neg 0.03, mcl neg 0.03, mm neg 0.15, lm na 0.10, \
eff neg 0.10.
"Grade II MCL sprain. Chondromalacia patellae ICRS 2. 12 mm popliteal cyst." -> \
mcl neg 0.40, pfoa neg 0.15, bak neg 0.15.
"Resultados: Rotura de menisco interno. Derrame." -> mm pos 0.92, lm na 0.10, eff pos 0.55, \
acl na 0.05.
"PCL complete tear. ACL and collateral ligaments without alterations." -> acl neg 0.03, mcl neg 0.03.
"Marked bone marrow edema of the lateral femoral condyle and posterolateral tibial plateau \
after twisting injury, no fracture line. Moderate effusion." -> con pos 0.92, frx neg 0.05, eff pos 0.85.

=== OUTPUT ===
For every finding return:
  e : the shortest decisive quote ABOUT THIS FINDING'S OWN STRUCTURE (max 8 words, \
original language), or "silent" if nothing in the report concerns it, or \
"group: <quote>" when a group/global statement decided it. Never quote text about a \
different structure.
  v : "pos" | "neg" | "na"
  p : probability (0.00-1.00) that the expert IMAGE grade is positive, using the anchors above.
Return only the JSON object."""

SCHEMA = {
    "type": "object",
    "properties": {
        k: {
            "type": "object",
            # property order matters: evidence first, then the decision
            "properties": {
                "e": {"type": "string"},
                "v": {"type": "string", "enum": ["pos", "neg", "na"]},
                "p": {"type": "number"},
            },
            "required": ["e", "v", "p"],
        }
        for k in KEY.values()
    },
    "required": list(KEY.values()),
}


# Ablation variant: identical rules, but no evidence quote (about half the output tokens).
SYSTEM_PLAIN = SYSTEM.split("=== OUTPUT ===")[0] + """=== OUTPUT ===
For every finding return:
  v : "pos" | "neg" | "na"
  p : probability (0.00-1.00) that the expert IMAGE grade is positive, using the anchors above.
Return only the JSON object."""

SCHEMA_PLAIN = {
    "type": "object",
    "properties": {
        k: {
            "type": "object",
            "properties": {
                "v": {"type": "string", "enum": ["pos", "neg", "na"]},
                "p": {"type": "number"},
            },
            "required": ["v", "p"],
        }
        for k in KEY.values()
    },
    "required": list(KEY.values()),
}


def build_request(report: str, args) -> dict:
    return {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PLAIN if args.no_evidence else SYSTEM},
            {"role": "user", "content": "REPORT:\n" + report.strip()},
        ],
        "stream": False,
        "think": args.think,
        "format": SCHEMA_PLAIN if args.no_evidence else SCHEMA,
        "keep_alive": "30m",
        # explicit sampling: qwen3.5 ships with temperature 1 and presence_penalty 1.5,
        # both of which are wrong for deterministic structured extraction
        "options": {"temperature": 0.0, "top_k": 1, "top_p": 1.0,
                    "presence_penalty": 0.0, "repeat_penalty": 1.0,
                    "num_ctx": args.num_ctx, "num_predict": args.num_predict},
    }


def clean_cell(cell: dict) -> tuple[str | None, float | None, str | None]:
    """Enforce the prompt's consistency rule (v=pos iff p>=0.5) on the model output."""
    v, p, e = cell.get("v"), cell.get("p"), cell.get("e")
    if p is not None:
        try:
            p = float(p)
        except (TypeError, ValueError):
            p = None
    if p is not None:
        p = min(max(p, 0.0), 1.0)
        if v == "pos" and p < 0.5:
            v = "neg"
        elif v == "neg" and p >= 0.5:
            v = "pos"
        elif v == "na" and p >= 0.5:
            # "silent" but confident is self-contradictory; keep the stance, cap belief
            p = 0.45
    return v, p, e


_lock = threading.Lock()


def process(row: dict, args, out_handle, stats: dict) -> None:
    sid = row["StudyInstanceUID"]
    t0 = time.perf_counter()
    rec = {"StudyInstanceUID": sid, "model": args.model,
           "prompt": PROMPT_VERSION + ("-noev" if args.no_evidence else "")}
    for attempt in range(2):
        try:
            req = urllib.request.Request(
                OLLAMA, data=json.dumps(build_request(row["Report"], args)).encode("utf-8"),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            parsed = json.loads(body["message"]["content"])
            for short in KEY.values():
                v, p, e = clean_cell(parsed.get(short) or {})
                rec[short + "_v"], rec[short + "_p"], rec[short + "_e"] = v, p, e
            missing = [k for k in KEY.values() if rec[k + "_v"] is None or rec[k + "_p"] is None]
            if missing:
                raise ValueError("missing cells: %s" % missing)
            rec["eval_count"] = body.get("eval_count")
            rec["prompt_eval_count"] = body.get("prompt_eval_count")
            rec["ok"] = True
            rec.pop("error", None)
            break
        except Exception as e:
            rec["ok"] = False
            rec["error"] = "%s: %s" % (type(e).__name__, e)
    rec["secs"] = round(time.perf_counter() - t0, 2)

    with _lock:
        out_handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_handle.flush()
        stats["done"] += 1
        stats["failed"] += 0 if rec["ok"] else 1
        n, el = stats["done"], time.perf_counter() - stats["t0"]
        if n % args.log_every == 0 or n == stats["total"]:
            rate = n / el
            print("[v2] %5d/%d  %5.3f rep/s  elapsed %6.1fm  eta %6.1fm  failed %d"
                  % (n, stats["total"], rate, el / 60,
                     (stats["total"] - n) / max(rate, 1e-9) / 60, stats["failed"]), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:14b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", default=None, help="file with one StudyInstanceUID per line")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--num-predict", type=int, default=1200)
    ap.add_argument("--think", action="store_true", help="enable model reasoning (3-5x slower)")
    ap.add_argument("--no-evidence", action="store_true",
                    help="ablation: drop the evidence quote field (same rules, fewer tokens)")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--dump-prompt", action="store_true")
    args = ap.parse_args()

    if args.dump_prompt:
        print(SYSTEM_PLAIN if args.no_evidence else SYSTEM)
        return 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(10_000_000)
    with open(DATA / "train.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.ids:
        keep = set(Path(args.ids).read_text(encoding="utf-8").split())
        rows = [r for r in rows if r["StudyInstanceUID"] in keep]

    # resume: skip studies that already have a successful record
    done = set()
    if out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            try:
                d = json.loads(line)
                if d.get("ok"):
                    done.add(d["StudyInstanceUID"])
            except Exception:
                pass
        print("[v2] resuming: %d already done" % len(done))
    todo = [r for r in rows if r["StudyInstanceUID"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print("[v2] prompt=%s model=%s think=%s workers=%d num_ctx=%d -> %s"
          % (PROMPT_VERSION, args.model, args.think, args.workers, args.num_ctx, out_path))
    print("[v2] %d reports to process" % len(todo))
    if not todo:
        return 0

    stats = {"done": 0, "failed": 0, "total": len(todo), "t0": time.perf_counter()}
    with open(out_path, "a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(lambda r: process(r, args, out, stats), todo))
    el = time.perf_counter() - stats["t0"]
    print("[v2] finished %d in %.1f min (%.3f rep/s), %d failed"
          % (stats["done"], el / 60, stats["done"] / el, stats["failed"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
