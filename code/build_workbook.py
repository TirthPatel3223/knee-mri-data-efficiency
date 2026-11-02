"""Build report/Final_Report_Workbook.docx -- the scaffold the team writes the final report in.

Course AI policy (syllabus, "Use of AI"): AI is allowed only as support and an AI-generated
report may not be submitted. So this workbook contains the finished EVIDENCE -- figures,
tables, numbers with their source files, the syllabus checklist and page budget, and the
questions each section must answer -- and deliberately NO report prose. The team writes
the text in the "Your text" spaces and deletes the shaded guidance boxes before submitting.

Every number is read from results/ at build time, so rerunning the analysis and then this
script keeps the workbook in sync. Formatting follows the syllabus: US Letter, 11 pt,
1.5 line spacing for the report text.
"""
from __future__ import annotations

import json
import sys
from datetime import date

import numpy as np
import polars as pl
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from common import GP, RESULTS
import viz

T = RESULTS / "tables"
FIG = GP / "figures"
OUT = GP / "report" / "Final_Report_Workbook.docx"

INK = RGBColor(0x0B, 0x0B, 0x0B)
INK2 = RGBColor(0x52, 0x51, 0x4E)
MUTED = RGBColor(0x89, 0x87, 0x81)
ACCENT = RGBColor(0x1C, 0x5C, 0xAB)
GUIDE_FILL, HEAD_FILL, WRITE_FILL = "EEF4FC", "E8E7E1", "FFFFFF"
OUTLINE_FILL = "FFF6E5"          # outline boxes; make_submission.py strips GUIDE_FILL and OUTLINE_FILL boxes
NAME = {"M1": "DINOv3 ViT-B/16", "M2": "DINOv3 ConvNeXt-B", "M3": "ImageNet ConvNeXt-B",
        "M4": "DINOv2 ViT-B/14", "M5": "ResNet-18 (scratch)"}
REG = {"ft": "fine-tuned", "frozen": "frozen", "scratch": "from scratch"}
fig_no = [0]
tab_no = [0]


# ===================================================================== docx helpers
def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def cell_margins(table, top=80, bottom=80, left=120, right=120):
    tblPr = table._tbl.tblPr
    mar = OxmlElement("w:tblCellMar")
    for k, v in (("top", top), ("bottom", bottom), ("left", left), ("right", right)):
        e = OxmlElement("w:" + k)
        e.set(qn("w:w"), str(v))
        e.set(qn("w:type"), "dxa")
        mar.append(e)
    tblPr.append(mar)


def borders(table, color="C3C2B7", size=4, inside=True):
    tblPr = table._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    edges = ["top", "left", "bottom", "right"] + (["insideH", "insideV"] if inside else [])
    for e in edges:
        x = OxmlElement("w:" + e)
        x.set(qn("w:val"), "single")
        x.set(qn("w:sz"), str(size))
        x.set(qn("w:color"), color)
        b.append(x)
    tblPr.append(b)


def keep_with_next(p):
    p.paragraph_format.keep_with_next = True
    return p


def run(p, text, bold=False, italic=False, size=None, color=None):
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    if size:
        r.font.size = Pt(size)
    if color is not None:
        r.font.color.rgb = color
    return r


def para(container, text="", style=None, size=None, color=None, bold=False, italic=False,
         spacing=None, after=4, align=None):
    p = container.add_paragraph(style=style)
    if text:
        run(p, text, bold=bold, italic=italic, size=size, color=color)
    pf = p.paragraph_format
    pf.space_after = Pt(after)
    pf.space_before = Pt(0)
    if spacing:
        pf.line_spacing = spacing
    if align:
        p.alignment = align
    return p


def rich(container, parts, style=None, size=None, after=2, spacing=1.0):
    """parts: list of (text, 'b'|'i'|'') -- for bullets with bold lead-ins."""
    p = container.add_paragraph(style=style)
    for text, fmt in parts:
        run(p, text, bold="b" in fmt, italic="i" in fmt, size=size)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = spacing
    return p


def callout(doc, title, blocks, fill=GUIDE_FILL, label="GUIDANCE - delete before submitting"):
    """A one-cell shaded box. blocks: list of ('h', text) | ('p', text) | ('b', text or parts)."""
    t = doc.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    c = t.rows[0].cells[0]
    c.width = Inches(6.5)
    shade(c, fill)
    borders(t, color={GUIDE_FILL: "B7D3F6", OUTLINE_FILL: "F0C987"}.get(fill, "C3C2B7"), size=6, inside=False)
    cell_margins(t, 100, 100, 160, 160)
    p = c.paragraphs[0]
    if label:
        run(p, label.upper() + "   ", size=7, color=MUTED, bold=True)
    run(p, title, bold=True, size=10, color=ACCENT)
    p.paragraph_format.space_after = Pt(3)
    for kind, val in blocks:
        if kind == "h":
            q = para(c, val, size=9, bold=True, color=INK, after=1)
            q.paragraph_format.space_before = Pt(4)
        elif kind == "p":
            para(c, val, size=9, color=INK2, after=2)
        elif kind == "b":
            parts = val if isinstance(val, list) else [(val, "")]
            rich(c, parts, style="List Bullet", size=9, after=1)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def outline(doc, title, budget, paras, decide=None):
    """Paragraph-by-paragraph plan for one section: what each paragraph does, its word budget and the evidence it
    carries. Fragments only, no report prose (course AI policy). paras: list of (heading, [bullets]).
    decide: judgement calls the team must make before writing."""
    blocks = [("p", f"Word budget: {budget}. One block = one paragraph. Bullets list the content and its source, "
                    "not sentences to copy. All numbers were checked against results\\ on 2026-11-01.")]
    for h, bullets in paras:
        blocks.append(("h", h))
        blocks += [("b", b) for b in bullets]
    if decide:
        blocks.append(("h", "Decide before you write (your judgement, not the outline's)"))
        blocks += [("b", q) for q in decide]
    return callout(doc, title, blocks, fill=OUTLINE_FILL, label="OUTLINE - delete before submitting")


def write_area(doc, prompt, paragraphs=3):
    """Where the team writes. 11 pt, 1.5 spacing = the syllabus format."""
    p = para(doc, "", spacing=1.5, after=0)
    run(p, "[Your text - " + prompt + "]", italic=True, size=11, color=MUTED)
    for _ in range(paragraphs - 1):
        para(doc, "", spacing=1.5, after=0)


def heading(doc, text, level):
    h = doc.add_heading(text, level=level)
    for r in h.runs:
        r.font.color.rgb = INK
    return h


def figure(doc, name, width, what, takeaway_hint):
    fig_no[0] += 1
    p = para(doc, after=2, align=WD_ALIGN_PARAGRAPH.CENTER)
    p.add_run().add_picture(str(FIG / f"{name}.png"), width=Inches(width))
    keep_with_next(p)
    c = para(doc, after=8, size=9)
    run(c, f"Figure {fig_no[0]}. ", bold=True, size=9)
    run(c, what + " ", size=9, color=INK2)
    run(c, "[Takeaway sentence - yours: " + takeaway_hint + "]", italic=True, size=9, color=MUTED)
    return fig_no[0]


def table(doc, title, header, rows, widths, note=None, size=8.5, first_bold=False, numbered=True):
    cap = keep_with_next(para(doc, after=2, size=9))
    if numbered:
        tab_no[0] += 1
        run(cap, f"Table {tab_no[0]}. ", bold=True, size=9)
    run(cap, title, size=9, color=INK2)
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    borders(t, size=4)
    cell_margins(t, 15, 15, 70, 70)
    for j, (h, w) in enumerate(zip(header, widths)):
        c = t.rows[0].cells[j]
        c.width = Inches(w)
        shade(c, HEAD_FILL)
        c.paragraphs[0].paragraph_format.space_after = Pt(0)
        run(c.paragraphs[0], h, bold=True, size=size)
    for i, r in enumerate(rows, start=1):
        for j, (v, w) in enumerate(zip(r, widths)):
            c = t.rows[i].cells[j]
            c.width = Inches(w)
            q = c.paragraphs[0]
            q.paragraph_format.space_after = Pt(0)
            v = str(v)
            if " [" in v and v.endswith("]"):          # estimate, then its CI on a muted second line
                est, ci = v.split(" [", 1)
                run(q, est, size=size, bold=(first_bold and j == 0))
                q.add_run().add_break()
                run(q, "[" + ci, size=size - 1, color=MUTED)
            else:
                run(q, v, size=size, bold=(first_bold and j == 0))
            if j > 0 and _numeric(v):
                q.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    if note:
        para(doc, note, size=8, color=MUTED, italic=True, after=8)
    else:
        para(doc, after=6)
    return tab_no[0]


def _numeric(v):
    import re
    v = str(v)
    if re.search(r"[A-Za-z]", v):
        return False
    return bool(re.match(r"^[\s$+\-−]*\d", v))


def page_break(doc):
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


# ===================================================================== numbers
def f3(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.3f}"


def sgn(x, nd=3):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+.{nd}f}"


def nfmt(x):
    if x is None or not np.isfinite(x):
        return "not reached"
    return (f"{x:,.0f}" if x < 1e5 else ">100,000") + ("†" if x > 2600 else "")


def pfmt(p):
    """Bootstrap / Wald p-value; 0.000 at 3 decimals is reported as <0.001."""
    if p is None or not np.isfinite(p):
        return ""
    return "<0.001" if p < 0.0005 else f"{p:.3f}"


def money(x):
    if x is None or not np.isfinite(x):
        return "-"
    return f"${x:,.0f}" if x >= 10 else f"${x:,.2f}"


def load():
    d = {}
    d["t1"] = pl.read_csv(T / "table1_data.csv")
    d["prev"] = pl.read_csv(T / "prevalence.csv")
    d["t3"] = pl.read_csv(T / "table3_curves.csv")
    d["pd"] = pl.read_csv(T / "paired_deltas.csv")
    d["fits"] = json.loads((RESULTS / "fits" / "fits_macro.json").read_text())
    d["ff"] = pl.read_csv(T / "fits_findings.csv")
    d["aic"] = pl.read_csv(T / "aic_curve_forms.csv")
    d["ratio"] = pl.read_csv(T / "efficiency_ratios.csv")
    d["cost"] = pl.read_csv(T / "table5_cost.csv")
    d["gh"] = pl.read_csv(T / "gpu_hours_by_n.csv")
    d["lsens"] = pl.read_csv(T / "labeling_cost_sensitivity.csv")
    d["gold"] = pl.read_csv(T / "gold58_n2600.csv")
    d["grank"] = pl.read_csv(T / "gold58_rank_agreement.csv")
    d["gdelta"] = pl.read_csv(T / "gold58_deltas_n2600.csv")
    d["het"] = pl.read_csv(T / "finding_heterogeneity.csv")
    d["sensR"] = pl.read_csv(T / "sensitivity_fallbackR.csv")
    d["sens10"] = pl.read_csv(T / "sensitivity_macro_ge10pos.csv")
    d["rot"] = pl.read_csv(T / "rotation_frozen_n2600.csv")
    d["vend"] = pl.read_csv(T / "vendor_auc_n2600.csv")
    d["mixed"] = pl.read_csv(T / "mixed_model.csv")
    d["W"] = (T / "site_kendall_w.txt").read_text().split("=")[1].split()[0]
    rp = pl.read_parquet(RESULTS / "runs.parquet")
    d["runs"] = rp
    d["n_runs"] = rp["run"].n_unique()
    d["gpu_total"] = float(rp.filter(pl.col("split") == "val").select(pl.col("gpu_h").sum()).item())
    # like-for-like comparison for the from-scratch floor (it exists on test fold 0 only)
    f0 = rp.filter((pl.col("finding") == "macro") & (pl.col("n_train") == 2600) & (pl.col("rotation") == 0)
                   & (pl.col("split").is_in(["test", "gold"])) & (pl.col("selected") | (pl.col("regime") != "frozen")))
    d["fold0"] = {(r["split"], r["regime"], r["model"]): r["auc"]
                  for r in f0.group_by(["split", "regime", "model"]).agg(pl.col("auc").mean()).iter_rows(named=True)}
    return d


def t3v(d, rg, m, n, col="test_mean", design="matched"):
    s = d["t3"].filter((pl.col("design") == design) & (pl.col("regime") == rg) & (pl.col("model") == m)
                       & (pl.col("n_train") == n))
    return s[col][0] if s.height else np.nan


def pdv(d, comp, n, field="delta", finding="macro"):
    s = d["pd"].filter((pl.col("comparison") == comp) & (pl.col("n_train") == n) & (pl.col("finding") == finding))
    return s[field][0]


def pd_str(d, comp, n):
    return f"{sgn(pdv(d, comp, n))} [{sgn(pdv(d, comp, n, 'lo'))}, {sgn(pdv(d, comp, n, 'hi'))}]"


C1V, C1C = "Fine-tuned vs frozen, DINOv3 ViT-B", "Fine-tuned vs frozen, DINOv3 ConvNeXt-B"
C2F, C2Z = "ConvNeXt-B minus ViT-B, fine-tuned", "ConvNeXt-B minus ViT-B, frozen"
C3 = "ImageNet minus DINOv3 pretraining (ConvNeXt-B, frozen)"
C4 = "DINOv3 minus DINOv2 (ViT-B, frozen)"
XC = "Fine-tuned DINOv3 ConvNeXt-B minus best frozen (ImageNet ConvNeXt-B)"
XV = "Fine-tuned DINOv3 ViT-B minus best frozen (ImageNet ConvNeXt-B)"


# ===================================================================== document
def build(d):
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Inches(1))
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    st.font.size = Pt(11)
    for lvl, sz in ((1, 15), (2, 12.5), (3, 11)):
        hs = doc.styles[f"Heading {lvl}"]
        hs.font.name = "Calibri"
        hs.font.size = Pt(sz)
        hs.font.bold = True
        hs.paragraph_format.space_before = Pt(14 if lvl == 1 else 10)
        hs.paragraph_format.space_after = Pt(4)
        hs.paragraph_format.keep_with_next = True
    # footer page numbers
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for tag, txt in (("begin", None), (None, "PAGE"), ("end", None)):
        r = fp.add_run()
        if tag:
            e = OxmlElement("w:fldChar")
            e.set(qn("w:fldCharType"), tag)
            r._r.append(e)
        else:
            e = OxmlElement("w:instrText")
            e.set(qn("xml:space"), "preserve")
            e.text = txt
            r._r.append(e)
        r.font.size = Pt(8)
        r.font.color.rgb = MUTED

    cover(doc, d)
    s1_motivation(doc, d)
    s2_data(doc, d)
    s3_methods(doc, d)
    s4_results(doc, d)
    s5_implications(doc, d)
    appendices(doc, d)
    OUT.parent.mkdir(exist_ok=True)
    doc.save(OUT)
    return OUT


# --------------------------------------------------------------------- cover
def cover(doc, d):
    para(doc, "MGMTMSA 432 Healthcare Analytics · Fall 2026 · Final group project", size=9, color=MUTED, after=18)
    para(doc, "How Much Data Does a Hospital Need?", size=24, bold=True, color=INK, after=4)
    para(doc, "Data efficiency of CNNs vs. pretrained transformers for knee-MRI diagnosis", size=14, color=INK2, after=14)
    para(doc, "Team: Tirth Patel, Giulio Elmi", size=11, color=INK2, after=2)
    para(doc, f"Workbook generated {date.today():%B %d, %Y} from T:\\Group_Project\\results ({d['n_runs']:,} training runs)",
         size=9, color=MUTED, italic=True, after=16)

    callout(doc, "How to use this workbook", [
        ("p", "This workbook holds everything the report needs except the writing: every figure, table and number, "
              "arranged in the order and page budget the syllabus asks for. The syllabus allows AI \"only as support\" and "
              "does not allow \"copying an AI-generated report\", so the prose must be your team's own. Write in the grey "
              "\"[Your text]\" spaces (already 11 pt, 1.5 spacing) and delete every shaded guidance box before submitting."),
        ("b", [("Evidence lists ", "b"), ("are facts with their source file; check any number against results\\ before you use it.", "")]),
        ("b", [("Questions to answer ", "b"), ("tell you what each section must explain. They do not tell you what to conclude.", "")]),
        ("b", [("Outline boxes ", "b"), ("(orange) plan each section paragraph by paragraph, with a word budget and the numbers "
               "each paragraph carries. They are fragments, not text to paste; the 'Decide before you write' items are yours to judge.", "")]),
        ("b", [("Word budgets ", "b"), ("were measured in Word: with the main-text figures and tables in place, about 2,230 words "
               "of text fit in 10 pages (≈500 words per page at 11 pt, 1.5 spacing), with ~4 inches spare on page 10.", "")]),
        ("b", [("Figure captions ", "b"), ("describe what is plotted. Add the one-sentence takeaway yourself.", "")]),
        ("b", [("Submitting: ", "b"), ("run code\\make_submission.py. It strips every shaded box and the workbook front matter, warns "
               "about any placeholder left, and writes report\\Final_Report.docx and .pdf with the main-text page count.", "")]),
        ("b", [("Main text vs appendix. ", "b"), ("Items marked \"Appendix\" do not count toward the 10 pages.", "")]),
        ("b", [("Re-running the analysis ", "b"), ("(code\\collect_results.py ... make_figures.py, then code\\build_workbook.py) rebuilds this file with updated numbers.", "")]),
    ], label="read first")

    heading(doc, "Syllabus checklist (Section 3, Final group project)", 2)
    table(doc, "What the syllabus requires and where this workbook covers it.",
          ["Syllabus requirement", "Where", "Done"],
          [["1. Motivate the question: the healthcare problem, why it matters, the specific question", "Section 1", "[ ]"],
           ["2. Describe the data: sources, relevant variables, important limitations", "Section 2, Table 1, Figure 1", "[ ]"],
           ["3. Describe the analysis: data preparation, methodology, model development, evaluation", "Section 3, Fig. 2, Table 2", "[ ]"],
           ["4. Present and interpret findings: main results, what they mean, implications for the healthcare problem", "Sections 4-5", "[ ]"],
           ["Maximum 10 pages, 11 pt font, 1.5 spacing, plus appendices", "Page budget below; make_submission.py counts", "[ ]"],
           ["AI disclaimer: how and for what purpose AI tools were used", "Appendix F (draft facts)", "[ ]"],
           ["Kaggle data allowed, but no reproducing or adapting existing code, notebooks or analyses", "Appendix G", "[ ]"],
           ["Grading: relevance and originality of the question, correct method, execution, insight", "Box \"What is new\" in Section 1", "[ ]"],
           ["Submit in Bruin Learn by Tue Nov 3", "-", "[ ]"]],
          [4.1, 1.8, 0.6], numbered=False)

    heading(doc, "Page budget (10 pages)", 2)
    table(doc, "Page budget for the main text, measured in Word (11 pt, 1.5 spacing, ≈500 words per page).",
          ["Report section", "Words", "Figures / tables in the main text"],
          [["1. Motivation and question", "380", "-"],
           ["2. Data", "380", "Table 1 (cohort), Figure 1 (report vs expert)"],
           ["3. Analysis", "520", "Figure 2 (pipeline), Table 2 (models)"],
           ["4. Results", "620", "Figures 3-5, Table 3 (exams to target)"],
           ["5. Implications for hospitals, limitations", "330", "Table 4 (cost)"],
           ["Total", "2,230", "Measured fit: ends ~4 in. above the foot of page 10. At 2,350 words a figure jumps and the text reaches page 11. Appendices A-I do not count."]],
          [2.6, 0.6, 3.3], numbered=False)
    page_break(doc)


# --------------------------------------------------------------------- 1
def s1_motivation(doc, d):
    heading(doc, "1. Motivation and research question", 1)
    callout(doc, "Section 1 - what the syllabus asks", [
        ("p", "\"Motivate the question. Explain the healthcare problem, why it matters, and the specific question your "
              "analysis will address.\" Budget: about 380 words of text."),
        ("h", "Questions your text must answer"),
        ("b", "What is the clinical and operational problem: knee MRI volume, reading workload, many findings per exam?"),
        ("b", "Why is \"how much labelled data\" a management question for a hospital (labelling cost, build vs buy, governance)?"),
        ("b", "What exactly do you ask? State RQ1-RQ5 (below) in your own words."),
        ("b", "What is new compared with prior work, and why should a hospital decision-maker care?"),
        ("h", "Your research questions (from your proposal / PROJECT_PLAN 2.2)"),
        ("b", [("RQ1 data scaling: ", "b"), ("how does diagnostic AUC grow with the number of training exams N (65 to 2,600)?", "")]),
        ("b", [("RQ2 regime: ", "b"), ("does fine-tuning a pretrained backbone beat using it frozen, and from which N?", "")]),
        ("b", [("RQ3 architecture vs pretraining: ", "b"), ("same DINOv3 pretraining, ViT vs CNN; same CNN, DINOv3 vs ImageNet pretraining.", "")]),
        ("b", [("RQ4 heterogeneity: ", "b"), ("do focal findings need more data than diffuse ones, after accounting for positive examples?", "")]),
        ("b", [("RQ5 management: ", "b"), ("what does reaching a target AUC cost in exams, GPU-hours and labelling, and what should a hospital do?", "")]),
        ("h", "What is new (for the originality criterion)"),
        ("b", "A data-size study on multi-sequence, multi-site 3D knee MRI with 12 findings; most prior learning-curve work is single 2D images (chest X-ray)."),
        ("b", "Frozen vs fine-tuned 2025 foundation models (DINOv3) with an architecture-controlled pair: same pretraining, ViT vs CNN."),
        ("b", "Test sets are unseen scanner/site groups, so every number is an out-of-site estimate."),
        ("b", "Results are translated into hospital cost terms (exams, GPU-hours, labelling dollars)."),
        ("h", "Background you can cite (checked 2026-11-01; full APA 7 list in Appendix H)"),
        ("b", "CNN vs ViT on medical images: Matsoukas et al. 2021; Murphy et al., Radiology: AI 2022 (DeiT-B vs DenseNet121: slightly lower AUC, similar sample efficiency)."),
        ("b", "Self-supervised pretraining and data efficiency: Azizi et al., REMEDIS, Nat. Biomed. Eng. 2023 (1-33% of the data to match supervised baselines out of distribution); DINOv3: Siméoni et al. 2025."),
        ("b", "Knee MRI AI: MRNet (Bien et al., PLOS Med 2018); CoPAS (Qiu et al., Nat. Commun. 2024: 12 abnormalities, 1,748 patients from 5 centres, labels from MRI + arthroscopy, internal AUC 0.812)."),
        ("b", "Learning curves and sample size: Cho et al. 2015; Figueroa et al. 2012; Hestness et al. 2017; Viering & Loog, IEEE TPAMI 2023."),
        ("b", "Reading time for a knee MRI: median 8 min (IQR 4.6-11.4) in a survey of 144 radiologists (Kwee et al., Tomography 2024)."),
        ("b", "The RSNA 2026 Knee Abnormality Detection challenge (Kaggle) is the data source and a natural external benchmark (Section 4.7)."),
    ])
    outline(doc, "Section 1 - paragraph plan", "about 380 words, 4 paragraphs", [
        ("¶1 The clinical problem (~100 words)", [
            "Knee MRI: high-volume musculoskeletal exam; one study = 5 series (IQR 5-6), median 162 slices (Table 1).",
            "The radiologist reads every exam for many findings (12 here); reading time median 8 min, IQR 4.6-11.4 (Kwee et al., 2024).",
            "Where AI could help: suggesting likely findings, triaging the worklist. Your proposal's ¶1 is your own wording; reuse and extend it.",
        ]),
        ("¶2 Why 'how much data' is a management question (~100 words)", [
            "A hospital building its own model pays mainly for labelled exams: expert labels ≈ $45k per 1,000 exams at central assumptions, range $9.6k-$250k (Table 4, Appendix D).",
            "Governance: images and reports normally cannot leave the hospital, so labelling and training must run locally.",
            "The three decisions the study informs: how many exams to label; frozen features vs fine-tuning; which backbone.",
        ]),
        ("¶3 Prior work and the gap (~110 words)", [
            "CNN vs transformer data efficiency was studied on 2D radiographs: Matsoukas et al. 2021; Murphy et al. 2022 (similar sample efficiency).",
            "Self-supervised pretraining improves data efficiency: Azizi et al. 2023 (REMEDIS). DINOv3 (Siméoni et al., 2025) is the newest general-purpose case.",
            "Knee MRI models: MRNet (Bien et al., 2018), CoPAS (Qiu et al., 2024, 12 abnormalities, AUC 0.812). Not checked: whether either reports AUC against training-set size; skim both before you claim they don't.",
            "Learning-curve / sample-size methods: Figueroa et al. 2012; Cho et al. 2015; Viering & Loog 2023.",
            "Gap: no learning-curve study on multi-sequence, multi-site knee MRI comparing frozen vs fine-tuned 2025 foundation models with an architecture-controlled pair.",
        ]),
        ("¶4 The question and what is new (~70 words)", [
            "One-sentence overall question (your proposal has one), then RQ1-RQ5 in one line each; a numbered list saves words.",
            "Contribution in 3-4 short points (see 'What is new' above), including out-of-site test folds and hospital cost terms.",
        ]),
    ], decide=[
        "The proposal asked 'at what data size does each finding stop improving?'. No curve plateaus by 2,600 (Section 4.4). Rephrase the RQ now "
        "(gain per doubling, exams to a target) and disclose the change in Section 3, or keep the original wording and answer 'not within 2,600'?",
        "How strongly to frame the hospital angle: a build-vs-buy decision, or only 'how many labels'?",
    ])
    write_area(doc, "about 380 words: problem, why it matters, RQ1-RQ5, what is new", 4)
    page_break(doc)


# --------------------------------------------------------------------- 2
def s2_data(doc, d):
    heading(doc, "2. Data", 1)
    pv = d["prev"]
    total_pos_audit = "64 of 84"
    callout(doc, "Section 2 - what the syllabus asks", [
        ("p", "\"Describe the data. Explain the data sources, relevant variables, and any important limitations of the data.\" "
              "Budget: about 380 words, plus Table 1 and Figure 1."),
        ("h", "Questions your text must answer"),
        ("b", "Where does the data come from, and what is one unit of analysis (a study = one patient's knee exam)?"),
        ("b", "What are the 12 outcome variables, and where do their labels come from?"),
        ("b", "Why are the report-derived labels a biased proxy, and how do you handle that (two rulers: report labels + expert gold set)?"),
        ("b", "Why does splitting by scanner group matter (unseen-site testing)?"),
        ("h", "Evidence (source: results\\tables\\table1_data.csv, prevalence.csv; FINDINGS.md; PROJECT_PLAN 4, 15)"),
        ("b", [("Source: ", "b"), ("RSNA 2026 Knee Abnormality Detection (Kaggle). Only the data is used. ", ""),
               ("4,407 studies (one per patient), 24,371 series, 819,078 slices, ~20 sites on 5 continents, 5 vendors.", "")]),
        ("b", [("Outcomes: ", "b"), ("12 binary findings: ACL, MCL, medial/lateral meniscus, medial/lateral/patellofemoral OA, effusion, "
               "synovitis, Baker's cyst, contusion, fracture.", "")]),
        ("b", [("Labels: ", "b"), ("a local LLM (qwen3:14b) read every free-text report (9+ languages) and returned positive / negative / "
               "not addressed per finding (8.4 GPU-hours; no data left the machine). Against expert labels on the 58 gold studies they "
               "score macro AUC 0.866 (synovitis lowest).", "")]),
        ("b", [("Label audit (by hand): ", "b"), (f"{total_pos_audit} randomly sampled 'positive' cells correct; weakest for MCL, lateral OA, "
               "contusion; 'not addressed' cells were almost always truly negative. Same label noise for every model compared.", "")]),
        ("b", [("Gold set: ", "b"), ("58 studies labelled from the images by two MSK radiologists with a third adjudicating; abnormality-"
               "enriched (all 58 have at least one finding). Never used for training or model selection.", "")]),
        ("b", [("Report vs expert on the same 58 knees (Figure 1): ", "b"),
               (", ".join(f"{r['finding']} {r['gold_report_pos']} vs {r['gold_pos']}" for r in pv.iter_rows(named=True)
                          if r["finding"] in ("PF OA", "Effusion", "Synovitis", "Fracture", "Contusion")) +
                " (reports vs experts). Reports mostly under-call on these knees; contusion is over-called.", "")]),
        ("b", [("Splits: ", "b"), ("studies are packed by scanner fingerprint (vendor|model|coil) into 5 folds of ~870. Fold vendor "
               "mixes differ sharply (fold 0 is 80% Philips; fold 3 is 100% 1.5 T), so each test fold behaves like a new hospital.", "")]),
        ("h", "Interpretation flag for your team"),
        ("p", "The earlier pipeline notes describe reports as over-calling mild effusion. On the 58 gold knees the reports call effusion "
              "positive less often than the experts (20 vs 35). Both can be true (reports call mild effusion, experts count moderate-"
              "large; enrichment differs), so decide how you phrase the direction of the bias."),
    ])
    t1 = d["t1"]
    keep = {"Cohort", "Scanner vendor", "Field strength", "Studies with sequence"}
    rows = [[r["section"] if i == 0 or t1["section"][i - 1] != r["section"] else "", r["item"].strip(), r["value"]]
            for i, r in enumerate(t1.iter_rows(named=True)) if r["section"] in keep]
    table(doc, "Cohort characteristics (all 4,407 studies).", ["", "Characteristic", "Value"], rows, [1.5, 3.3, 1.5],
          note="Source: results\\tables\\table1_data.csv. Sex and report-language rows are in Appendix I. "
               "FS = fat-suppressed (fluid-sensitive) sequence.", size=8)
    figure(doc, "F1b_prevalence", 5.6,
           "Share of the 58 gold-standard knees called positive for each finding by the radiology report (as read by the "
           "LLM, blue circles) and by the expert image labels (orange diamonds); grouped by the pre-registered finding groups.",
           "what does the gap say about training on report labels?")
    callout(doc, "Data limitations to state (pick the ones that matter for your argument)", [
        ("b", "Training and primary test labels are report-derived (proxy); the expert gold set is small (macro AUC resolves ~±0.03)."),
        ("b", "Gold set is abnormality-enriched, so its prevalence is not the clinic's prevalence."),
        ("b", "Rare findings: MCL and fracture have ~3% report prevalence, i.e. only 2-4 positives in the smallest training sets."),
        ("b", "359 series (~65 studies) have a guessed left/right orientation; the sensitivity check in Section 4.6 shows it does not matter."),
        ("b", "One dataset (one challenge's sites); sex is a DICOM tag with 10% missing/other; no age, symptoms or outcomes."),
    ], label="guidance")
    outline(doc, "Section 2 - paragraph plan", "about 380 words, 4 paragraphs", [
        ("¶1 Source and unit of analysis (~80 words)", [
            "RSNA 2026 Knee Abnormality Detection (Kaggle); data only (Appendix G).",
            "Unit = one study = one patient's knee exam. 4,407 studies, 24,371 series, 819,078 slices; ~20 sites, 5 continents.",
            "Heterogeneity (Table 1): Siemens 44%, Philips 30%, GE 21%; 1.5 T 58%, 3 T 36%. Axial fat-suppressed series in 100% of studies, axial non-FS in only 19%.",
        ]),
        ("¶2 Outcome variables and how they were labelled (~130 words)", [
            "12 binary findings (list them once, grouped: ligaments/menisci, OA compartments, fluid/soft tissue, bone).",
            "Only 58 studies have expert image labels (2 MSK radiologists + adjudicator), so labels for the rest come from the free-text report.",
            "Local qwen3:14b read each report (9+ languages, Appendix I) into positive / negative / not addressed; 8.4 GPU-h; no data left the machine.",
            "Quality: macro AUC 0.866 against the expert labels; hand audit 64 of 84 sampled positives correct (weakest MCL, lateral OA, contusion); 'not addressed' was almost always truly negative.",
        ]),
        ("¶3 Direction of the label bias, and the two rulers (~90 words)", [
            "Figure 1, same 58 knees, reports vs experts: PF OA 7 vs 21, effusion 20 vs 35, synovitis 13 vs 27, fracture 9 vs 18; contusion over-called (27 vs 19).",
            "Hence two evaluation rulers: report-label test fold (~870 studies, large, noisy) and expert gold-58 (small, clean; macro resolves ~±0.03).",
            "Every model sees the same labels, so comparisons are like-for-like; the noise still lowers absolute AUC, and can interact with "
            "capacity (a fine-tuned model can fit noise a frozen head cannot). The gold-58 check in 4.6 is what supports the rankings.",
        ]),
        ("¶4 Splits and limitations (~80 words)", [
            "Folds built from scanner fingerprints (vendor | model | coil), 5 × ~870; fold 0 is 80% Philips, fold 3 is 100% 1.5 T, so each test fold behaves like a new hospital.",
            "Limitations: pick 3-4 from the box above. Strongest: proxy labels; small, enriched gold set; rare findings (MCL, fracture ≈3%, 2-4 positives at N = 65); single dataset, no age/symptoms/outcomes.",
        ]),
    ], decide=[
        "Effusion wording: earlier notes say reports over-call mild effusion, but on the gold knees reports call it less often (20 vs 35). Which direction do you state, and with what qualifier?",
        "Figure 1 takeaway sentence: what does the gap imply for a model trained on report labels? Do you expect it to under-call the same findings?",
    ])
    write_area(doc, "about 380 words: source, unit, variables, labels and their bias, splits, limitations", 4)
    page_break(doc)


# --------------------------------------------------------------------- 3
def s3_methods(doc, d):
    heading(doc, "3. Analysis", 1)
    gh = d["gh"]
    callout(doc, "Section 3 - what the syllabus asks", [
        ("p", "\"Describe the analysis. Explain your analytical approach, including data preparation, methodology, model "
              "development, and evaluation.\" Budget: about 520 words, plus Figure 2 and Table 2."),
        ("h", "Questions your text must answer"),
        ("b", "How do you turn a whole MRI exam into 12 probabilities (slice encoder + study-level attention head)?"),
        ("b", "What exactly differs between models and regimes, and what is held fixed so the comparison is fair?"),
        ("b", "How are training sets of size N drawn, and how are hyperparameters and checkpoints selected without touching the test data?"),
        ("b", "How do you evaluate (two rulers), fit learning curves, and test differences statistically?"),
        ("h", "Evidence (source: PROJECT_PLAN 5, 15; code\\*.py)"),
        ("b", [("Preprocessing: ", "b"), ("DICOM decoded, resampled to 0.5 mm/pixel (320 px = 160 mm), ordered lateral-to-medial for both knees, "
               "per-series intensity window; 6 sequence slots (sagittal/coronal/axial x fat-suppressed or not). 224 px model input.", "")]),
        ("b", [("Splits: ", "b"), ("5 rotations: test = fold r, validation = fold r+1, training pool = the other 3 folds (2,609-2,610 studies). "
               "Training subsets N = 65, 130, 260, 520, 1,040, 2,080, 2,600 drawn as nested random prefixes (3 seeds).", "")]),
        ("b", [("Frozen regime: ", "b"), ("backbone run once over all 819k slices; only the study head is trained; 4 head configurations "
               "(learning rate x dropout) per cell, best chosen on validation AUC. 4 backbones x 5 rotations x 3 seeds x 7 N x 4 = 1,680 runs.", "")]),
        ("b", [("Fine-tuned regime: ", "b"), ("backbone + head trained end-to-end: layer-wise LR decay 0.75, frozen backbone for the first 10% "
               "of steps (LP-FT), AdamW, cosine schedule, bf16; budget = clip(10 N, 6,000, 26,000) study-views; 8 slices per series; "
               "10 validations, best checkpoint scored once. Learning rate 1e-4 chosen on validation only (pilot at N = 520). "
               "2 backbones x 3 rotations x 7 N = 42 runs.", "")]),
        ("b", [("From-scratch floor: ", "b"), ("ResNet-18 with random weights, rotation 0, N = 260 / 1,040 / 2,600, 3x budget.", "")]),
        ("b", [("Same for every model: ", "b"), ("study head (12 learned queries attending over all slices), loss (weighted soft-label BCE), labels, "
               "splits, selection rule. No left/right flips (they would swap medial and lateral labels).", "")]),
        ("b", [("Evaluation: ", "b"), ("primary = macro ROC-AUC over 12 findings on the unseen-site test fold with report labels (~870 studies); "
               "secondary = macro AUC on the 58 expert-labelled studies. Both scored once, after selection.", "")]),
        ("b", [("Learning curves: ", "b"), ("pre-registered inverse power law AUC(N) = a - b N^-c (a = plateau), compared by AIC with "
               "log-linear and exponential-saturation curves; summarised as AUC gained per doubling of N and exams needed to reach "
               "AUC 0.75 / 0.80 (interpolated inside the data; beyond 2,600 extrapolated, †).", "")]),
        ("b", [("Statistics: ", "b"), ("paired comparisons on identical training subsets and test folds; study-level paired bootstrap (2,000) "
               "combined with resampling of rotation units; Holm correction across the 12 findings; linear mixed model "
               "AUC ~ model x regime x log2 N with random intercepts for test fold (and seed).", "")]),
        ("b", [("Pre-registration: ", "b"), ("hypotheses H1-H5 and the three finding groups were written down on 2026-10-25 22:50, before any main "
               "result existed (PROJECT_PLAN 15).", "")]),
        ("b", [("Compute: ", "b"), (f"one laptop GPU (RTX 5070 Ti, 12 GB). All {d['n_runs']:,} runs: {d['gpu_total']:.0f} GPU-hours of training "
               "plus ~1.6 h of embedding extraction. One fine-tuned run takes "
               f"{gh.filter((pl.col('regime') == 'ft') & (pl.col('n_train') == 65))['gpu_h'].min():.1f}-"
               f"{gh.filter((pl.col('regime') == 'ft') & (pl.col('n_train') == 2600))['gpu_h'].max():.1f} h; a frozen head seconds.", "")]),
        ("h", "Deviations from the plan to disclose"),
        ("b", "Fine-tuning used one seed on rotations 0-2 (compute); all frozen-vs-fine-tuned statements use those matched cells only."),
        ("b", "Curve-fit intervals add study-level test resampling to the planned rotation bootstrap (3 rotations alone give only 10 distinct resamples)."),
        ("b", "The planned 'exams to reach the plateau' could not be estimated (no plateau within the data); gain per doubling and exams-to-target are reported instead."),
        ("b", "The chosen fine-tuning LR (1e-4) was the top of the pilot grid for both models, so fine-tuning may be slightly under-tuned (equally for both)."),
    ])
    figure(doc, "F1_pipeline", 6.4,
           "Analysis pipeline: from DICOM studies and free-text reports to 12 finding probabilities per exam, evaluated once on "
           "an unseen-site test fold (report labels) and on 58 expert-labelled studies.",
           "optional for a pipeline figure")
    table(doc, "Backbones compared. Parameter counts measured with timm (no classifier head).",
          ["ID", "Backbone", "Type", "Params", "Pretraining", "Regimes"],
          [["M1", "ViT-B/16", "Transformer", "85.6 M", "DINOv3 self-supervised, 1.7 B images", "frozen, fine-tuned"],
           ["M2", "ConvNeXt-B", "CNN", "87.6 M", "DINOv3 (distilled from ViT-7B), 1.7 B images", "frozen, fine-tuned"],
           ["M3", "ConvNeXt-B", "CNN", "87.6 M", "supervised ImageNet-22k -> 1k", "frozen"],
           ["M4", "ViT-B/14 + reg.", "Transformer", "86.6 M", "DINOv2 self-supervised, 142 M images", "frozen"],
           ["M5", "ResNet-18", "CNN", "11.2 M", "none (random initialisation)", "from scratch"]],
          [0.4, 1.0, 0.85, 0.6, 2.45, 1.2], note="M1 vs M2 isolates architecture (same pretraining); M2 vs M3 isolates pretraining (same architecture).")
    outline(doc, "Section 3 - paragraph plan", "about 520 words, 5 paragraphs", [
        ("¶1 From an exam to 12 probabilities (~100 words)", [
            "Walk the reader through Figure 2 once: DICOM → 0.5 mm/pixel, 320 px field (160 mm), lateral-to-medial ordering, per-series window → 224 px slices.",
            "Six sequence slots (3 planes × fat-suppressed or not). Slice encoder (the backbone) → study head: 12 learned queries attend over all slices of all series "
            "(attention-based pooling, Ilse et al., 2018) → 12 probabilities.",
        ]),
        ("¶2 What is compared and what is held fixed (~120 words)", [
            "Table 2: M1 vs M2 isolates architecture (same DINOv3 pretraining); M2 vs M3 isolates pretraining (same ConvNeXt-B); M4 = previous generation (DINOv2); M5 = no pretraining (floor).",
            "Frozen: backbone run once over all slices, only the head is trained. Fine-tuned: end-to-end with a frozen-backbone warm-up for the first 10% of steps "
            "(LP-FT, Kumar et al., 2022), layer-wise LR decay 0.75, AdamW, cosine schedule.",
            "Held fixed: head, loss (weighted soft-label BCE), labels, splits, selection rule. No left/right flips (they would swap medial/lateral labels).",
        ]),
        ("¶3 Data-size design and model selection (~100 words)", [
            "5 rotations: test = fold r, validation = fold r+1, training pool = other 3 folds (2,609-2,610 studies). N = 65, 130, 260, 520, 1,040, 2,080, 2,600, nested random subsets.",
            "Frozen: 4 backbones × 5 rotations × 3 seeds × 7 N, best of 4 head configurations on validation AUC (1,680 runs). Fine-tuned: 2 backbones × rotations 0-2 × 7 N, one seed "
            "(42 runs); LR 1e-4 from a validation-only pilot at N = 520. Scratch: ResNet-18, fold 0, N = 260/1,040/2,600.",
            "Test folds and gold-58 are scored once, after selection.",
        ]),
        ("¶4 Evaluation and learning curves (~100 words)", [
            "Primary metric: macro ROC-AUC over 12 findings on the unseen-site test fold (report labels). Secondary: macro AUC on gold-58.",
            "Curves: pre-registered inverse power law AUC(N) = a − b·N^(−c), compared by AIC with log-linear and exponential saturation.",
            "Summaries: AUC gained per doubling of N; exams to reach AUC 0.75 / 0.80 (interpolated inside the data, † = extrapolated).",
        ]),
        ("¶5 Statistics, pre-registration, deviations, compute (~100 words)", [
            "Paired comparisons on identical subsets and test folds; 2,000-sample study-level bootstrap combined with rotation resampling; Holm across 12 findings; "
            "mixed model AUC ~ model × regime × log2 N with random intercepts (Appendix C).",
            "H1-H5 and the finding groups pre-registered 2026-10-25 22:50, before any main result (PROJECT_PLAN 15).",
            "Deviations from the plan (box above) AND from the proposal: training sizes 175-3,480 → 65-2,600; 'exams to within 0.01 of the plateau' → gain per doubling "
            "and exams-to-target; ImageNet ConvNeXt-B was planned to be fine-tuned too but only ran frozen (compute); DINOv2 and the scratch floor added.",
            f"Compute: one laptop RTX 5070 Ti (12 GB); {d['n_runs']:,} runs, {d['gpu_total']:.0f} GPU-h training + ~1.6 h embedding.",
        ]),
    ], decide=[
        "How much hyperparameter detail goes in the main text and how much in an appendix? A table of settings would save words; Appendix C can hold it.",
        "The proposal promised a fine-tuned ImageNet ConvNeXt-B. Do you say it is missing because of compute, and what does its absence limit (RQ3 in the fine-tuned regime)?",
    ])
    write_area(doc, "about 520 words: preparation, models, regimes, sampling, selection, evaluation, curve fitting, statistics", 5)
    page_break(doc)


# --------------------------------------------------------------------- 4
def s4_results(doc, d):
    heading(doc, "4. Results", 1)
    callout(doc, "Section 4 - what the syllabus asks", [
        ("p", "\"Present and interpret your findings. Clearly communicate your main results, what they mean...\" "
              "Budget: about 620 words, plus Figures 3-5 and Table 3. Lead each subsection with the result, then the evidence."),
        ("h", "Questions your text must answer"),
        ("b", "RQ1: how fast does AUC grow with N, and is it levelling off by 2,600?"),
        ("b", "RQ2: when does fine-tuning pay off, and is there a small-N regime where frozen is as good or better?"),
        ("b", "RQ3: does the transformer-vs-CNN choice matter once pretraining is equal? Does modern self-supervised pretraining beat ImageNet?"),
        ("b", "RQ4: which findings need the most data, and is that just because they have fewer positive cases?"),
        ("b", "Do the expert gold labels tell the same story? How do your results sit next to the competition's?"),
    ])
    # ---- 4.1 curves
    heading(doc, "4.1 Learning curves (RQ1)", 2)
    figure(doc, "F2_learning_curves", 6.2,
           "Macro test AUC (12 findings, report labels, unseen-site test fold) against training-set size N, mean of rotations "
           "0-2 (seed 0) with ±1 SE bands. Solid = fine-tuned, dashed = frozen, dotted = from scratch (fold 0 only).",
           "what is the single most important pattern?")
    sizes = [65, 130, 260, 520, 1040, 2080, 2600]
    rows = []
    for rg, m in [("frozen", "M1"), ("frozen", "M2"), ("frozen", "M3"), ("frozen", "M4"), ("ft", "M1"), ("ft", "M2")]:
        rows.append([f"{NAME[m]}, {REG[rg]}"] + [f3(t3v(d, rg, m, n)) for n in sizes]
                    + [f3(t3v(d, rg, m, 2600, "gold_mean"))])
    sc = d["runs"].filter((pl.col("regime") == "scratch") & (pl.col("split").is_in(["test", "gold"])) & (pl.col("finding") == "macro"))
    scv = {(r["split"], r["n_train"]): r["auc"] for r in sc.iter_rows(named=True)}
    rows.append(["ResNet-18, scratch (fold 0)"] + [f3(scv.get(("test", n), np.nan)) if ("test", n) in scv else "-" for n in sizes]
                + [f3(scv.get(("gold", 2600), np.nan))])
    d["_t_auc_by_n"] = (sizes, rows)          # rendered in Appendix C (moved out of the main text for the page budget)
    f = d["fits"]
    f0 = d["fold0"]
    above = [NAME[m] + " frozen" for m in ("M1", "M2", "M3", "M4") if f0[("test", "frozen", m)] < f0[("test", "scratch", "M5")]]
    below = [NAME[m] + " frozen" for m in ("M1", "M2", "M3", "M4") if f0[("test", "frozen", m)] >= f0[("test", "scratch", "M5")]]
    scratch_txt = (f"ResNet-18 without pretraining reaches {f3(f0[('test', 'scratch', 'M5')])} at N = 2,600 (fold 0 only). On the same fold it beats "
                   f"{', '.join(above) or 'no pretrained model'} ({', '.join(f3(f0[('test', 'frozen', m)]) for m in ('M1', 'M2', 'M3', 'M4') if NAME[m] + ' frozen' in above)}) "
                   f"but trails {', '.join(below)} ({', '.join(f3(f0[('test', 'frozen', m)]) for m in ('M1', 'M2', 'M3', 'M4') if NAME[m] + ' frozen' in below)}) "
                   f"and both fine-tuned models ({f3(f0[('test', 'ft', 'M1')])}, {f3(f0[('test', 'ft', 'M2')])}). Gold-58: scratch "
                   f"{f3(f0[('gold', 'scratch', 'M5')])} vs frozen DINOv3 ConvNeXt-B {f3(f0[('gold', 'frozen', 'M2')])} on fold 0. Frozen ImageNet "
                   f"ConvNeXt-B already reaches {f3(t3v(d, 'frozen', 'M3', 520))} with 520 exams (3-fold mean). Do not write 'below every pretrained model'.")
    callout(doc, "Evidence for 4.1", [
        ("b", [("Growth per doubling of N: ", "b"), ("frozen +0.020 AUC, fine-tuned +0.031 (mixed model, log2 N slope 0.020 + interaction "
               "0.011, p < 0.001; results\\tables\\mixed_model.csv).", "")]),
        ("b", [("AUC gained per doubling of N (log-linear slope, 95% CI): ", "b"), ("; ".join(
            f"{NAME[m]} {REG[rg]} {f['%s|%s|%s' % (dz, rg, m)]['gain_per_doubling']:+.3f} "
            f"[{f['%s|%s|%s' % (dz, rg, m)]['gain_per_doubling_lo']:+.3f}, {f['%s|%s|%s' % (dz, rg, m)]['gain_per_doubling_hi']:+.3f}]"
            for dz, rg, m in [("frozen_full", "frozen", "M1"), ("frozen_full", "frozen", "M2"), ("frozen_full", "frozen", "M3"),
                              ("frozen_full", "frozen", "M4"), ("matched", "ft", "M1"), ("matched", "ft", "M2")]) +
            " (frozen: full design; fine-tuned: matched design; results\\tables\\table4_fits.csv).", "")]),
        ("b", [("No plateau within 65-2,600 exams: ", "b"), (f"the pre-registered power law put the plateau at its 1.0 upper bound for "
               f"{sum(v['ipl_a_at_bound'] for v in f.values())} of {len(f)} curves, i.e. the data cannot locate a plateau; 'exams to reach the "
               "plateau' is therefore not reported (Section 4.4).", "")]),
        ("b", [("From-scratch baseline (corrected 2026-11-01): ", "b"), (scratch_txt, "")]),
        ("b", [("Per-size means ", "b"), ("for every configuration are in Appendix C (Table 'Mean macro AUC by training-set size').", "")]),
    ])
    # ---- 4.2 fine-tuning vs frozen
    heading(doc, "4.2 Fine-tuning versus frozen features (RQ2, H1)", 2)
    figure(doc, "F4_paired_differences", 6.4,
           "Paired differences in macro test AUC with 95% bootstrap CIs (same training subsets and test folds). Left: fine-tuned "
           "minus frozen for each DINOv3 backbone. Right: architecture (ConvNeXt-B minus ViT-B) and pretraining contrasts. "
           "Colour = the model credited when the line is above zero; line style = regime.",
           "where does each line cross zero?")
    callout(doc, "Evidence for 4.2 (H1 predicted: frozen >= fine-tuned at the smallest N; crossover N* between 130 and 1,040). "
                 "Every paired difference at all 7 sizes, with Holm counts, is in Appendix C.", [
        ("b", [("ViT-B: ", "b"), (f"frozen is better at N = 65 ({pd_str(d, C1V, 65)}); no difference at 130-520; fine-tuning better from "
               f"N = 1,040 ({pd_str(d, C1V, 1040)}) and by {sgn(pdv(d, C1V, 2600))} at 2,600.", "")]),
        ("b", [("ConvNeXt-B: ", "b"), (f"fine-tuning is ahead at every N; the CI excludes zero from N = 260 ({pd_str(d, C1C, 260)}); "
               f"{sgn(pdv(d, C1C, 2600))} at 2,600.", "")]),
        ("b", [("Against the best cheap option ", "b"), (f"(frozen ImageNet ConvNeXt-B): fine-tuned ConvNeXt-B is not distinguishable up to "
               f"N = 520 ({pd_str(d, XC, 520)}) and better from 1,040 ({pd_str(d, XC, 1040)}); fine-tuned ViT-B is worse at N = 65 "
               f"({pd_str(d, XV, 65)}) and better only from 2,080.", "")]),
        ("b", [("Gold set agrees at N = 2,600: ", "b"), ("; ".join(
            f"{r['comparison'].split(',')[-1].strip() if r['id'] == 'C1' else r['id']} {sgn(r['gold_delta'])} [{sgn(r['lo'])}, {sgn(r['hi'])}]"
            for r in d["gdelta"].filter(pl.col("id") == "C1").iter_rows(named=True)) + " (fine-tuned - frozen on the 58 expert-labelled knees).", "")]),
    ])
    # ---- 4.3 architecture / pretraining
    heading(doc, "4.3 Architecture and pretraining (RQ3, H2, H3)", 2)
    r = d["ratio"]

    def ratio_txt(label, tau):
        s = r.filter((pl.col("comparison") == label) & (pl.col("tau") == tau))
        if not s.height or not np.isfinite(s["ratio_A_over_B"][0]):
            return "n/a"
        return f"{s['ratio_A_over_B'][0]:.2f}x [{s['ratio_lo'][0]:.2f}-{s['ratio_hi'][0]:.2f}]"
    callout(doc, "Evidence for 4.3 (H2 two-sided; H3 predicted DINOv3 > ImageNet, most at small N)", [
        ("b", [("Same DINOv3 pretraining, fine-tuned: ", "b"), (f"ConvNeXt-B ahead at N = 260-1,040 ({pd_str(d, C2F, 260)} at 260); "
               f"no difference at 2,080-2,600 ({pd_str(d, C2F, 2600)}).", "")]),
        ("b", [("Same DINOv3 pretraining, frozen: ", "b"), (f"the ViT-B is ahead, increasingly with N ({pd_str(d, C2Z, 2600)} at 2,600). "
               "So the architecture ranking flips with the regime (mixed-model model x regime interaction +0.033, p < 0.001).", "")]),
        ("b", [("Pretraining (frozen ConvNeXt-B): ", "b"), (f"ImageNet weights beat DINOv3 weights at every N ({pd_str(d, C3, 65)} at 65; "
               f"{pd_str(d, C3, 2600)} at 2,600): the opposite direction to H3. ImageNet ConvNeXt-B is the best frozen backbone in all 5 test folds.", "")]),
        ("b", [("DINOv3 vs DINOv2 ViT-B (frozen): ", "b"), (f"within ±0.011 at every N; small DINOv3 edge at 2,080-2,600 ({pd_str(d, C4, 2600)}).", "")]),
        ("b", [("Data-efficiency ratios ", "b"), ("(exams the first model needs / exams the second needs to reach AUC 0.75; >1 favours the second): "
               f"frozen->fine-tuned ViT-B {ratio_txt('C1 fine-tuning vs frozen, ViT-B', 0.75)}; frozen->fine-tuned ConvNeXt-B "
               f"{ratio_txt('C1 fine-tuning vs frozen, ConvNeXt-B', 0.75)}; fine-tuned ViT-B->ConvNeXt-B {ratio_txt('C2 ConvNeXt-B vs ViT-B, fine-tuned', 0.75)} "
               "(results\\tables\\efficiency_ratios.csv; wide where a curve barely reaches 0.75).", "")]),
    ])
    # ---- 4.4 fits
    heading(doc, "4.4 How many exams? Learning-curve fits (RQ1, H5)", 2)
    rows = []
    for key, lab in [("frozen_full|frozen|M1", "ViT-B DINOv3, frozen"), ("frozen_full|frozen|M2", "ConvNeXt-B DINOv3, frozen"),
                     ("frozen_full|frozen|M3", "ConvNeXt-B ImageNet, frozen"), ("frozen_full|frozen|M4", "ViT-B DINOv2, frozen"),
                     ("matched|frozen|M3", "ConvNeXt-B ImageNet, frozen (matched)"),
                     ("matched|ft|M1", "ViT-B DINOv3, fine-tuned"), ("matched|ft|M2", "ConvNeXt-B DINOv3, fine-tuned")]:
        x = f[key]
        rows.append([lab, f3(x["auc_at_2600"]),
                     f"{x['gain_per_doubling']:+.3f} [{x['gain_per_doubling_lo']:+.3f}, {x['gain_per_doubling_hi']:+.3f}]",
                     f"{nfmt(x['N_tau0.75'])} [{nfmt(x['N_tau0.75_lo'])}-{nfmt(x['N_tau0.75_hi'])}]",
                     f"{nfmt(x['N_tau0.80'])} [{nfmt(x['N_tau0.80_lo'])}-{nfmt(x['N_tau0.80_hi'])}]",
                     f"{f3(x['ipl_a'])} [{f3(x['ipl_a_lo'])}-{f3(x['ipl_a_hi'])}]"])
    table(doc, "How many exams? Macro test AUC at 2,600 exams, AUC gained per doubling of N, exams needed to reach AUC 0.75 and 0.80 "
               "[95% bootstrap CI], and the pre-registered power-law plateau.",
          ["Model, regime", "AUC at 2,600", "Gain per doubling", "Exams to AUC 0.75", "Exams to AUC 0.80", "Power-law plateau a"],
          rows, [1.75, 0.6, 1.1, 1.1, 1.15, 0.8], size=7.2,
          note="Exams to target: read off the observed curve (log-N interpolation) inside 65-2,600; † = beyond 2,600, extrapolated "
               "from the log-linear trend of the three largest sizes - do not over-claim. A plateau of 1.000 means the fit hit its "
               "upper bound (no plateau visible in the data). Source: results\\tables\\table4_fits.csv.")
    aic = d["aic"]
    n_best = int((aic["best"] == "power_law").sum())
    callout(doc, "Evidence for 4.4", [
        ("b", [("H5 (curve shape): ", "b"), (f"the power law has the lowest AIC for {n_best} of {aic.height} curves; otherwise "
               + ", ".join(f"{r['regime']} {r['model']} ({r['design']}): {r['best'].replace('_', ' ')}, ΔAIC {r['dAIC_power_minus_best']:.1f}"
                           for r in aic.filter(pl.col("best") != "power_law").iter_rows(named=True)) +
               " (results\\tables\\aic_curve_forms.csv).", "")]),
        ("b", [("Deviation to disclose: ", "b"), ("H5 and the planned 'exams to reach the plateau' (N_0.02) relied on a fitted plateau. "
               "Because no curve shows an identifiable plateau, the report uses the gain per doubling and exams-to-target instead "
               "(logged in PROJECT_PLAN 15).", "")]),
        ("b", [("AUC 0.80: ", "b"), ("both fine-tuned models reach it inside the data (" + ", ".join(
            f"{NAME[m]} {nfmt(f['matched|ft|' + m]['N_tau0.80'])} exams" for m in ("M1", "M2")) + "); no frozen backbone reaches it by "
            "2,600 (extrapolated need: " + ", ".join(f"{NAME[m]} {nfmt(f['frozen_full|frozen|' + m]['N_tau0.80'])}" for m in ("M1", "M3")) + ").", "")]),
    ])
    # ---- 4.5 findings
    heading(doc, "4.5 Which findings need the most data? (RQ4, H4)", 2)
    figure(doc, "F6_auc_vs_positives", 6.4,
           "Per-finding test AUC against the number of positive training examples of that finding (log scale), mean of rotations "
           "0-2 at each N, coloured by pre-registered finding group. Left: fine-tuned DINOv3 ConvNeXt-B; right: frozen ImageNet ConvNeXt-B.",
           "do the focal findings line up with the others once you plot by positives?")
    het = d["het"]

    def hc(model, grp):
        s = het.filter((pl.col("model").str.starts_with(model)) & pl.col("term").str.contains(grp))
        return f"{s['coef'][0]:+.3f} [{s['lo'][0]:+.3f}, {s['hi'][0]:+.3f}]"
    callout(doc, "Evidence for 4.5 (H4 predicted: focal findings need more data than diffuse/OA findings, also per positive example)", [
        ("b", [("Average AUC gap vs fluid/marrow/soft-tissue findings, same N and configuration: ", "b"),
               (f"focal {hc('A', 'focal')}, cartilage/OA {hc('A', 'cartilage')}.", "")]),
        ("b", [("Same gaps at equal numbers of positive training examples: ", "b"),
               (f"focal {hc('B', 'focal')}, cartilage/OA {hc('B', 'cartilage')}. About a quarter of the focal gap is explained by "
                "focal findings having fewer positives; most of it remains.", "")]),
        ("b", "Each doubling of positive examples adds about +0.018 AUC for a finding (same model; results\\tables\\finding_heterogeneity.csv; "
              "OLS with run-clustered SEs, all selected frozen and fine-tuned runs)."),
        ("b", "Per-finding curves and per-finding gain per doubling are in Appendix A (slopes in results\\tables\\fits_findings.csv). MCL, fracture "
              "and synovitis have few test positives (22-32 per fold), so their per-finding numbers are noisy (wide CIs in the Appendix A gain figure)."),
    ])
    # ---- 4.6 robustness
    heading(doc, "4.6 Do the conclusions hold? Gold labels, sites and sensitivity checks", 2)
    gr = d["grank"]
    sr = d["sensR"]
    vend = d["vend"]
    gold = d["gold"].filter(pl.col("finding") == "macro")
    callout(doc, "Evidence for 4.6", [
        ("b", [("Expert gold set: ", "b"), (f"model rankings agree with the report-label test set (Spearman {gr['spearman'][0]:.2f} over "
               f"{gr['n_cells'][0]} model x regime x N cells; {gr['spearman'][1]:.2f} over the 7 configurations at N = 2,600). Gold macro AUC at "
               "N = 2,600: " + "; ".join(f"{NAME[r['model']]} {REG[r['regime']]} {f3(r['gold_auc'])} [{f3(r['lo'])}-{f3(r['hi'])}]"
                                        for r in gold.sort("gold_auc", descending=True).iter_rows(named=True)) + ".", "")]),
        ("b", [("Test sites: ", "b"), (f"the ranking of the four frozen backbones is nearly identical across the 5 test folds (Kendall W = {float(d['W']):.2f}); "
               "AUC level shifts by fold (Appendix B).", "")]),
        ("b", [("Vendors inside a test fold (N = 2,600): ", "b"), ("macro AUC ranges " + "; ".join(
            f"fold {k[0]} {k[1]} (n={k[2]}) {g['macro'].min():.2f}-{g['macro'].max():.2f}"
            for k, g in sorted(vend.group_by(['rotation', 'vendor', 'n_studies']), key=lambda kv: (kv[0][0], kv[0][1]))) +
            " across the 6 configurations (results\\tables\\vendor_auc_n2600.csv).", "")]),
        ("b", [("Uncertain left/right orientation: ", "b"), (f"excluding the ~65 studies with a guessed orientation changes medial/lateral "
               f"finding AUCs by at most {sr['change'].abs().max():.3f}.", "")]),
        ("b", [("Rare findings: ", "b"), ("restricting the macro to findings with >= 10 training positives raises small-N AUCs "
               "(e.g. +0.03 at N = 65) but leaves every ranking and difference unchanged (results\\tables\\sensitivity_macro_ge10pos.csv).", "")]),
    ])
    # ---- 4.7 competition
    heading(doc, "4.7 External benchmark: the RSNA 2026 competition", 2)
    d["_t_leaderboard"] = [
        ["1st, Team Kangaroo", "0.960", "Frontier multimodal LLM agents (GPT-6 Astra) reading reports + images; finding probabilities, ROI and lesion boxes", "single ConvNeXt-T (~28 M params), 256 px, ROI pooling"],
        ["2nd / 3rd", "0.960 / 0.959", "-", "-"],
        ["8th (public repo)", "0.957-0.958", "two rounds of pseudo-labels", "12-16-model ensembles (CoAtNet-2, CAFormer-S36), 384 px"],
        ["Gold-medal cutoff (rank 22)", "0.954", "-", "-"],
        ["This study, best (gold 58)", f"{f3(gold.sort('gold_auc', descending=True)['gold_auc'][0])}",
         "local 14B LLM reading report text only (0.866 vs gold)", "single fine-tuned ViT-B / ConvNeXt-B (~86-88 M), 224 px"]]
    callout(doc, "Evidence and caveats for 4.7 (requested by the team: include the competition, with the labelling caveat). "
                 "The leaderboard table is now in Appendix E; in the main text this is 2-3 sentences plus a pointer.", [
        ("b", [("The winner's core image model is small: ", "b"), ("a ConvNeXt-T has about a third of the parameters of our backbones. "
               "This bears directly on 'how big a model is needed'.", "")]),
        ("b", [("Related facts from this study: ", "b"), (f"fine-tuned models reach gold macro ≈ {f3(gold.filter(pl.col('regime') == 'ft')['gold_auc'].mean())} "
               "at N = 2,600 while the labels they learn from score 0.866 on the same 58 knees; at N = 2,600 the fine-tuned ViT-B and "
               "ConvNeXt-B are tied, and an older ImageNet CNN beats both DINOv3 backbones when frozen.", "")]),
        ("b", [("Why the gap (≈0.96 vs ≈0.84) cannot be attributed to labels alone: ", "b"), ("different test sets (≈1,300 hidden studies vs 58; "
               "our primary report-label metric is a third ruler and must not be compared with the leaderboard), ~4,400 vs at most 2,600 "
               "training exams, 256-384 px crops vs 224 px, large ensembles, task-specific ROI pooling.", "")]),
        ("b", [("Why we could not label like the winner: ", "b"), ("cost of frontier multimodal LLM agents over ~4,400 studies, and our "
               "rule that no images or reports leave the machine, which rules out hosted APIs regardless of budget. A hospital faces the "
               "same governance constraint.", "")]),
    ])
    outline(doc, "Section 4 - paragraph plan (write each subsection under its own heading above, then delete the boxes)",
            "about 620 words: ~90 per subsection 4.1-4.6, ~80 for 4.7", [
        ("4.1 Learning curves (~90 words) - lead: AUC keeps rising roughly linearly in log N, with no plateau by 2,600", [
            "Figure 3. Gain per doubling: frozen +0.017 to +0.020; fine-tuned +0.027 (ConvNeXt-B) and +0.031 (ViT-B) (Table 3).",
            "Mixed model: fine-tuning adds +0.011 AUC per doubling on top of frozen's +0.020 (p < 0.001; Appendix C).",
            "Scratch floor: use the corrected wording in the 4.1 evidence box (fold 0 only; it beats frozen DINOv3 ConvNeXt-B).",
        ]),
        ("4.2 Fine-tuning vs frozen (~90 words) - lead: when fine-tuning pays off depends on the backbone", [
            "Figure 4 left. ViT-B: frozen better at N = 65 (−0.019 [−0.032, −0.005]); no difference 130-520; fine-tuning better from 1,040 (+0.027); +0.068 at 2,600.",
            "ConvNeXt-B: ahead at every N, CI clear of zero from 260 (+0.047); +0.086 at 2,600.",
            "Against the cheapest strong option (frozen ImageNet ConvNeXt-B): fine-tuned ConvNeXt-B is indistinguishable up to 520 and better from 1,040 (+0.029 [+0.015, +0.041]).",
            "Gold-58 agrees at 2,600: +0.068 (ViT-B), +0.102 (ConvNeXt-B).",
        ]),
        ("4.3 Architecture and pretraining (~90 words) - lead: the architecture ranking flips with the regime; ImageNet beats DINOv3 when frozen", [
            "Same DINOv3 pretraining. Fine-tuned: ConvNeXt-B ahead at 260-1,040 (+0.030 at 260), tied at 2,600 (+0.002 [−0.010, +0.015]). "
            "Frozen: ViT-B ahead (−0.025 at 2,600). Model × regime interaction +0.033 (p < 0.001).",
            "Same ConvNeXt-B, frozen: ImageNet beats DINOv3 at every N (+0.033 to +0.050), the opposite of H3; best frozen backbone in all 5 test folds.",
            "DINOv3 vs DINOv2 ViT-B: within ±0.011 at every N.",
        ]),
        ("4.4 How many exams? (~90 words) - lead: only fine-tuned models reach AUC 0.80 within the data", [
            "Table 3: AUC 0.75 at 313 exams (fine-tuned ConvNeXt-B), 671 (fine-tuned ViT-B), 870 (frozen ImageNet, full design).",
            "AUC 0.80 at 1,333 and 1,652 exams (fine-tuned); frozen backbones need ≥ 6,355† (extrapolated: say so).",
            "H5: the power law has the lowest AIC for only 2 of 10 curves and its plateau sits at the 1.0 bound for 4, so no plateau is identifiable. H5 is not supported, "
            "which is why gain per doubling replaces 'exams to the plateau'.",
        ]),
        ("4.5 Which findings need the most data? (~90 words) - lead: focal findings lag, and only part of the lag is prevalence", [
            "Figure 5. Gap vs fluid/soft-tissue findings: focal −0.054, cartilage/OA −0.025. At equal numbers of positives: −0.040 and −0.021.",
            "So about a quarter of the focal gap is explained by fewer positives. Each doubling of a finding's positives adds about +0.018 AUC.",
            "Per-finding results are noisy for MCL, fracture and synovitis (22-32 test positives per fold); point to Appendix A.",
        ]),
        ("4.6 Robustness (~90 words) - lead: the conclusions hold on expert labels, across sites and under the sensitivity checks", [
            "Gold-58 vs report-label rankings: Spearman 0.97 over 45 cells. Frozen ranking across 5 test folds: Kendall W = 0.94 (Appendix B).",
            "Left/right exclusion changes AUC by ≤ 0.004; dropping rare findings changes no ranking.",
        ]),
        ("4.7 External benchmark (~80 words)", [
            "Winner 0.960 (gold-medal cut-off 0.954) with a ~28 M ConvNeXt-T and frontier-LLM labels from images + reports; this study's best is 0.840 on gold-58 (Appendix E).",
            "Not comparable as a score: different test sets, ~4,400 vs ≤ 2,600 training exams, 256-384 px vs 224 px, ensembles. Say this before any interpretation.",
        ]),
    ], decide=[
        "H1 verdict: supported for ViT-B (crossover inside the pre-registered 130-1,040 range) but not for ConvNeXt-B? State each pre-registered hypothesis as supported / not supported / reversed.",
        "H3 reversed: what explanations do you offer, and how firmly? Candidates (none tested here): DINOv3 ConvNeXt-B is distilled from a ViT-7B teacher; supervised ImageNet "
        "features may transfer better to this frozen head; 224 px input. Label them as hypotheses.",
        "4.7: how far can 'label quality is the binding constraint' be pushed? Fine-tuned gold 0.84 vs the labels' own 0.866 is suggestive, not proof (gold-58 resolves ±0.03).",
        "Each figure still needs its one-sentence takeaway (grey text in the caption).",
    ])
    write_area(doc, "or write here if you prefer one block; about 620 words in total for Section 4", 2)
    page_break(doc)


# --------------------------------------------------------------------- 5
def s5_implications(doc, d):
    heading(doc, "5. Implications for hospitals and limitations", 1)
    callout(doc, "Section 5 - what the syllabus asks", [
        ("p", "\"...and their implications for the healthcare problem you studied.\" Budget: about 330 words, with the cost table (Table 4); the compute-vs-AUC figure is in Appendix D."),
        ("h", "Questions your text must answer"),
        ("b", "If a hospital wants a knee-MRI triage model, what should it do at 300, 1,000 and 3,000 labelled exams?"),
        ("b", "Where should the next dollar go: more exams, better labels, or a bigger/fancier model? Use the cost table, Section 4.5 and Section 4.7."),
        ("b", "What would change your recommendation (assumptions in the cost model, label quality, site shift)?"),
        ("b", "Which limitations most affect how far the results generalise?"),
    ])
    c = d["cost"]
    rows = []
    for rg, m in [("frozen", "M3"), ("frozen", "M1"), ("ft", "M1"), ("ft", "M2")]:
        for tgt in ("AUC 0.75", "AUC 0.80", "AUC 0.85"):
            if rg == "frozen" and tgt == "AUC 0.85":      # unreachable once 0.80 is; dropped for the page budget
                continue
            s = c.filter((pl.col("regime") == rg) & (pl.col("model") == m) & (pl.col("target") == tgt))
            if not s.height:
                continue
            x = s.row(0, named=True)
            reach = x["reachable"]
            rows.append([f"{NAME[m]}, {REG[rg]}", tgt, nfmt(x["exams"]) if reach else "> 26,000 (not estimated)",
                         f"{x['gpu_h']:.2f}" if reach and x.get("gpu_h") is not None else "-",
                         money(x.get("compute_usd")) if reach else "-",
                         f"{money(x.get('label_manual_usd'))} ({money(x.get('label_manual_usd_lo'))}-{money(x.get('label_manual_usd_hi'))})" if reach else "-",
                         money(x.get("label_llm_usd")) if reach else "-"])
    table(doc, "What reaching a target costs. Exams from the fitted curves; compute measured; labelling priced under stated assumptions.",
          ["Model, regime", "Target", "Exams", "GPU-h", "Compute $", "Expert labels $ (range)", "LLM labels $"], rows,
          [1.6, 0.85, 0.7, 0.5, 0.65, 1.55, 0.65], size=7.2,
          note="Exams come from the matched design (test folds 0-2, seed 0: the cells fine-tuning was run on), so frozen rows differ from the "
               "5-fold x 3-seed rows of Table 3 (e.g. frozen ImageNet ConvNeXt-B 555 here vs 870 there; frozen DINOv3 ViT-B 1,404 vs 2,311); "
               "'> 26,000' = more than 10x the largest training set, not estimated. Exams exclude the ~870-study validation set a hospital also needs. "
               "Assumptions (Appendix H for sources): cloud GPU $0.59/h (range $0.34-0.89; RunPod, Oct 2026); expert labelling 8 min per exam "
               "(range 5-20; Kwee et al. 2024) x 2 readers (range 1-3) x $170.17/h (range $115-250; BLS May 2023 mean for radiologists); "
               "LLM labels measured at 6.9 s per report. † = extrapolated beyond 2,600 exams. Source: results\\tables\\table5_cost.csv.")
    ls = d["lsens"]
    gh = d["gh"]
    g26 = {(r["regime"], r["model"]): r["gpu_h"] for r in gh.filter(pl.col("n_train") == 2600).iter_rows(named=True)}
    callout(doc, "Evidence for Section 5", [
        ("b", [("Size of each cost line: ", "b"), (f"expert labels cost about ${ls.filter((pl.col('minutes') == 8.0) & (pl.col('readers') == 2) & (pl.col('wage') == 170.17))['usd_per_1000_exams'][0]:,.0f} "
               f"per 1,000 exams at the central assumptions (range ${ls['usd_per_1000_exams'].min():,.0f}-${ls['usd_per_1000_exams'].max():,.0f}); "
               "training compute for the most expensive run is single-digit dollars; local LLM report-reading is about $1 per 1,000 exams "
               "but gives the noisier labels this study trained on.", "")]),
        ("b", [("Compute per model at N = 2,600: ", "b"), (f"frozen ImageNet ConvNeXt-B {g26[('frozen', 'M3')]:.2f} GPU-h "
               f"(embedding + 4-configuration head search); fine-tuned ViT-B {g26[('ft', 'M1')]:.2f} h, ConvNeXt-B {g26[('ft', 'M2')]:.2f} h. "
               "Fine-tuning compute is nearly flat up to ~1,000 exams because of the minimum training budget (6,000 study-views).", "")]),
        ("b", [("Small data: ", "b"), (f"up to N = 520 the fine-tuned DINOv3 ConvNeXt-B and the frozen ImageNet ConvNeXt-B are not "
               f"significantly different ({pd_str(d, XC, 520)} at 520; Section 4.2).", "")]),
        ("b", [("External benchmark ", "b"), ("(Section 4.7): the competition winner used a ~28 M-parameter ConvNeXt-T with frontier-LLM "
               "supervision and scored 0.960 on its expert-labelled test set.", "")]),
        ("h", "Limitations to state"),
        ("b", "Report-derived labels for training and the primary test; small gold set; findings with few positives."),
        ("b", "One fine-tuning seed and 3 test folds; fine-tuning LR at the top of the pilot grid; fixed 870-study validation set at every N (a hospital also needs validation data)."),
        ("b", "224 px input for every model (fine detail such as meniscal tears may need more); single dataset; laptop GPU (absolute times are hardware-specific)."),
        ("b", "DINOv3 ConvNeXt-B is distilled from a ViT teacher, so 'CNN vs ViT' is not fully independent."),
        ("b", "Cost model rests on stated assumptions (reading time, readers, wage, cloud price); the sensitivity table (Appendix D) shows the range."),
        ("b", "No curve reached a plateau inside 65-2,600 exams, so every 'exams needed' figure beyond 2,600 is an extrapolation."),
    ])
    outline(doc, "Section 5 - paragraph plan", "about 330 words, 3 paragraphs", [
        ("¶1 What a hospital should do at each data size (~120 words)", [
            "Up to ~500 labelled exams: frozen ImageNet ConvNeXt-B and fine-tuned DINOv3 ConvNeXt-B are not distinguishable (+0.016 [−0.002, +0.033] at 520), and the frozen model costs a fraction of the compute.",
            "But in Table 4 fine-tuned ConvNeXt-B reaches AUC 0.75 with fewer exams (313 vs 555), so its labelling bill is lower ($14.2k vs $25.2k). Resolve this tension in your recommendation.",
            "~1,000 exams: fine-tuning pays (+0.029 over frozen ImageNet at 1,040). ≥ 2,000: either fine-tuned DINOv3 backbone (tied); AUC 0.80 at 1,333-1,652 exams; "
            "0.85 would need ~4,000-4,900† (extrapolated).",
            "Exam counts exclude the ~870-study validation set; a hospital needs one too (note under Table 4).",
        ]),
        ("¶2 Where the next dollar goes (~110 words)", [
            "Cost lines differ by orders of magnitude (Table 4): compute ≤ $2 per model; local LLM labels $0.35-5.49 per target; expert labels $14k-$222k per target. Spending is about labels, not GPUs.",
            "Label quality as a lever: fine-tuned gold 0.84 vs the training labels' own 0.866 on the same knees; the competition winner used a small model with far better labels (Section 4.7).",
            "Governance: the winner's labelling used hosted frontier models, which a hospital usually cannot send images or reports to. That is why local labelling was used here.",
        ]),
        ("¶3 Limitations and what would change the recommendation (~100 words)", [
            "Pick the 4 that most limit generalisation (list in the box above): proxy labels; one fine-tuning seed / 3 folds / LR at the edge of the grid; 224 px input; "
            "single dataset; extrapolation beyond 2,600; cost assumptions (Appendix D range: $9.6k-$250k per 1,000 exams).",
            "For each, say which way it would move the conclusion, if you can.",
        ]),
    ], decide=[
        "Your bottom-line recommendation in one sentence: buy labels or buy models? And at what data size does it switch?",
        "Is AUC 0.80 a clinically meaningful target for triage, or just a convenient threshold? Say why you chose 0.75 / 0.80.",
    ])
    write_area(doc, "about 330 words: recommendation for a hospital, where to invest, limitations", 4)
    page_break(doc)


# --------------------------------------------------------------------- references (APA 7)
# Parts: plain text, "*...*" = italic. CHECK = a detail not yet confirmed at the source; make_submission.py warns while any remain.
CHECK = "  [CHECK: {}]"
REFS = [
    "Azizi, S., Culp, L., Freyberg, J., Mustafa, B., Baur, S., Kornblith, S., … Natarajan, V. (2023). Robust and data-efficient generalization of "
    "self-supervised machine learning for diagnostic imaging. *Nature Biomedical Engineering, 7*(6), 756–779. https://doi.org/10.1038/s41551-023-01049-7"
    + CHECK.format("APA 7 lists the first 19 authors, then … and the last; copy them from the DOI page"),
    "Bien, N., Rajpurkar, P., Ball, R. L., Irvin, J., Park, A., Jones, E., … Lungren, M. P. (2018). Deep-learning-assisted diagnosis for knee magnetic "
    "resonance imaging: Development and retrospective validation of MRNet. *PLOS Medicine, 15*(11), e1002699. https://doi.org/10.1371/journal.pmed.1002699"
    + CHECK.format("full author list from the DOI page"),
    "Cho, J., Lee, K., Shin, E., Choy, G., & Do, S. (2015). *How much data is needed to train a medical image deep learning system to achieve necessary "
    "high accuracy?* (arXiv:1511.06348). arXiv. https://doi.org/10.48550/arXiv.1511.06348",
    "Darcet, T., Oquab, M., Mairal, J., & Bojanowski, P. (2024). Vision transformers need registers. In *The Twelfth International Conference on "
    "Learning Representations (ICLR 2024)*.",
    "Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. In *2009 IEEE "
    "Conference on Computer Vision and Pattern Recognition* (pp. 248–255). IEEE. https://doi.org/10.1109/CVPR.2009.5206848",
    "Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., "
    "Uszkoreit, J., & Houlsby, N. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. In *The Ninth International "
    "Conference on Learning Representations (ICLR 2021)*.",
    "Figueroa, R. L., Zeng-Treitler, Q., Kandula, S., & Ngo, L. H. (2012). Predicting sample size required for classification performance. *BMC Medical "
    "Informatics and Decision Making, 12*, Article 8. https://doi.org/10.1186/1472-6947-12-8",
    "He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *2016 IEEE Conference on Computer Vision and Pattern "
    "Recognition* (pp. 770–778). IEEE. https://doi.org/10.1109/CVPR.2016.90",
    "Hestness, J., Narang, S., Ardalani, N., Diamos, G., Jun, H., Kianinejad, H., Patwary, M. M. A., Yang, Y., & Zhou, Y. (2017). *Deep learning scaling "
    "is predictable, empirically* (arXiv:1712.00409). arXiv. https://doi.org/10.48550/arXiv.1712.00409",
    "Hirose, T. (2026). *rsna_2026_8th_place_solution* [Computer software]. GitHub. https://github.com/TomokiHirose/rsna_2026_8th_place_solution",
    "Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics, 6*(2), 65–70.",
    "Ilse, M., Tomczak, J. M., & Welling, M. (2018). Attention-based deep multiple instance learning. In *Proceedings of the 35th International "
    "Conference on Machine Learning* (Vol. 80, pp. 2127–2136). PMLR.",
    "Kumar, A., Raghunathan, A., Jones, R., Ma, T., & Liang, P. (2022). Fine-tuning can distort pretrained features and underperform out-of-distribution. "
    "In *The Tenth International Conference on Learning Representations (ICLR 2022)*.",
    "Kwee, R. M., Amasha, A. A. H., & Kwee, T. C. (2024). Reading times of common musculoskeletal MRI examinations: A survey study. *Tomography, 10*(9), "
    "1527–1533. https://doi.org/10.3390/tomography10090112",
    "Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s. In *2022 IEEE/CVF Conference on Computer "
    "Vision and Pattern Recognition* (pp. 11976–11986). IEEE.",
    "Matsoukas, C., Haslum, J. F., Söderberg, M., & Smith, K. (2021). *Is it time to replace CNNs with transformers for medical images?* "
    "(arXiv:2108.09038). arXiv. https://doi.org/10.48550/arXiv.2108.09038",
    "Murphy, Z. R., Venkatesh, K., Sulam, J., & Yi, P. H. (2022). Visual transformers and convolutional neural networks for disease classification on "
    "radiographs: A comparison of performance, sample efficiency, and hidden stratification. *Radiology: Artificial Intelligence, 4*(6), e220012. "
    "https://doi.org/10.1148/ryai.220012",
    "Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., … Bojanowski, P. (2024). DINOv2: Learning robust visual features "
    "without supervision. *Transactions on Machine Learning Research*." + CHECK.format("author list (first 19 + last) from openreview.net"),
    "Qiu, Z., … (2024). Learning co-plane attention across MRI sequences for diagnosing twelve types of knee abnormalities. *Nature Communications, 15*, "
    "Article [number]. https://doi.org/10.1038/s41467-024-51888-4" + CHECK.format("authors after Qiu, Z. and the article number, from the DOI page"),
    "Radiological Society of North America. (2026). *RSNA 2026 knee abnormality detection* [Data set and leaderboard]. Kaggle. Retrieved October 31, "
    "2026, from https://www.kaggle.com/competitions/rsna-knee-abnormality-detection",
    "RunPod. (2026). *GPU cloud pricing*. Retrieved October 31, 2026, from https://www.runpod.io/pricing",
    "Siméoni, O., Vo, H. V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., Massa, F., "
    "Haziza, D., Wehrstedt, L., Wang, J., Darcet, T., Moutakanni, T., Sentana, L., Roberts, C., Vedaldi, A., … Bojanowski, P. (2025). *DINOv3* "
    "(arXiv:2508.10104). arXiv. https://doi.org/10.48550/arXiv.2508.10104",
    "Team Kangaroo. (2026). *1st place solution* [Competition write-up]. Kaggle. "
    "https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/1st-place-solution",
    "U.S. Bureau of Labor Statistics. (2024). *Occupational employment and wages, May 2023: 29-1224 Radiologists*. U.S. Department of Labor. "
    "https://www.bls.gov/oes/2023/may/oes291224.htm",
    "Viering, T., & Loog, M. (2023). The shape of learning curves: A review. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 45*(6), "
    "7799–7819. https://doi.org/10.1109/TPAMI.2022.3220744",
]


def references(doc):
    callout(doc, "Reference list - APA 7, checked 2026-11-01", [
        ("p", "Checked online on 2026-11-01: Azizi, Cho, Figueroa, Hestness, Murphy, Qiu (DOI only), Siméoni, Viering & Loog. "
              "Checked on 2026-10-31: Kwee, BLS, RunPod and the competition sources. Standard ML references written from Claude's "
              "knowledge and NOT opened (glance at each before submitting): Bien, Darcet, Deng, Dosovitskiy, He, Holm, Ilse, Kumar, Liu, "
              "Matsoukas, Oquab. Entries tagged [CHECK: ...] still need one detail from the source. Delete entries you do not cite; "
              "in the text cite as (Author, year), or (Author et al., year) for three or more authors."),
    ], label="guidance")
    import re
    for ref in REFS:
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.left_indent, pf.first_line_indent = Inches(0.5), Inches(-0.5)
        pf.space_after, pf.line_spacing = Pt(4), 1.15
        body, _, chk = ref.partition("  [CHECK:")
        for i, piece in enumerate(re.split(r"\*", body)):
            if piece:
                run(p, piece, italic=(i % 2 == 1), size=10)
        if chk:
            run(p, "  [CHECK:" + chk, size=8, color=RGBColor(0xB4, 0x3C, 0x1E), bold=True)


# --------------------------------------------------------------------- appendices
def appendices(doc, d):
    heading(doc, "Appendices (do not count toward the 10 pages)", 1)
    heading(doc, "Appendix A. Per-finding results", 2)
    figure(doc, "F3_findings", 6.4,
           "Test AUC per finding against training-set size for the two fine-tuned DINOv3 models and three frozen backbones "
           "(rotations 0-2, seed 0).", "optional")
    figure(doc, "F5_gain_per_doubling", 5.9,
           "Per-finding test AUC gained per doubling of training exams (log-linear slope over 65-2,600, rotations 0-2) with 95% "
           "bootstrap CIs, for the two fine-tuned DINOv3 models and the best frozen backbone, grouped by pre-registered finding group.",
           "which findings benefit most from more exams?")
    gold = d["gold"].filter(pl.col("finding") != "macro")
    from common import LABELS
    rows = []
    for lab in LABELS:
        row = [lab]
        for rg, m in [("ft", "M1"), ("ft", "M2"), ("frozen", "M3")]:
            s = gold.filter((pl.col("finding") == lab) & (pl.col("regime") == rg) & (pl.col("model") == m)).row(0, named=True)
            row.append(f"{f3(s['gold_auc'])} [{f3(s['lo'])}-{f3(s['hi'])}]")
        rows.append(row)
    table(doc, "Expert gold-set AUC per finding at N = 2,600 (mean of folds 0-2) with 95% bootstrap CI over the 58 studies.",
          ["Finding", "ViT-B DINOv3 fine-tuned", "ConvNeXt-B DINOv3 fine-tuned", "ConvNeXt-B ImageNet frozen"], rows,
          [1.5, 1.65, 1.65, 1.65], size=7.5, note="Source: results\\tables\\gold58_n2600.csv. Single findings resolve only to about ±0.12.")
    heading(doc, "Appendix B. Results by test site", 2)
    figure(doc, "F8_by_test_site", 6.6,
           "Macro test AUC by training-set size, separately for each test fold (each fold = a different scanner/site mix). "
           "Frozen: 3 seeds averaged; fine-tuned: folds 0-2 only.", "optional")
    rot = d["rot"]
    table(doc, "Frozen backbones at N = 2,600 by test fold (mean of 3 seeds).",
          ["Test fold", NAME["M1"], NAME["M2"], NAME["M3"], NAME["M4"]],
          [[str(r["rotation"])] + [f3(r[m]) for m in ("M1", "M2", "M3", "M4")] for r in rot.iter_rows(named=True)],
          [0.8, 1.4, 1.4, 1.4, 1.4], note=f"Kendall's W for rank agreement across folds = {float(d['W']):.3f}. Source: results\\tables\\rotation_frozen_n2600.csv.")
    heading(doc, "Appendix C. Full statistics", 2)
    sizes, rows = d["_t_auc_by_n"]
    table(doc, "Mean macro AUC on the unseen-site test fold by training-set size (matched design: rotations 0-2, seed 0); last column = expert gold set at N = 2,600.",
          ["Model, regime"] + [f"{n:,}" for n in sizes] + ["Gold 2,600"], rows,
          [1.85] + [0.55] * 7 + [0.75], size=7.5,
          note="Source: results\\tables\\table3_curves.csv (design 'matched'; SEs there). Full frozen design (5 folds x 3 seeds): "
               "same file, design 'frozen_full'.")
    pdl = d["pd"]
    rows = []
    for comp in [C1V, C1C, C2F, C2Z, C3, C4, XC, XV]:
        for n in (65, 130, 260, 520, 1040, 2080, 2600):
            s = pdl.filter((pl.col("comparison") == comp) & (pl.col("n_train") == n))
            mac = s.filter(pl.col("finding") == "macro").row(0, named=True)
            sig = s.filter((pl.col("finding") != "macro") & (pl.col("p_holm") < 0.05))
            rows.append([comp if n == 65 else "", f"{n:,}", sgn(mac["delta"]), f"[{sgn(mac['lo'])}, {sgn(mac['hi'])}]",
                         pfmt(mac["p"]), f"{(sig['delta'] > 0).sum()} / {(sig['delta'] < 0).sum()}"])
    table(doc, "Paired differences in macro AUC at every training size, and the number of findings (of 12) with a Holm-significant difference. "
               "The last two contrasts are exploratory (not pre-registered).",
          ["Contrast (B - A)", "N", "Δ AUC", "95% CI", "p", "Findings sig. up / down"], rows,
          [2.6, 0.55, 0.65, 1.15, 0.55, 1.0], size=7,
          note="Source: results\\tables\\paired_deltas.csv. Fine-tuned contrasts: 3 folds (seed 0); frozen-only contrasts: 5 folds x 3 seeds.")
    mx = d["mixed"]
    table(doc, "Linear mixed model: macro test AUC ~ model x regime x log2(N), random intercept for test fold (seed as variance component); "
               "DINOv3 ViT-B frozen at N = 520 is the reference.",
          ["Term", "Coef.", "SE", "p", "95% CI"],
          [[str(r[""]).replace("C(model, Treatment('M1'))[T.M2]", "ConvNeXt-B").replace("C(regime, Treatment('frozen'))[T.ft]", "fine-tuned")
            .replace("log2n_c", "log2 N").replace(":", " x "), f"{r['Coef.']:+.3f}", f"{r['Std.Err.']:.3f}", pfmt(r["P>|z|"]),
            f"[{r['[0.025']:+.3f}, {r['0.975]']:+.3f}]" if r["[0.025"] is not None else ""] for r in mx.iter_rows(named=True)],
          [2.9, 0.7, 0.7, 0.6, 1.3], size=7.5, note="Source: results\\tables\\mixed_model.csv.")
    heading(doc, "Appendix D. Cost details", 2)
    figure(doc, "F7_cost_vs_auc", 6.0,
           "Measured GPU-hours per trained model against macro test AUC; points are the 7 training sizes (matched design). "
           "Frozen cost includes embedding the training and validation studies and the full 4-configuration head search; "
           "fine-tuning cost is nearly flat below ~1,000 exams because every run trains for at least 6,000 study-views.",
           "what does a hospital buy with 10x the compute?")
    ls = d["lsens"]
    rows = [[f"{r['minutes']:g}", f"{r['readers']}", f"${r['wage']:,.0f}", money(r["usd_per_1000_exams"])]
            for r in ls.filter(pl.col("wage").is_in([115.0, 170.17, 250.0]) & pl.col("minutes").is_in([5.0, 8.0, 20.0])).iter_rows(named=True)]
    table(doc, "Sensitivity of expert labelling cost per 1,000 exams to reading time, number of readers and wage.",
          ["Minutes / exam", "Readers", "Wage / h", "Cost per 1,000 exams"], rows, [1.4, 1.0, 1.2, 1.8], size=7.5,
          note="Source: results\\tables\\labeling_cost_sensitivity.csv.")

    heading(doc, "Appendix E. External benchmark: RSNA 2026 competition leaderboard", 2)
    table(doc, "Final private leaderboard of the RSNA 2026 Knee Abnormality Detection challenge vs this study (different test sets: read the caveats in Section 4.7).",
          ["", "Macro AUC", "Labels used for training", "Image model"], d["_t_leaderboard"],
          [1.45, 0.8, 2.45, 1.8], size=7.5,
          note="Competition: Kaggle API leaderboard pulled 2026-10-31; winner details from its write-up, checked by the team "
               "(results\\competition_context.md lists every source).")

    heading(doc, "Appendix F. AI-use disclosure (required by the syllabus)", 2)
    callout(doc, "Draft facts for your disclaimer - rewrite in your own words and correct anything that is not accurate", [
        ("p", "The syllabus requires \"a brief disclaimer specifying how and for what purpose the tool was used.\" These are the "
              "uses recorded in RESULTS_LOG.md and PROJECT_PLAN.md:"),
        ("b", [("Claude (Anthropic), chat and Claude Code: ", "b"), ("refining the idea and proposal; writing the project plan; "
               "writing the analysis code in code\\ (splits, subsets, frozen and fine-tuning training, sweeps, result collection, "
               "curve fitting, statistics, cost model, figures); running and monitoring the experiments; compiling this workbook "
               "(evidence, figures, tables, paragraph outlines with word budgets); checking and formatting the reference list; "
               "fact-checking the team's drafts against the results files. The research question, design decisions, interpretation "
               "and all report text are the team's.", "")]),
        ("b", [("Earlier preprocessing ", "b"), ("(DICOM scan, label extraction, pixel cache, frozen embeddings) was built by a team member with AI "
               "assistance before this course project; it is reused as infrastructure (T:\\Kaggle_Knee_Competition\\src).", "")]),
        ("b", [("A local LLM (qwen3:14b) ", "b"), ("was used as a measurement instrument to read radiology reports into labels; this is part "
               "of the method (Section 2), not writing assistance.", "")]),
        ("b", "No other AI tools were used (team, 2026-11-01). If you later run a grammar checker on the final text, add it here."),
    ], label="guidance")
    write_area(doc, "your AI-use disclaimer", 3)
    heading(doc, "Appendix G. Data and code originality statement", 2)
    callout(doc, "Facts for your statement", [
        ("b", "Only the RSNA 2026 Knee Abnormality Detection data were used. No public Kaggle notebooks, code, labels or analyses were used or adapted."),
        ("b", "The preprocessing pipeline is a team member's own prior work (Appendix F)."),
        ("b", "Competition results in Section 4.7 and Appendix E are cited as external context with sources; none of their methods are part of this analysis."),
        ("b", "Reusing the team's own pipeline was confirmed with the professor/TA (team, 2026-11-01)."),
    ], label="guidance")
    write_area(doc, "your originality statement", 2)
    heading(doc, "Appendix H. References", 2)
    references(doc)
    heading(doc, "Appendix I. Remaining cohort characteristics and reproducibility", 2)
    t1 = d["t1"].filter(pl.col("section").is_in(["Sex (DICOM tag)", "Report language*"]))
    table(doc, "Sex (DICOM tag) and report language.", ["", "Item", "Value"],
          [[r["section"] if i == 0 or t1["section"][i - 1] != r["section"] else "", r["item"], r["value"]]
           for i, r in enumerate(t1.iter_rows(named=True))], [1.6, 3.0, 1.5], size=8,
          note="*Language from a crude script/keyword classification in the earlier pipeline (RESEARCH.md 1.4).")
    table(doc, "Where every result comes from (rerun top to bottom to reproduce).",
          ["Step", "Script", "Output"],
          [["Collect all runs", "code\\collect_results.py", "results\\runs.parquet"],
           ["Data description", "code\\describe_data.py", "tables\\table1_data.csv, prevalence.csv; figures\\F1b"],
           ["Bootstrap cache", "code\\bootcache.py", "results\\cache\\boot_*.npz"],
           ["Curves and fits", "code\\fit_curves.py", "tables\\table3_curves.csv, table4_fits.csv, fits_findings.csv, aic, ratios"],
           ["Statistics", "code\\stats_tests.py", "tables\\paired_deltas.csv, mixed_model.csv, gold58_*, sensitivity_*, vendor, rotation"],
           ["Cost model", "code\\cost_model.py", "tables\\table5_cost.csv, gpu_hours_by_n.csv, labeling_cost_sensitivity.csv"],
           ["Figures", "code\\make_figures.py", "figures\\F1-F8 (.png, .pdf)"],
           ["This workbook", "code\\build_workbook.py", "report\\Final_Report_Workbook.docx"],
           ["Submission copy", "code\\make_submission.py", "report\\Final_Report.docx, .pdf"]],
          [1.3, 1.8, 3.4], size=7.5)


def main() -> int:
    viz.setup()
    out = build(load())
    print("[workbook] wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
