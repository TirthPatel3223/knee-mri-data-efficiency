"""Turn the team's edited workbook into the file that is submitted.

    python code\\make_submission.py [path\\to\\edited_workbook.docx] [--strip-placeholders]

Reads report\\Final_Report_Workbook.docx (or the given file) and NEVER modifies it. Writes report\\Final_Report.docx
and report\\Final_Report.pdf after:
  * removing every shaded guidance / outline box (one-cell tables filled GUIDE_FILL or OUTLINE_FILL);
  * removing the workbook front matter (generation note, syllabus checklist, page-budget table) but keeping the
    title block and team names;
  * removing the page breaks between Sections 1-5 (the word budgets assume continuous text) and keeping one
    before the appendices;
  * collapsing runs of empty paragraphs.
It then reports anything still unfinished ("[Your text", "[Takeaway", "[CHECK:") and, using Word, the main-text
page count (title through the end of Section 5) against the 10-page limit.
--strip-placeholders also deletes leftover "[Takeaway sentence - yours: ...]" runs and empty "[Your text ...]" lines.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn

GP = Path(__file__).resolve().parents[1]
SRC = GP / "report" / "Final_Report_Workbook.docx"
OUT = GP / "report" / "Final_Report.docx"
BOX_FILLS = {"EEF4FC", "FFF6E5"}          # build_workbook.GUIDE_FILL, OUTLINE_FILL
PLACEHOLDERS = ("[Your text", "[Takeaway sentence", "[CHECK:")


def text_of(el):
    return "".join(t.text or "" for t in el.iter(qn("w:t")))


def is_box(tbl):
    rows = tbl.findall(qn("w:tr"))
    if len(rows) != 1 or len(rows[0].findall(qn("w:tc"))) != 1:
        return False
    shd = rows[0].find(qn("w:tc")).find(f"{qn('w:tcPr')}/{qn('w:shd')}")
    return shd is not None and (shd.get(qn("w:fill")) or "").upper() in BOX_FILLS


def is_page_break(p):
    return not text_of(p).strip() and any(br.get(qn("w:type")) == "page" for br in p.iter(qn("w:br")))


def style_of(p):
    ps = p.find(f"{qn('w:pPr')}/{qn('w:pStyle')}")
    return ps.get(qn("w:val")) if ps is not None else ""


def clean(src: Path, strip: bool) -> Document:
    doc = Document(str(src))
    body = doc.element.body
    kids = [k for k in body.iterchildren() if k.tag in (qn("w:p"), qn("w:tbl"))]

    # 1. front matter: drop everything from the "Workbook generated" line up to the first Heading 1
    drop, in_front = [], False
    for k in kids:
        if k.tag == qn("w:p") and style_of(k).startswith("Heading1"):
            break
        if k.tag == qn("w:p") and text_of(k).startswith("Workbook generated"):
            in_front = True
        if in_front:
            drop.append(k)
    # 2. shaded boxes, and the spacer paragraph each box is followed by
    for i, k in enumerate(kids):
        if k.tag == qn("w:tbl") and is_box(k):
            drop.append(k)
            if i + 1 < len(kids) and kids[i + 1].tag == qn("w:p") and not text_of(kids[i + 1]).strip():
                drop.append(kids[i + 1])
    for k in set(drop):
        if k.getparent() is not None:
            body.remove(k)

    # 3. page breaks: none inside the main text, exactly one before the appendices
    kids = [k for k in body.iterchildren() if k.tag in (qn("w:p"), qn("w:tbl"))]
    for k in kids:
        if k.tag == qn("w:p") and is_page_break(k):
            body.remove(k)
    for k in body.iterchildren(qn("w:p")):
        if style_of(k).startswith("Heading1") and text_of(k).startswith("Appendices"):
            ppr = k.find(qn("w:pPr"))
            if ppr is None:
                ppr = k.makeelement(qn("w:pPr"), {})
                k.insert(0, ppr)
            if ppr.find(qn("w:pageBreakBefore")) is None:
                ppr.append(ppr.makeelement(qn("w:pageBreakBefore"), {}))
            break

    # 4. optional placeholder stripping
    if strip:
        for p in list(body.iter(qn("w:p"))):
            for r in list(p.iter(qn("w:r"))):
                if text_of(r).startswith("[Takeaway sentence"):
                    r.getparent().remove(r)
            t = text_of(p).strip()
            if t.startswith("[Your text") and t.endswith("]"):
                p.getparent().remove(p)

    # 5. collapse consecutive empty body paragraphs (keep one)
    prev_empty = False
    for k in list(body.iterchildren()):
        if k.tag != qn("w:p"):
            prev_empty = False
            continue
        empty = not text_of(k).strip() and k.find(f".//{qn('w:drawing')}") is None
        if empty and prev_empty:
            body.remove(k)
        prev_empty = empty
    return doc


def report_placeholders(doc):
    left = []
    for p in doc.element.body.iter(qn("w:p")):
        t = text_of(p)
        for ph in PLACEHOLDERS:
            if ph in t:
                left.append((ph, t.strip()[:90]))
    return left


def word_pages(docx: Path, pdf: Path):
    """Export the PDF with Word and return (main-text pages, total pages, words of body text in Sections 1-5)."""
    import win32com.client as w32
    word = w32.DispatchEx("Word.Application")
    word.Visible = False
    try:
        d = word.Documents.Open(str(docx), ReadOnly=True)
        d.Repaginate()
        app_page, words, in_main = None, 0, False
        for p in d.Paragraphs:
            st = str(p.Style.NameLocal) if p.Style else ""
            txt = p.Range.Text.strip()
            if st.startswith("Heading 1"):
                if txt.startswith("Appendices"):
                    app_page = p.Range.Information(3)
                    break
                in_main = True
                continue
            # body prose only: Normal style, not inside a table, not a caption/source note (those start with Figure/Table/Source)
            if in_main and st == "Normal" and not p.Range.Information(12) and not re.match(r"(Figure|Table) \d+\.|Source:|Exams come from", txt):
                words += len(txt.split())
        total = d.ComputeStatistics(2)
        d.ExportAsFixedFormat(str(pdf), 17)        # 17 = wdExportFormatPDF
        d.Close(False)
    finally:
        word.Quit()
    main_pages = (app_page - 1) if app_page else total
    return main_pages, total, words


def main(argv) -> int:
    strip = "--strip-placeholders" in argv
    args = [a for a in argv if not a.startswith("--")]
    src = Path(args[0]) if args else SRC
    if src.resolve() == OUT.resolve():
        print("refusing to read and write the same file")
        return 2
    doc = clean(src, strip)
    doc.save(str(OUT))
    print(f"[submission] wrote {OUT}")
    left = report_placeholders(doc)
    if left:
        print(f"[submission] {len(left)} unfinished item(s):")
        for ph, t in left:
            print(f"    {ph:<20} {t}")
    else:
        print("[submission] no placeholders left")
    try:
        main_pages, total, words = word_pages(OUT, OUT.with_suffix(".pdf"))
        verdict = "OK" if main_pages <= 10 else "OVER THE 10-PAGE LIMIT"
        print(f"[submission] main text (title to end of Section 5): {main_pages} pages - {verdict}; "
              f"{total} pages with appendices; ~{words:,} words of body text in Sections 1-5")
        print(f"[submission] wrote {OUT.with_suffix('.pdf')}")
    except Exception as e:                       # Word missing or busy: the .docx is still written
        print(f"[submission] page count skipped (Word automation failed: {e})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
