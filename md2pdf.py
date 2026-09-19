#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal Markdown -> PDF renderer (CJK aware) used for the paper deliverables.

Supported syntax
----------------
    # / ## / ### / ####        headings
    plain text                 paragraph (blank line separates blocks)
    - item /   - sub item      bullet list (2 space indent = level 2)
    1. item                    ordered list
    > quoted text              blockquote
    ``` ... ```                code block
    | a | b |  + |---|---|     table (first row = header)
    ![alt](path.png)           figure + caption
    $$ formula $$              centred formula
    ---                        horizontal rule
    !!! text                   call-out box
    @caption text              small centred caption
Inline: **bold**, `code`, <sub>1</sub>, <super>2</super>
"""
from __future__ import annotations

import os
import re
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, HRFlowable,
                                Image, KeepTogether, ListFlowable, ListItem,
                                PageBreak, PageTemplate, Paragraph, Spacer, Table,
                                TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(HERE, "fonts")

BODY = "NotoSC"
BODY_B = "NotoSC-Bold"
MONO = "DejaVuMono"

INK = colors.HexColor("#1a1a1a")
ACCENT = colors.HexColor("#1f4e79")
ACCENT_LIGHT = colors.HexColor("#eaf1f8")
GREY = colors.HexColor("#5a5a5a")
LINE = colors.HexColor("#c9d4e0")
CODE_BG = colors.HexColor("#f4f5f7")
BOX_BG = colors.HexColor("#fdf6e3")
BOX_LINE = colors.HexColor("#e0c97f")


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont(BODY, os.path.join(FONT_DIR, "NotoSansSC-Regular.ttf")))
    pdfmetrics.registerFont(TTFont(BODY_B, os.path.join(FONT_DIR, "NotoSansSC-Bold.ttf")))
    pdfmetrics.registerFont(
        TTFont(MONO, "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"))
    pdfmetrics.registerFontFamily(BODY, normal=BODY, bold=BODY_B, italic=BODY, boldItalic=BODY_B)
    pdfmetrics.registerFontFamily(MONO, normal=MONO, bold=MONO, italic=MONO, boldItalic=MONO)


# --------------------------------------------------------------------------- styles
def build_styles() -> dict:
    s = {}
    s["title"] = ParagraphStyle("title", fontName=BODY_B, fontSize=19, leading=27,
                                alignment=TA_CENTER, textColor=ACCENT, spaceAfter=4)
    s["subtitle"] = ParagraphStyle("subtitle", fontName=BODY, fontSize=11.5, leading=18,
                                   alignment=TA_CENTER, textColor=GREY, spaceAfter=2)
    s["h1"] = ParagraphStyle("h1", fontName=BODY_B, fontSize=15.5, leading=22,
                             textColor=ACCENT, spaceBefore=15, spaceAfter=7, wordWrap="CJK")
    s["h2"] = ParagraphStyle("h2", fontName=BODY_B, fontSize=13, leading=19,
                             textColor=ACCENT, spaceBefore=12, spaceAfter=5, wordWrap="CJK")
    s["h3"] = ParagraphStyle("h3", fontName=BODY_B, fontSize=11.5, leading=17,
                             textColor=INK, spaceBefore=10, spaceAfter=4, wordWrap="CJK")
    s["h4"] = ParagraphStyle("h4", fontName=BODY_B, fontSize=10.5, leading=16,
                             textColor=INK, spaceBefore=8, spaceAfter=3, wordWrap="CJK")
    s["body"] = ParagraphStyle("body", fontName=BODY, fontSize=10.5, leading=17.5,
                               alignment=TA_JUSTIFY, textColor=INK, spaceAfter=6,
                               rightIndent=5, wordWrap="CJK", firstLineIndent=0)
    s["li"] = ParagraphStyle("li", parent=s["body"], spaceAfter=3, leading=17,
                              leftIndent=16, bulletIndent=4, bulletFontName=BODY,
                              bulletFontSize=8.5, bulletColor=ACCENT, wordWrap="CJK")
    s["li2"] = ParagraphStyle("li2", parent=s["li"], fontSize=10, leading=16,
                              leftIndent=32, bulletIndent=20)
    s["quote"] = ParagraphStyle("quote", parent=s["body"], leftIndent=12, rightIndent=6,
                                textColor=GREY, spaceBefore=2, spaceAfter=6)
    s["caption"] = ParagraphStyle("caption", fontName=BODY, fontSize=9, leading=14,
                                  alignment=TA_CENTER, textColor=GREY, spaceBefore=3,
                                  spaceAfter=8, wordWrap="CJK")
    s["eq"] = ParagraphStyle("eq", fontName=BODY, fontSize=11, leading=19,
                             alignment=TA_CENTER, textColor=INK, spaceBefore=6,
                             spaceAfter=8, wordWrap="CJK")
    s["code"] = ParagraphStyle("code", fontName=MONO, fontSize=8.4, leading=12.6,
                               textColor=INK, backColor=CODE_BG, borderPadding=6,
                               leftIndent=6, rightIndent=6, spaceBefore=4, spaceAfter=8)
    s["cell"] = ParagraphStyle("cell", fontName=BODY, fontSize=8.6, leading=12.2,
                               textColor=INK, rightIndent=2, wordWrap="CJK")
    s["cellh"] = ParagraphStyle("cellh", parent=s["cell"], fontName=BODY_B)
    s["cellc"] = ParagraphStyle("cellc", parent=s["cell"], alignment=TA_CENTER)
    s["cellhc"] = ParagraphStyle("cellhc", parent=s["cellc"], fontName=BODY_B)
    s["box"] = ParagraphStyle("box", parent=s["body"], fontSize=10, leading=16,
                              spaceAfter=0, spaceBefore=0, wordWrap="CJK")
    s["boxh"] = ParagraphStyle("boxh", parent=s["box"], fontName=BODY_B, textColor=colors.HexColor("#8a6d1f"))
    return s


# --------------------------------------------------------------------------- inline
ALLOWED = r"(?:/?(?:b|i|u|sub|super|br|font)(?:\s+[^<>]*)?/?)"
_unescape_tags = re.compile(r"&lt;(" + ALLOWED + r")&gt;")


def inline(text: str) -> str:
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    # keep the whitelisted tags
    text = _unescape_tags.sub(r"<\1>", text)
    # `code` -> mono font (fall back to the CJK face when the snippet has CJK)
    def _code(m):
        body = m.group(1)
        face = MONO if not re.search(r"[\u2e80-\u9fff\uff00-\uffef]", body) else BODY
        return '<font name="%s" size="9">%s</font>' % (face, body)
    text = re.sub(r"`([^`]+)`", _code, text)
    # **bold**
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    # *italic* -> rendered with the regular face (no italic CJK face available)
    text = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"\1", text)
    # [text](url) -> text (url)
    text = re.sub(r"\[([^\]]+)\]\((http[^)]+)\)", r'\1（\2）', text)
    # inline math $...$ -> plain text, literal \n -> line break
    text = re.sub(r"\$([^$]+)\$", r"\1", text)
    text = text.replace("\\n", "<br/>")
    return text


# --------------------------------------------------------------------------- tables
def make_table(rows: list[list[str]], width: float, styles: dict) -> Table:
    header, body = rows[0], rows[1:]
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    header, body = rows[0], rows[1:]
    if ncol >= 8:
        styles = dict(styles)
        styles["cell"] = ParagraphStyle("cell_s", parent=styles["cell"], fontSize=7.4, leading=10.4)
        styles["cellh"] = ParagraphStyle("cellh_s", parent=styles["cell"], fontName=BODY_B)
        styles["cellc"] = ParagraphStyle("cellc_s", parent=styles["cell"], alignment=TA_CENTER)
        styles["cellhc"] = ParagraphStyle("cellhc_s", parent=styles["cellc"], fontName=BODY_B)

    # column width: proportional to the longest cell, then normalised to `width`
    weights = []
    for c in range(ncol):
        lens = [max(len(x) for x in str(r[c]).split("\n")) for r in rows]
        weights.append(max(4.0, max(lens) ** 0.85))
    tot = sum(weights)
    colw = [max(34.0, width * w / tot) for w in weights]
    scale = width / sum(colw)
    colw = [w * scale for w in colw]

    numeric = [True] * ncol
    for r in body:
        for c in range(ncol):
            if r[c] and not re.fullmatch(r"[-–—\d.,%\s]*", r[c]):
                numeric[c] = False
    data = []
    for i, r in enumerate(rows):
        line = []
        for c, cell in enumerate(r):
            st = styles["cellhc"] if (i == 0 and (numeric[c] or ncol > 6)) else \
                 styles["cellh"] if i == 0 else \
                 styles["cellc"] if numeric[c] else styles["cell"]
            line.append(Paragraph(inline(cell) if cell else "&nbsp;", st))
        data.append(line)

    t = Table(data, colWidths=colw, repeatRows=1, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT_LIGHT),
        ("TEXTCOLOR", (0, 0), (-1, 0), ACCENT),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, ACCENT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafbfc")]),
    ]))
    return t


def make_image(path: str, caption: str, width: float) -> list:
    if not os.path.isabs(path):
        path = os.path.join(HERE, path)
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(path).getSize()
    maxw = min(width, 132 * mm)
    maxh = 190 * mm
    w = maxw
    h = w * ih / iw
    if h > maxh:
        h = maxh
        w = h * iw / ih
    img = Image(path, width=w, height=h)
    img.hAlign = "CENTER"
    out = [Spacer(1, 4), img]
    if caption:
        out.append(Paragraph(inline(caption), STYLES["caption"]))
    else:
        out.append(Spacer(1, 8))
    return out


def make_box(lines: list[str], styles: dict, width: float) -> Table:
    flow = []
    for i, ln in enumerate(lines):
        st = styles["boxh"] if i == 0 and ln.startswith("!!!") else styles["box"]
        txt = ln[3:].strip() if st is styles["boxh"] else ln
        flow.append(Paragraph(inline(txt), st))
        if i < len(lines) - 1:
            flow.append(Spacer(1, 3))
    t = Table([[flow]], colWidths=[width], hAlign="CENTER")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BOX_BG),
        ("BOX", (0, 0), (-1, -1), 0.6, BOX_LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


# --------------------------------------------------------------------------- parser
def md_to_flow(md: str, styles: dict, width: float) -> list:
    lines = md.replace("\r\n", "\n").split("\n")
    flow: list = []
    i = 0
    n = len(lines)

    def flush_para(buf: list[str]) -> None:
        if not buf:
            return
        text = " ".join(x.strip() for x in buf)
        flow.append(Paragraph(inline(text), styles["body"]))

    para: list[str] = []

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # ---- code fence
        if stripped.startswith("```"):
            flush_para(para); para = []
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            txt = "<br/>".join(inline(b).replace(" ", "&nbsp;") for b in buf)
            st = styles["code"]
            if re.search(r"[\u2e80-\u9fff]", txt):
                st = ParagraphStyle("codecjk", parent=st, fontName=BODY, fontSize=9, leading=13.5)
            flow.append(Paragraph(txt, st))
            continue

        # ---- blank
        if not stripped:
            flush_para(para); para = []
            i += 1
            continue

        # ---- rule
        if re.fullmatch(r"-{3,}", stripped):
            flush_para(para); para = []
            flow.append(Spacer(1, 4))
            flow.append(HRFlowable(width="100%", thickness=0.6, color=LINE))
            flow.append(Spacer(1, 6))
            i += 1
            continue

        # ---- headings
        m = re.match(r"(#{1,4})\s+(.*)$", stripped)
        if m:
            flush_para(para); para = []
            level = len(m.group(1))
            flow.append(Paragraph(inline(m.group(2)), styles[f"h{level}"]))
            i += 1
            continue

        # ---- caption directive
        if stripped.startswith("@caption"):
            flush_para(para); para = []
            flow.append(Paragraph(inline(stripped[len("@caption"):].strip()), styles["caption"]))
            i += 1
            continue

        # ---- callout
        if stripped.startswith("!!!"):
            flush_para(para); para = []
            buf = []
            while i < n and lines[i].strip().startswith("!!!"):
                buf.append(lines[i].strip())
                i += 1
            flow.append(make_box(buf, styles, width))
            flow.append(Spacer(1, 8))
            continue

        # ---- formula
        if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4:
            flush_para(para); para = []
            flow.append(Paragraph(inline(stripped[2:-2].strip()), styles["eq"]))
            i += 1
            continue
        if stripped == "$$":
            flush_para(para); para = []
            i += 1
            buf = []
            while i < n and lines[i].strip() != "$$":
                buf.append(lines[i].strip())
                i += 1
            i += 1
            flow.append(Paragraph(inline(" ".join(buf)), styles["eq"]))
            continue

        # ---- table
        if stripped.startswith("|"):
            flush_para(para); para = []
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not re.fullmatch(r"[-: ]*", "".join(cells)):
                    rows.append(cells)
                i += 1
            if rows:
                flow.append(Spacer(1, 3))
                flow.append(make_table(rows, width, styles))
                flow.append(Spacer(1, 7))
            continue

        # ---- image
        m = re.match(r"!\[(.*?)\]\((.+?)\)", stripped)
        if m:
            flush_para(para); para = []
            flow.extend(make_image(m.group(2), m.group(1), width))
            i += 1
            continue

        # ---- blockquote
        if stripped.startswith(">"):
            flush_para(para); para = []
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip())
                i += 1
            for b in buf:
                flow.append(Paragraph(inline(b), styles["quote"]))
            continue

        # ---- lists
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
        if m:
            flush_para(para); para = []
            items = []
            kind = "num" if m.group(2)[0].isdigit() else "bullet"
            while i < n:
                m2 = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", lines[i])
                if not m2:
                    if not lines[i].strip():
                        break
                    if lines[i].startswith((" ", "\t")):
                        items[-1][1].append(lines[i].strip())
                        i += 1
                        continue
                    break
                k2 = "num" if m2.group(2)[0].isdigit() else "bullet"
                if k2 != kind:
                    break
                items.append((len(m2.group(1)), [m2.group(3)]))
                i += 1
            flow.extend(build_list(items, styles, kind))
            continue

        para.append(line)
        i += 1

    flush_para(para)
    return flow


def build_list(items: list, styles: dict, kind: str = "bullet") -> list:
    """Return bullet/numbered paragraphs (manual bullets: predictable indentation)."""
    level0 = min(i for i, _ in items)
    out: list = []
    counter = [0, 0]
    for indent, txt in items:
        level = 0 if indent == level0 else 1
        text = inline(" ".join(txt))
        if kind == "num":
            counter[level] += 1
            bullet = "%d." % counter[level]
            if level == 0:
                counter[1] = 0
        else:
            bullet = "•" if level == 0 else "–"
        out.append(Paragraph(text, styles["li"] if level == 0 else styles["li2"],
                             bulletText=bullet))
    return out


# --------------------------------------------------------------------------- canvas
from reportlab.pdfgen import canvas as _pdfcanvas


class NumberedCanvas(_pdfcanvas.Canvas):
    """two-pass canvas: keeps the page states so that 'page x / y' can be stamped."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._saved_page_states = []
        self._doc_title = ""

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_footer(total)
            super().showPage()
        super().save()

    def draw_footer(self, total: int) -> None:
        self.saveState()
        self.setStrokeColor(LINE)
        self.setLineWidth(0.5)
        self.line(20 * mm, 15 * mm, A4[0] - 20 * mm, 15 * mm)
        self.setFont(BODY, 8)
        self.setFillColor(GREY)
        self.drawString(20 * mm, 11 * mm, self._doc_title)
        self.drawRightString(A4[0] - 20 * mm, 11 * mm,
                             "第 %d 页 / 共 %d 页" % (self.getPageNumber(), total))
        self.restoreState()


STYLES: dict = {}


def convert(md_path: str, pdf_path: str, title: str, subtitle: str = "",
            footer: str | None = None) -> str:
    global STYLES
    register_fonts()
    STYLES = build_styles()
    width = A4[0] - 40 * mm
    doc = BaseDocTemplate(pdf_path, pagesize=A4,
                          leftMargin=20 * mm, rightMargin=20 * mm,
                          topMargin=18 * mm, bottomMargin=20 * mm,
                          title=title, author="论文翻译/解析")
    frame = Frame(doc.leftMargin, doc.bottomMargin, width,
                  A4[1] - doc.topMargin - doc.bottomMargin, id="main")
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame])])

    md = open(md_path, encoding="utf-8").read()
    story = [Paragraph(inline(title), STYLES["title"])]
    if subtitle:
        story.append(Paragraph(inline(subtitle), STYLES["subtitle"]))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1, color=ACCENT))
    story.append(Spacer(1, 8))
    story.extend(md_to_flow(md, STYLES, width))

    def make_canvas(*a, **kw):
        c = NumberedCanvas(*a, **kw)
        c._doc_title = footer if footer is not None else title
        return c

    doc.build(story, canvasmaker=make_canvas)
    return pdf_path


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2], sys.argv[3],
            sys.argv[4] if len(sys.argv) > 4 else "",
            sys.argv[5] if len(sys.argv) > 5 else None)
    print("written", sys.argv[2])
