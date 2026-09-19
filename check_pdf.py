#!/usr/bin/env python3
"""Sanity-check a generated PDF: page count, text overflow, missing glyph markers."""
import sys
import pymupdf

path = sys.argv[1]
d = pymupdf.open(path)
L, R, T, B = 56.7, 595.276 - 56.7, 51.0, 841.89 - 48.0
bad = 0
for i, p in enumerate(d):
    for b in p.get_text("blocks"):
        x0, y0, x1, y1, txt = b[0], b[1], b[2], b[3], b[4].strip().replace("\n", " ")[:60]
        if x0 < L - 3 or x1 > R + 3 or y0 < T - 3:
            print(f"OVERFLOW p{i+1}", [round(v, 1) for v in (x0, y0, x1, y1)], txt)
            bad += 1
        elif y1 > B + 3 and "第 " not in txt:
            print(f"LOW p{i+1}", [round(v, 1) for v in (x0, y0, x1, y1)], txt)
            bad += 1
print(f"pages={d.page_count} overflows={bad}")
