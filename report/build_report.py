"""M4 - Assemble report/sections/*.md into one self-contained HTML report.

    python report/build_report.py
    -> report/PathSnap_Report.html  (images embedded; open in a browser -> Print -> Save as PDF)

Missing sections or figures are reported on the console and shown in red in the HTML,
so nothing is silently left out. Nothing here generates numbers - it only assembles.
"""
from __future__ import annotations

import base64
import mimetypes
import re
import sys
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
SECTIONS = [
    ("00_title.md", "Title & overview (M4)"),
    ("02_dataset_eda.md", "2. Dataset understanding & EDA (M1)"),
    ("03_models.md", "3. Model experimentation & comparison (M2)"),
    ("04_tuning.md", "4. Bayesian hyperparameter optimisation (M2)"),
    ("05_hmm.md", "5. HMM & Viterbi route reconstruction (M3)"),
    ("06_folium.md", "6. Geographical visualisation (M4 + M1/M3 maps)"),
    ("07_deployment.md", "7. API, UI & deployment (M4)"),
    ("08_ai_disclosure.md", "8. AI-assistance disclosure (all)"),
]
CSS = """
body{font-family:Georgia,'Times New Roman',serif;max-width:900px;margin:32px auto;padding:0 20px;line-height:1.45;color:#111}
h1{border-bottom:2px solid #333;padding-bottom:4px;margin-top:36px} h2{margin-top:26px}
table{border-collapse:collapse;margin:12px 0;font-size:13px} th,td{border:1px solid #999;padding:4px 8px;text-align:left}
th{background:#eee} code{font-size:12px;background:#f4f4f4;padding:1px 3px}
pre{background:#f4f4f4;padding:10px;overflow-x:auto;font-size:12px} img{max-width:100%;border:1px solid #ccc;margin:8px 0}
.missing{color:#b00;font-weight:bold;border:2px dashed #b00;padding:8px;margin:10px 0}
section{page-break-before:always} section:first-of-type{page-break-before:avoid}
@media print{body{margin:0;max-width:none}}
"""


def embed_images(md_text: str, base: Path, problems: list[str]) -> str:
    def repl(m: re.Match) -> str:
        alt, src = m.group(1), m.group(2)
        if src.startswith(("http://", "https://", "data:")):
            return m.group(0)
        p = (base / src).resolve()
        if not p.is_file():
            problems.append(f"missing figure: {src} (from {base.name})")
            return f'<div class="missing">MISSING FIGURE: {src} — {alt}</div>'
        mime = mimetypes.guess_type(p.name)[0] or "image/png"
        data = base64.b64encode(p.read_bytes()).decode()
        return f"![{alt}](data:{mime};base64,{data})"
    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", repl, md_text)


def main() -> int:
    problems: list[str] = []
    parts = []
    for fname, label in SECTIONS:
        p = HERE / "sections" / fname
        if not p.is_file():
            problems.append(f"missing section: sections/{fname} ({label})")
            parts.append(f'<section><div class="missing">MISSING SECTION: {label}</div></section>')
            continue
        md_text = embed_images(p.read_text(encoding="utf-8"), p.parent, problems)
        html = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
        parts.append(f"<section>{html}</section>")

    out = HERE / "PathSnap_Report.html"
    out.write_text("<!doctype html><html><head><meta charset='utf-8'><title>PathSnap Report</title>"
                   f"<style>{CSS}</style></head><body>{''.join(parts)}</body></html>", encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    if problems:
        print("\nTO FIX BEFORE SUBMITTING:")
        for x in problems:
            print("  -", x)
    else:
        print("all sections and figures present -> open in a browser, Print -> Save as PDF")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
