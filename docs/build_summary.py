#!/usr/bin/env python3
"""Render docs/concept-summary.md to a print-ready HTML and then to PDF.

The markdown file is the single source. This script converts the small subset of markdown the
summary actually uses -- headings, bold, italic, inline code, links, one table, bullet lists,
one blockquote, horizontal rules -- rather than pulling in a dependency, so the build works on
a clean checkout with nothing installed but Python and a Chromium-family browser.

    python docs/build_summary.py            # writes docs/concept-summary.{html,pdf}
    python docs/build_summary.py --no-pdf   # HTML only

Layout is two-column A4 so ~855 words and a comparison table land on ONE page at a readable
size. Cramming to one page by shrinking the type would fail the brief's actual test, which is
whether a data scientist can read it once and explain the claim.

The fit is TIGHT -- at the current 8.4pt one extra line of prose spills to a second page, and
.foot must NOT be column-span:all (a spanner at the end of a column-fill:auto multicol opens a
new page in Chromium even with room left). Re-run this after any edit to the markdown and check
the reported page count.
"""

import argparse
import html as htmllib
import io
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "concept-summary.md")
OUT_HTML = os.path.join(HERE, "concept-summary.html")
OUT_PDF = os.path.join(HERE, "concept-summary.pdf")

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]

CSS = """
@page { size: A4; margin: 10mm 10mm 9mm; }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  margin: 0; background: #fff; color: #16161a;
  font: 8.4pt/1.36 "Charter", "Iowan Old Style", Georgia, "Times New Roman", serif;
  font-variant-numeric: tabular-nums;
  hyphens: auto;
}
.sheet { column-count: 2; column-gap: 7mm; column-fill: auto; }
.full { column-span: all; }

h1 {
  margin: 0 0 2mm; font-size: 17pt; line-height: 1.1; letter-spacing: -.01em;
  font-weight: 600;
}
.deck {
  margin: 0 0 2mm; font-size: 10.2pt; line-height: 1.3; color: #b1450f; font-weight: 600;
}
.meta {
  margin: 0 0 3mm; padding-bottom: 2.4mm; border-bottom: .5pt solid #c9c7c1;
  font-family: ui-sans-serif, "Segoe UI", system-ui, sans-serif;
  font-size: 7.2pt; letter-spacing: .04em; text-transform: uppercase; color: #6b6862;
}
h3 {
  margin: 3.4mm 0 1.4mm; font-size: 9.4pt; font-weight: 700; letter-spacing: .005em;
  color: #0a0a0c; break-after: avoid;
}
h3:first-of-type { margin-top: 0; }
p { margin: 0 0 1.9mm; text-align: justify; orphans: 2; widows: 2; }
b, strong { font-weight: 700; color: #000; }
a { color: #1a4f8a; text-decoration: none; }
code {
  font-family: ui-monospace, Consolas, "SF Mono", monospace; font-size: .88em;
  background: #f1f0ec; padding: 0 .18em; border-radius: 2px;
}
blockquote {
  margin: 2mm 0 2.4mm; padding: 1.6mm 0 1.6mm 3mm;
  border-left: 1.4pt solid #b1450f; font-size: 9.4pt; color: #22222a;
  break-inside: avoid;
}
blockquote p { margin: 0; text-align: left; }
ul { margin: 0 0 2mm; padding-left: 3.6mm; }
li { margin-bottom: 1.1mm; text-align: justify; }
hr { display: none; }

table {
  width: 100%; border-collapse: collapse; margin: 1.5mm 0 2.6mm;
  font-family: ui-sans-serif, "Segoe UI", system-ui, sans-serif;
  font-size: 7.1pt; line-height: 1.28; break-inside: avoid;
}
th, td {
  border-top: .5pt solid #d5d3cd; padding: 1.15mm 1.6mm; text-align: left;
  vertical-align: top;
}
thead th {
  border-top: .8pt solid #16161a; border-bottom: .5pt solid #16161a;
  font-weight: 700; color: #000;
}
tbody tr:last-child td { border-bottom: .8pt solid #16161a; }
tbody td:first-child { color: #55534d; }
td b { color: #b1450f; }

.foot {
  /* NOT column-span:all. A spanning element at the END of a column-fill:auto multicol makes
     Chromium open a second page for it, even with room left in column two -- measured: spanning
     foot 2 pages, non-spanning foot 1 page, same content. The table above still spans, which is
     what it needs. */
  margin-top: 2.6mm; padding-top: 2mm; break-inside: avoid;
  border-top: .5pt solid #c9c7c1;
  font-family: ui-sans-serif, "Segoe UI", system-ui, sans-serif;
  font-size: 6.7pt; line-height: 1.4; color: #6b6862;
}
.foot b { color: #3a3833; }
"""

FOOT = (
    '<p class="foot"><b>Provenance.</b> Every figure attributed to us is reproducible from '
    "<b>github.com/snivellus38/DataForge</b> by a named script; every BDH-CQ figure carries a "
    "section or table locator into arXiv:2608.09888 and is never recomputed here. "
    "<code>vendor/bdh.py</code> is Pathway&rsquo;s, MIT and unmodified. Our models are a 131K BDH "
    "trained for this submission and an 8M BDH trained by one author beforehand and reused with "
    "disclosure (it adds a learned positional embedding over RoPE and has no decay term). Neither "
    "is an official BDH model; neither is BDH-CQ. AI-assisted, disclosed in the README.</p>"
)


def inline(t: str) -> str:
    """Escape, then re-introduce the inline markup the summary uses."""
    t = htmllib.escape(t, quote=False)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*([^*]+)\*(?![\w*])", r"<em>\1</em>", t)
    return t


def convert(md: str) -> str:
    lines = md.split("\n")
    out, i = [], 0
    while i < len(lines):
        ln = lines[i]

        if not ln.strip() or ln.strip() == "---":
            i += 1
            continue

        if ln.startswith("# "):
            out.append(f"<h1>{inline(ln[2:].strip())}</h1>")
            i += 1
            continue

        if ln.startswith("### "):
            out.append(f"<h3>{inline(ln[4:].strip())}</h3>")
            i += 1
            continue

        # the deck line: a bold-only paragraph immediately after the title
        if ln.startswith("**") and ln.rstrip().endswith("**") and len(out) == 1:
            out.append(f'<p class="deck">{inline(ln.strip()[2:-2])}</p>')
            i += 1
            continue

        # the italic dateline
        if ln.startswith("*") and not ln.startswith("**") and len(out) == 2:
            buf = [ln]
            while i + 1 < len(lines) and lines[i + 1].strip():
                i += 1
                buf.append(lines[i])
            txt = " ".join(x.strip() for x in buf).strip("*")
            out.append(f'<p class="meta">{inline(txt)}</p>')
            i += 1
            continue

        if ln.startswith("> "):
            buf = []
            while i < len(lines) and lines[i].startswith(">"):
                buf.append(lines[i].lstrip("> ").rstrip())
                i += 1
            # join with <br>, not a space: the two equations are separate lines and ran
            # together into one unreadable string when this collapsed them.
            out.append("<blockquote><p>" + "<br>".join(inline(x) for x in buf) + "</p></blockquote>")
            continue

        if ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            head, body = rows[0], [r for r in rows[2:]]
            th = "".join(f"<th>{inline(c)}</th>" for c in head)
            tb = "".join(
                "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body
            )
            out.append(f'<table class="full"><thead><tr>{th}</tr></thead><tbody>{tb}</tbody></table>')
            continue

        if ln.startswith("- "):
            items = []
            while i < len(lines) and (lines[i].startswith("- ") or lines[i].startswith("  ")):
                if lines[i].startswith("- "):
                    items.append(lines[i][2:].strip())
                elif items:
                    items[-1] += " " + lines[i].strip()
                i += 1
            out.append("<ul>" + "".join(f"<li>{inline(x)}</li>" for x in items) + "</ul>")
            continue

        buf = [ln]
        while i + 1 < len(lines) and lines[i + 1].strip() and not re.match(
            r"^(#|\||>|- |\*\*)", lines[i + 1]
        ):
            i += 1
            buf.append(lines[i])
        out.append(f"<p>{inline(' '.join(x.strip() for x in buf))}</p>")
        i += 1

    return "\n".join(out)


def _wait_for_pdf(path: str, timeout: float = 30.0) -> bool:
    """Wait for a detached headless browser to finish writing `path`.

    Returns True once the file exists and its size has been stable across two checks; False if
    it never appears. Size-stability matters: the file is created empty and filled afterwards,
    so existence alone can hand the caller a zero-byte PDF.
    """
    deadline = time.time() + timeout
    last = -1
    while time.time() < deadline:
        if os.path.exists(path):
            size = os.path.getsize(path)
            if size > 0 and size == last:
                return True
            last = size
        time.sleep(0.25)
    return os.path.exists(path) and os.path.getsize(path) > 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()

    md = io.open(SRC, encoding="utf-8").read()
    body = convert(md)

    # the title block spans both columns; the rest flows in two
    head_end = body.index("<h3>")
    doc = (
        "<!doctype html>\n<html lang=\"en\">\n<meta charset=\"utf-8\">\n"
        "<title>Demonstrations Are Weights &mdash; concept summary</title>\n"
        f"<style>{CSS}</style>\n<body>\n"
        f'<div class="full">{body[:head_end]}</div>\n'
        f'<div class="sheet">{body[head_end:]}\n{FOOT}</div>\n'
    )
    # A markdown emphasis marker that failed to match renders as a literal asterisk in the
    # PDF. That happened once with the equation line -- a subscript is a word character, so
    # inline()'s lookbehind rejected the surrounding markers -- and it is invisible in the
    # source and in the word count. Only looking at the rendered page caught it. Guard it.
    # Code spans legitimately contain asterisks (G* is the name of the synapse matrix), so drop
    # them first; anything left in running prose is a marker that failed to convert.
    prose = re.sub(r"<code>.*?</code>", "", body, flags=re.S)
    stray = re.findall(r"\*", re.sub(r"<[^>]+>", "", prose))
    if stray:
        print(f"{len(stray)} stray emphasis marker(s) survived conversion", file=sys.stderr)
        return 1

    io.open(OUT_HTML, "w", encoding="utf-8", newline="\n").write(doc)
    print(f"wrote {os.path.relpath(OUT_HTML)}")

    if args.no_pdf:
        return 0

    browser = next((p for p in EDGE_CANDIDATES if os.path.exists(p)), None) or shutil.which(
        "chromium"
    )
    if not browser:
        print("no Chromium-family browser found; HTML written, PDF skipped", file=sys.stderr)
        return 1

    url = "file:///" + OUT_HTML.replace("\\", "/")
    if os.path.exists(OUT_PDF):
        os.remove(OUT_PDF)
    # Two traps here, both of which look like "the browser failed" and neither of which is:
    #  1. Edge's NEW headless mode silently writes nothing for --print-to-pdf on this build.
    #     The old mode is the one that works, so try it first.
    #  2. Old headless DETACHES -- the process we spawned exits before its child has written the
    #     file, and both modes return 0 regardless. So never trust the exit code; wait for the
    #     file to appear AND stop growing.
    for mode in ("--headless=old", "--headless"):
        subprocess.run(
            [browser, mode, "--disable-gpu", "--no-pdf-header-footer",
             f"--print-to-pdf={OUT_PDF}", url],
            capture_output=True,
        )
        if _wait_for_pdf(OUT_PDF):
            break
    else:
        print("browser produced no PDF; HTML written", file=sys.stderr)
        return 1
    print(f"wrote {os.path.relpath(OUT_PDF)}  ({os.path.getsize(OUT_PDF) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
