#!/usr/bin/env python3
"""Build docs/concept-summary.pdf from docs/concept-summary.tex.

    python docs/build_summary.py              # generate numbers, build, check
    python docs/build_summary.py --no-check   # build only, for iterating on layout

Three stages.

1. GENERATE. Every measured number in the summary is read out of research/runs/*.json and the
   committed training telemetry, and written to docs/generated-numbers.tex as \\newcommand
   definitions the .tex inputs. Nothing measured is typed into the prose. This exists because the
   previous summary quoted MCC +0.944 from one export while the run that carries the null says
   +0.940, and nothing caught it. Numbers with no committed JSON (the equivalence residuals, the
   1/pi capacity curve) stay literal in the .tex with a comment naming the script that produces
   them.

2. BUILD. latexmk -> docs/build, then copy to docs/concept-summary.pdf.

3. CHECK, and exit non-zero on failure. The page is a scored deliverable with a hard one-page
   limit, a word budget, and a list of claims it must carry and misconceptions it must not. All of
   that is asserted here rather than left to a reading. Page count is read out of the PDF because
   a LaTeX run that overflows to a second page still exits 0.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = os.path.join(ROOT, "research", "runs")
TEX = os.path.join(HERE, "concept-summary.tex")
GEN = os.path.join(HERE, "generated-numbers.tex")
BUILD = os.path.join(HERE, "build")
OUT_PDF = os.path.join(HERE, "concept-summary.pdf")

WORD_MIN, WORD_MAX = 500, 950          # the brief's recommended range


# --------------------------------------------------------------------------- generate

def load(name):
    with open(os.path.join(RUNS, name), encoding="utf-8") as fh:
        return json.load(fh)


def numbers():
    """Every measured figure in the summary, read from the runs that produced it."""
    q = load("quality.json")["models"]
    s = load("structure.json")
    r = load("runtime.json")
    rep = load("replication.json")["models"]

    IN = "in-domain (held-out Europarl)"
    OOD = "out-of-domain (French prose)"
    ours, gpt2 = q["8M BDH (ours)"], q["gpt2"]
    bloom, smol = q["bigscience/bloom-560m"], q["HuggingFaceTB/SmolLM2-135M"]

    # generalisation gap: out-of-domain bits/byte divided by in-domain, per model
    gaps = {k: v["bpb"][OOD] / v["bpb"][IN] for k, v in q.items()}
    others = [g for k, g in gaps.items() if k != "8M BDH (ours)"]

    kv_per_token = 2 * r["config"]["L"] * r["config"]["D"] * r["bytes_per_scalar"]
    at512 = next(row for row in r["rows"] if row["context"] == 512)

    # sparsity is LEARNED: the training run's own telemetry, x_sparsity is the zero fraction
    ev = os.path.join(ROOT, "models", "telemetry", "evolution", "evolution_french.json")
    with open(ev, encoding="utf-8") as fh:
        eva = json.load(fh)
    act = {e["iteration"]: 100 - 100 * sum(p["mean_x_sp"] for p in e["sparsity"]) / len(e["sparsity"])
           for e in eva}
    last = max(act)

    n = {
        # bits per byte, research/compare_quality.py
        "bpbOursIn": f"{ours['bpb'][IN]:.3f}",
        "bpbOursOod": f"{ours['bpb'][OOD]:.3f}",
        "bpbBloomIn": f"{bloom['bpb'][IN]:.3f}",
        "bpbBloomOod": f"{bloom['bpb'][OOD]:.3f}",
        "bpbSmolIn": f"{smol['bpb'][IN]:.3f}",
        "bpbGptIn": f"{gpt2['bpb'][IN]:.3f}",
        "paramsOurs": f"{ours['params'] / 1e6:.1f}",
        "gapOurs": f"{gaps['8M BDH (ours)']:.1f}",
        "gapOthersLo": f"{min(others):.2f}",
        "gapOthersHi": f"{max(others):.2f}",
        # sparsity and structure, research/compare_structure.py
        "bdhActive": f"{100 * s['bdh']['density']['above_1pct_of_peak']:.2f}",
        "bdhAboveZero": f"{100 * s['bdh']['density']['greater_than_zero']:.2f}",
        "gptActive": f"{100 * s['transformer']['density']['english']['above_1pct_of_peak']:.1f}",
        "sparserBy": f"{s['transformer']['density']['english']['above_1pct_of_peak'] / s['bdh']['density']['above_1pct_of_peak']:.1f}",
        "mccGstar": f"{s['bdh']['silence_prediction']['mcc']:.3f}",
        "mccNull": f"{s['bdh']['silence_prediction']['null_mcc_mean']:.3f}",
        "nullDraws": str(s["bdh"]["silence_prediction"]["null_draws"]),
        "isolatedShare": f"{100 * s['bdh']['silence_prediction']['isolated_share']:.1f}",
        "gptIsolated": f"{100 * s['transformer']['silence_prediction']['isolated_share_mean']:.2f}",
        # the gated path and the Portuguese sibling, research/replicate_structure.py
        "yActive": f"{100 * rep['french']['activations']['y_active']:.2f}",
        "mccPt": f"{rep['portuguese']['silence_prediction']['mcc']:.3f}",
        # memory and the recurrence, research/compare_runtime.py and compare_memory.py
        "sigmaMB": f"{r['rows'][0]['bdh_state_mb']:.1f}",
        "crossover": f"{r['predicted_memory_crossover']:,}",
        "kvPerToken": f"{kv_per_token:,}",
        "kvAtTrained": f"{at512['kv_cache_mb']:.1f}",
        "sigmaPremium": f"{at512['bdh_state_mb'] / at512['kv_cache_mb']:.1f}",
        "recurRel": f"{r['recurrence_check']['relative_error']:.1e}".replace("e-07", ""),
        # sparsity is learned, models/telemetry/evolution/
        "actInit": f"{act[0]:.1f}",
        "actFloor": f"{act[2500]:.1f}",
        "actEnd": f"{act[last]:.1f}",
        "floorStep": f"{2500:,}",
    }
    return n


def write_numbers(n):
    lines = [
        "% GENERATED by docs/build_summary.py from research/runs/*.json and models/telemetry/.",
        "% Do not edit. Re-run `npm run summary` after any re-measurement.",
    ]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in n.items()]
    with open(GEN, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return len(n)


# --------------------------------------------------------------------------- build

def build():
    os.makedirs(BUILD, exist_ok=True)
    cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
           "-outdir=build", "concept-summary.tex"]
    p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    log = os.path.join(BUILD, "concept-summary.log")
    if p.returncode != 0:
        sys.stderr.write(p.stdout[-4000:] + "\n" + p.stderr[-2000:] + "\n")
        errs = []
        if os.path.exists(log):
            with open(log, encoding="utf-8", errors="replace") as fh:
                errs = [ln for ln in fh if ln.startswith("!")]
        sys.stderr.write("\n".join(errs[:20]) + "\n")
        raise SystemExit("latexmk failed")
    shutil.copyfile(os.path.join(BUILD, "concept-summary.pdf"), OUT_PDF)
    return log


# --------------------------------------------------------------------------- check

BANNED_WORDS = [
    # slop adjectives and transitions the brief marks down; "significant" is allowed only in the
    # statistical sense, which this page does not use, so it is banned outright here.
    "powerful", "novel", "robust", "seamless", "cutting-edge", "comprehensive",
    "crucial", "significant", "dramatic", "leverage", "delve", "landscape",
    "importantly", "notably", "it is worth noting", "in other words", "that said",
]
FORBIDDEN = [
    # every one of these is a factual error a judge would catch; see CLAUDE.md
    "state-space model", "state space model", "ARC-AGI-2", "ARC Prize verified",
    "independent audit", "third-party audit", "SageMaker", "HyperPod",
    "Contextual Quantization", "contextual quantisation",
]
REQUIRED = [
    "0.00265246",       # the second, unreconciled cost for the same 118/400
    "co-authors",       # the audit was not external
    "ConceptARC",       # in the training mixture and then evaluated on
    "training mixture",
    "remain proprietary",
    "special case",     # additive accumulation is the named special case, not BDH-CQ's rule
    "replayed",         # every BDH-CQ figure is labelled
]
ARXIV = re.compile(r"(\d{2})(\d{2})\.\d{4,5}")


def decomment(src):
    return re.sub(r"(?<!\\)%.*", "", src)


def strip_tex(src):
    """Rough plain text, as a reader meets it.

    Links count once, as their visible label: \\href{url}{label} is one phrase on the page, and
    counting the URL as well would have inflated the budget by roughly a fifth.
    """
    src = decomment(src.split("\\begin{document}", 1)[-1])
    src = re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r"\1", src)
    src = re.sub(r"\\arx\{([^}]*)\}", r"arXiv:\1", src)
    src = re.sub(r"\$[^$]*\$", " ", src)
    src = re.sub(r"\\begin\{gather\*?\}.*?\\end\{gather\*?\}", " ", src, flags=re.S)
    src = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", src)
    return re.sub(r"[{}&\\~^_]", " ", src)


def check(log_path, generated):
    with open(TEX, encoding="utf-8") as fh:
        src = fh.read()
    body = strip_tex(src)
    fails = []

    # 1. exactly one page. latexmk exits 0 on a two-page document, so read the PDF.
    with open(OUT_PDF, "rb") as fh:
        raw = fh.read()
    m = re.search(rb"/Count\s+(\d+)", raw)
    pages = int(m.group(1)) if m else -1
    if pages != 1:
        fails.append(f"page count is {pages}, must be 1")

    # 2. word budget
    words = len(re.findall(r"[A-Za-z][A-Za-z'-]*", body))
    if not WORD_MIN <= words <= WORD_MAX:
        fails.append(f"word count {words} outside [{WORD_MIN}, {WORD_MAX}]")

    # 3. typography and register. Comment banners are allowed their rules of dashes.
    prose = decomment(src)
    if "---" in prose or "\u2014" in prose:
        fails.append("em dash present")
    bad = {c for c in src if ord(c) > 127}
    if bad:
        fails.append("non-ASCII characters present: " + " ".join(sorted(bad)))
    low = body.lower()
    for w in BANNED_WORDS:
        if re.search(rf"\b{re.escape(w)}\b", low):
            fails.append(f"banned word: {w}")

    # 4. the honesty layer, as assertions. Matched against whitespace-collapsed source, because
    # a required phrase is often broken across two source lines.
    flat = " ".join(prose.lower().split())
    for f in FORBIDDEN:
        if " ".join(f.lower().split()) in flat:
            fails.append(f"forbidden claim: {f}")
    for rq in REQUIRED:
        if " ".join(rq.lower().split()) not in flat:
            fails.append(f"missing required disclosure: {rq}")

    # 5. citations beside their claims, not collected at the end
    head = src.split("% FOOTER", 1)[0]
    ids = {i.group(0) for i in ARXIV.finditer(src)}
    in_body = {i.group(0) for i in ARXIV.finditer(head)}
    if len(ids) < 6:
        fails.append(f"{len(ids)} distinct arXiv ids, expected 6")
    if "pathwaycom/bdh" not in src:
        fails.append("the reference implementation is not cited")
    if len(in_body) < 4:
        fails.append(f"only {len(in_body)} arXiv ids appear before the footer, expected 4")
    for i in ids:
        yr = 2000 + int(ARXIV.match(i).group(1))
        if not 2022 <= yr <= 2026:
            fails.append(f"{i} is outside the 2022-2026 window")
    # BDH-CQ figures must carry a locator, so a reader can check the number in one lookup.
    # Counted rather than matched per mention: the masthead names the paper without reporting a
    # figure, and that mention correctly has nothing to locate.
    locs = re.findall(r"(?:\\S|Sec\.|Table|Eq\.)~?\s*\(?\d", src)
    if len(locs) < 6:
        fails.append(f"only {len(locs)} section or table locators, expected 6")

    # 6. layout: overfull boxes mean something is sticking out of its column
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            for ln in fh:
                o = re.match(r"Overfull \\hbox \((\d+\.\d+)pt", ln)
                if o and float(o.group(1)) > 3.0:
                    fails.append(f"overfull hbox {o.group(1)}pt")

    print(f"\n  {pages} page, {words} words, {generated} generated numbers")
    for f in fails:
        print(f"  FAIL  {f}")
    if fails:
        raise SystemExit(f"\nSUMMARY FAILED ({len(fails)})")
    print("  SUMMARY OK\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args()
    generated = write_numbers(numbers())
    log = build()
    if a.no_check:
        print(f"  built {OUT_PDF} ({generated} generated numbers), checks skipped")
    else:
        check(log, generated)


if __name__ == "__main__":
    main()
