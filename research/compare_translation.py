#!/usr/bin/env python
"""chrF on held-out Europarl: our 8M byte-level BDH beside a purpose-built NMT transformer.

WHY THIS SCRIPT EXISTS
----------------------
`compare_quality.py` measures how well each model PREDICTS text. This measures whether ours can
actually do the job it was trained for, on sentences it has never seen, scored the way machine
translation is normally scored -- and puts a real translation system next to it so the number means
something.

THE BASELINE, AND THE EXPECTED RESULT, STATED BEFORE RUNNING
-------------------------------------------------------------
`Helsinki-NLP/opus-mt-en-fr` is a MarianMT encoder-decoder built for exactly this language pair:
about 74M parameters, a trained subword vocabulary, and an architecture whose whole design is
conditional generation. **It should win, and the plan said so before the first run.** Ours is 8M,
decoder-only, byte-level, trained for about 50 minutes on one A100, and it has to spend capacity
learning what a French word even looks like from raw bytes.

So the honest framing is per parameter and per GPU-hour, and the number ships whichever way it
falls. What would NOT be honest is running several baselines and reporting the one we beat.
One baseline was chosen in advance; this is it.

chrF rather than BLEU, because chrF is a character-n-gram F-score and degrades gracefully for a
byte-level model that can get a word almost right. BLEU is reported too, since it is what most
people recognise, and it will look worse for both.

THE EVALUATION SET
------------------
Records are parsed straight out of `data/en-fr/val.bin` -- the last 5% of the byte stream, cut by
the training notebook's own 95/5 split, so these are sentences the model's own validation loss was
computed on and never trained on. The first record is dropped because the split can land mid-record.

A LIMIT WORTH KNOWING BEFORE READING THE OUTPUT
-----------------------------------------------
Our model has no end-of-sequence token. It was trained on an unbroken concatenation of records, so
it ends a translation by starting the next `<F:en>` example -- which is what the decode is cut on.
When it fails to emit that marker the output runs on, and a run-on is punished by chrF roughly as
hard as a wrong translation. That is a real property of the checkpoint, not a scoring artifact, and
it is not corrected for.

USAGE
    python research/compare_translation.py --n 200
    python research/compare_translation.py --skip-hf        # ours only, no downloads
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from bdh_big import load_big  # noqa: E402

REC = re.compile(r"<F:en>(.*?)<T:fr>(.*)", re.S)


def load_pairs(path, n, min_len=20, max_len=180):
    """(english, french) pairs parsed out of the held-out byte stream."""
    if not os.path.exists(path):
        return None
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    text = bytes(raw[: 400 * n + 20000]).decode("utf-8", errors="ignore")
    chunks = text.split("<F:en>")[1:-1]          # first may be truncated, last may be cut short
    pairs = []
    for c in chunks:
        m = REC.match("<F:en>" + c)
        if not m:
            continue
        en, fr = m.group(1).strip(), m.group(2).strip()
        if min_len <= len(en) <= max_len and min_len <= len(fr) <= max_len:
            pairs.append((en, fr))
        if len(pairs) >= n:
            break
    return pairs


@torch.no_grad()
def translate_bdh(model, en, device, max_new=220):
    ids = list(("<F:en>%s<T:fr>" % en).encode("utf-8"))
    start = len(ids)
    for _ in range(max_new):
        logits, _ = model(torch.tensor([ids[-2048:]], dtype=torch.long, device=device))
        ids.append(int(logits[0, -1].argmax()))
        tail = bytes(ids[start:])
        if b"<F:en>" in tail:                     # the model's only way to end: start a new example
            return tail.split(b"<F:en>")[0].decode("utf-8", errors="replace").strip()
    return bytes(ids[start:]).decode("utf-8", errors="replace").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "models/checkpoints/french_best.pt"))
    ap.add_argument("--val", default=os.path.join(ROOT, "data/en-fr/val.bin"))
    ap.add_argument("--hf", default="Helsinki-NLP/opus-mt-en-fr")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--skip-hf", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "research/runs/translation.json"))
    args = ap.parse_args()

    pairs = load_pairs(args.val, args.n)
    if not pairs:
        print("no %s -- run research/compare_quality.py once to fetch and build the corpus"
              % os.path.relpath(args.val, ROOT))
        return 1
    src = [p[0] for p in pairs]
    ref = [p[1] for p in pairs]
    print("  %d held-out pairs, mean source length %.0f bytes"
          % (len(pairs), np.mean([len(s) for s in src])))

    import sacrebleu
    model, cfg, _ = load_big(args.ckpt, device=args.device)
    ours = [translate_bdh(model, s, args.device) for s in src]
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()

    scored = {"8M BDH (ours)": {
        "params": 7962624,
        "chrf": round(sacrebleu.corpus_chrf(ours, [ref]).score, 2),
        "bleu": round(sacrebleu.corpus_bleu(ours, [ref]).score, 2),
        "mean_output_bytes": round(float(np.mean([len(o.encode("utf-8")) for o in ours])), 1),
    }}
    print("  ours          chrF %.2f   BLEU %.2f" % (scored["8M BDH (ours)"]["chrf"],
                                                     scored["8M BDH (ours)"]["bleu"]))

    theirs = None
    if not args.skip_hf:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.hf)
        mt = AutoModelForSeq2SeqLM.from_pretrained(args.hf).to(args.device).eval()
        theirs = []
        with torch.no_grad():
            for i in range(0, len(src), 16):
                batch = tok(src[i:i + 16], return_tensors="pt", padding=True,
                            truncation=True).to(args.device)
                theirs += tok.batch_decode(mt.generate(**batch, max_new_tokens=200),
                                           skip_special_tokens=True)
        scored[args.hf] = {
            "params": sum(p.numel() for p in mt.parameters()),
            "chrf": round(sacrebleu.corpus_chrf(theirs, [ref]).score, 2),
            "bleu": round(sacrebleu.corpus_bleu(theirs, [ref]).score, 2),
            "mean_output_bytes": round(float(np.mean([len(o.encode("utf-8")) for o in theirs])), 1),
        }
        del mt
        print("  %-12s chrF %.2f   BLEU %.2f" % (args.hf.split("/")[-1], scored[args.hf]["chrf"],
                                                 scored[args.hf]["bleu"]))

    samples = [{"en": src[i], "ours": ours[i],
                "baseline": theirs[i] if theirs else None, "reference": ref[i]}
               for i in range(min(8, len(src)))]
    out = {
        "pairs": len(pairs),
        "source": "data/en-fr/val.bin -- the notebook's own 95/5 split, never trained on",
        "baseline_chosen_before_running": args.hf,
        "caveat": ("Our model has no end-of-sequence token: it ends a translation by starting the "
                   "next <F:en> example, and a run-on when it fails to is punished by chrF about "
                   "as hard as a wrong translation. Not corrected for."),
        "scores": scored,
        "samples": samples,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2)

    print("")
    for s in samples[:4]:
        print("  EN   ", s["en"][:78])
        print("  ours ", s["ours"][:78])
        if s["baseline"]:
            print("  base ", s["baseline"][:78])
        print("  ref  ", s["reference"][:78])
        print("")
    if theirs:
        a, b = scored["8M BDH (ours)"], scored[args.hf]
        print("  chrF per million parameters:  ours %.2f   %s %.2f"
              % (a["chrf"] / (a["params"] / 1e6), args.hf.split("/")[-1],
                 b["chrf"] / (b["params"] / 1e6)))
    print("  wrote", os.path.relpath(args.out, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
