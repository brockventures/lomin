#!/usr/bin/env python3
"""Benchmark: eager vs CUDA-graph T3 decode in desktop_tts.py.

Loads the model exactly as the service does (same env: MDT_CHATTERBOX_REF_AUDIO,
MDT_CHATTERBOX_T3_WEIGHTS, HF_HUB_OFFLINE=1), then for 5 fixed sentences reports,
per mode (eager / fast):
  ms/token        steady-state T3 time per token (first token excluded)
  tok/s           1000 / ms-per-token
  TTFT            prompt -> first speech token (prefill + first sample)
  first audio     prompt -> first PCM chunk out of stream_pcm_chunks
  total synth     prompt -> last PCM chunk (whole streamed sentence)
Medians over --reps runs. Optionally writes one streamed WAV per mode/sentence
to --wav-dir for listening, and checks the audio for NaN/inf.

Run (does NOT bind any port; shares the GPU with the live service):
  <venv>/bin/python bench_t3.py [--reps 3] [--wav-dir DIR] [--dtype fp32|bf16|fp16]
"""
from __future__ import annotations

import argparse
import os
import statistics as st
import sys
import time
import wave
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

SENTENCES = [
    "Good evening.",
    "The garage door has been closed.",
    "The forecast says rain by midnight, so I moved the patio cushions inside.",
    "Your first meeting tomorrow is at nine thirty, and the drive is about twenty five minutes with current traffic.",
    "I checked the house before you got home: the front door is locked, the thermostat is at sixty eight degrees, and the dishwasher finished its cycle about an hour ago.",
]


def med(xs):
    return st.median(xs) if xs else float("nan")


def run_tokens(dt, model, text):
    """Returns (n_tokens, ttft_s, ms_per_token)."""
    import torch
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    stamps = []
    for _ in dt._t3_token_iter(model, text):
        stamps.append(time.perf_counter())
    torch.cuda.synchronize()
    n = len(stamps)
    ttft = stamps[0] - t0
    mpt = (stamps[-1] - stamps[0]) / (n - 1) * 1000 if n > 1 else float("nan")
    return n, ttft, mpt


def run_stream(dt, text):
    t0 = time.perf_counter()
    first = None
    chunks = []
    for c in dt.stream_pcm_chunks(text):
        if first is None:
            first = time.perf_counter() - t0
        chunks.append(c)
    return first, time.perf_counter() - t0, b"".join(chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--wav-dir", default="")
    ap.add_argument("--dtype", default=os.environ.get("MDT_T3_FAST_DTYPE", "fp32"))
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    os.environ["MDT_T3_FAST"] = "1"
    os.environ["MDT_T3_FAST_DTYPE"] = args.dtype
    import numpy as np
    import torch
    import desktop_tts as dt

    dt._load_chatterbox()
    model = dt._chatterbox_model
    if model is None:
        sys.exit(f"chatterbox failed to load: {dt._chatterbox_load_error}")
    fast = getattr(model, "_mdt_t3_fast", None)
    if fast is None:
        sys.exit("fast decoder did not build (see stderr); nothing to compare")
    modes = {"eager": None, "fast": fast}
    wav_dir = Path(args.wav_dir) if args.wav_dir else None
    if wav_dir:
        wav_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for si, text in enumerate(SENTENCES):
        row = {}
        for name, dec in modes.items():
            model._mdt_t3_fast = dec
            run_tokens(dt, model, text)  # per-mode warm run
            res = {"n": [], "ttft": [], "mpt": [], "fa": [], "tot": []}
            for r in range(args.reps):
                torch.manual_seed(args.seed + r)
                n, ttft, mpt = run_tokens(dt, model, text)
                torch.manual_seed(args.seed + r)
                fa, tot, pcm = run_stream(dt, text)
                res["n"].append(n); res["ttft"].append(ttft); res["mpt"].append(mpt)
                res["fa"].append(fa); res["tot"].append(tot)
            audio = np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0
            res["dur"] = len(audio) / model.sr
            res["finite"] = bool(np.isfinite(audio).all())
            if wav_dir:
                with wave.open(str(wav_dir / f"s{si + 1}_{name}.wav"), "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(model.sr)
                    w.writeframes(pcm)
            row[name] = res
        rows.append(row)
    model._mdt_t3_fast = fast

    print(f"\ndtype={args.dtype} reps={args.reps} (medians; GPU shared with live service)\n")
    hdr = f"{'sentence':<9}{'chars':>6}{'mode':>7}{'tokens':>8}{'ms/tok':>8}{'tok/s':>8}{'TTFT ms':>9}{'1st audio ms':>14}{'total s':>9}{'audio s':>9}{'finite':>8}"
    print(hdr)
    print("-" * len(hdr))
    for si, row in enumerate(rows):
        for name in modes:
            r = row[name]
            mpt = med(r["mpt"])
            print(f"{'S' + str(si + 1):<9}{len(SENTENCES[si]):>6}{name:>7}{med(r['n']):>8.0f}{mpt:>8.1f}"
                  f"{1000 / mpt:>8.1f}{med(r['ttft']) * 1000:>9.0f}{med(r['fa']) * 1000:>14.0f}"
                  f"{med(r['tot']):>9.2f}{r['dur']:>9.2f}{str(r['finite']):>8}")
    for key, label in (("mpt", "ms/token"), ("fa", "first audio (s)"), ("tot", "total (s)")):
        e = med([med(r["eager"][key]) for r in rows])
        f = med([med(r["fast"][key]) for r in rows])
        print(f"overall median {label}: eager {e:.3f}  fast {f:.3f}  ({e / f:.2f}x)")


if __name__ == "__main__":
    main()
