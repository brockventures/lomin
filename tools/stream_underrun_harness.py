#!/usr/bin/env python3
"""Measure /tts/stream pacing: records byte arrival times and computes, for a
listener who starts playback P ms after the first byte, the underrun count and
worst deficit (seconds of silence while waiting for the next audio).

    python3 tools/stream_underrun_harness.py --port 9197 [--json out.json]

Stdlib only. Requests format=pcm so every byte is audio (s16le, 24 kHz mono).
"""
import argparse
import json
import socket
import statistics
import time

SENTENCES = [
    "Good morning, Alice. I have gone through your overnight messages and nothing needs you right away.",
    "The garage host is healthy, and the desktop finished the nightly builds without any failures.",
    "Bob's calendar shows a dentist appointment at ten, so I will move the plumber to the afternoon.",
    "It should rain around four o'clock, which means the garden can skip watering today.",
    "If you want, I can read the headlines while the coffee finishes brewing.",
]
PREBUFFERS_MS = (200, 300, 500)
BYTES_PER_SEC = 24000 * 2


def request_stream(port: int, text: str, host: str = "127.0.0.1"):
    """Returns events = [(t_rel_to_request, nbytes)] of the audio body
    (headers excluded)."""
    body = json.dumps({"text": text, "format": "pcm"}).encode()
    s = socket.create_connection((host, port), timeout=120)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    t0 = time.perf_counter()
    s.sendall(b"POST /tts/stream HTTP/1.0\r\nContent-Type: application/json\r\n"
              b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
    buf = b""
    events = []
    header_done = False
    while True:
        data = s.recv(65536)
        now = time.perf_counter() - t0
        if not data:
            break
        if not header_done:
            buf += data
            i = buf.find(b"\r\n\r\n")
            if i < 0:
                continue
            data = buf[i + 4:]
            header_done = True
            if not data:
                continue
        events.append((now, len(data)))
    s.close()
    return events


def underruns(events, prebuffer_s: float, bytes_per_sec: int = BYTES_PER_SEC):
    """Simulate playback that starts prebuffer_s after the first byte.
    Returns (count, worst_deficit_s, deficits). A stall ends when the next
    chunk arrives and playback resumes from there."""
    if not events:
        return 0, 0.0, []
    t_first = events[0][0]
    ready_until = None  # wall time at which all received audio has been played
    deficits = []
    for t, n in events:
        dur = n / bytes_per_sec
        if ready_until is None:
            ready_until = t_first + prebuffer_s + dur
            continue
        if t > ready_until + 1e-9:
            deficits.append(t - ready_until)
            ready_until = t + dur
        else:
            ready_until += dur
    return len(deficits), (max(deficits) if deficits else 0.0), deficits


def summarize(events):
    total = sum(n for _, n in events)
    return {
        "ttfb_s": events[0][0] if events else None,
        "audio_s": total / BYTES_PER_SEC,
        "total_s": events[-1][0] if events else None,
        "chunks": len(events),
        "arrivals": [(round(t, 3), round(n / BYTES_PER_SEC, 3)) for t, n in events],
        "underruns": {str(p): dict(zip(("count", "worst_s"), underruns(events, p / 1000)[:2]))
                      for p in PREBUFFERS_MS},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9197)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--json")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    request_stream(a.port, "Warm up.")
    out = {"label": a.label, "sentences": [], "paragraph": []}
    for _ in range(a.reps):
        out["sentences"].append([summarize(request_stream(a.port, s)) for s in SENTENCES])
        out["paragraph"].append(summarize(request_stream(a.port, " ".join(SENTENCES))))
    print(f"== {a.label}")
    ttfbs = [r["ttfb_s"] for rep in out["sentences"] for r in rep]
    print(f"TTFB per sentence: median {statistics.median(ttfbs):.2f}s max {max(ttfbs):.2f}s; "
          f"paragraph {statistics.median(r['ttfb_s'] for r in out['paragraph']):.2f}s")
    for p in PREBUFFERS_MS:
        k = str(p)
        sc = [r["underruns"][k]["count"] for rep in out["sentences"] for r in rep]
        sw = [r["underruns"][k]["worst_s"] for rep in out["sentences"] for r in rep]
        pc = [r["underruns"][k]["count"] for r in out["paragraph"]]
        pw = [r["underruns"][k]["worst_s"] for r in out["paragraph"]]
        print(f"P={p}ms  per-sentence: mean underruns {statistics.mean(sc):.2f}, worst {max(sw):.2f}s | "
              f"paragraph ({out['paragraph'][0]['audio_s']:.1f}s audio): "
              f"mean underruns {statistics.mean(pc):.1f}, worst {max(pw):.2f}s")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
