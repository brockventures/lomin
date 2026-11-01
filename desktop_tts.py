#!/usr/bin/env python3
"""Desktop-side TTS server for the LÓMIN voice gateway.

Runs on a desktop host with a GPU, bound to 127.0.0.1:9097 by default. A
host that cannot reach this desktop directly (NAT-only, e.g. WSL2) opens an
ssh reverse tunnel (`ssh -N -R 127.0.0.1:9097:127.0.0.1:9097 <gateway-host>`)
so the gateway's own 127.0.0.1:9097 forwards here. See README.md.

Why this exists: the gateway host's own local Piper takes ~19-20s to synthesize
one sentence under load (load average ~9), which blows past the calling hub's
TTS timeout (default 10s). The desktop is otherwise idle and has a real GPU
(RTX 4090), so it's a much faster first hop in gateway.py's
desktop -> local-Piper chain.

Endpoints:
    POST /tts     JSON {"text": "..."} -> audio/wav bytes,
                  X-TTS-Engine: chatterbox|piper response header
    GET  /health  -> {"status": "ok", "engine": "chatterbox"|"piper", ...}

Two engines, tried in order:

1. **Chatterbox Turbo** — zero-shot voice cloning
   (ResembleAI/chatterbox-turbo, MIT), cloned onto the agent's voice using
   a reference clip of about 12s at
   $MVG_HOME/voices/default/ref.wav (see MDT_CHATTERBOX_REF_AUDIO). The model and its
   `prepare_conditionals()` embedding are loaded ONCE at process start —
   both are expensive (multi-second) and must never happen per request.
   Runs on CUDA. A prior Chatterbox (non-Turbo) integration was fully
   deprecated 2026-08-28 for reliability/ops reasons (flapping restarts,
   not voice quality) — see git history on this file; this is a fresh,
   simpler integration with no probe/restart machinery, just a try/except
   fallback to Piper.
2. **Piper** (`en_US-lessac-medium`, generic voice) — the fallback if
   Chatterbox fails to load or a specific synthesis call raises. Loaded
   ONCE at process start (piper.PiperVoice.load), not per request —
   reloading the ~63MB onnx model on every call is exactly the latency
   this server exists to avoid.

Both engines synthesize under their own lock; neither is documented as
thread-safe for concurrent inference, and this server only ever needs to
serve one voice conversation at a time.

Runs under its own venv (Python 3.11) — needs `torch`
(CUDA build), `chatterbox-tts`, and `piper-tts` all in the same venv, since
this one process now hosts both engines. Never the system Python.

Config is read at runtime, never copied into any worktree:
    $MVG_HOME/piper-voices/<model>.onnx(.json)   (MVG_HOME default ~/.mvg)
    $MVG_HOME/voices/default/ref.wav
"""

from __future__ import annotations

import io
import json
import os
import queue
import sys
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sentence_segmenter import segment_full_text  # noqa: E402

PORT = int(os.environ.get("MDT_PORT", "9097"))
BIND = os.environ.get("MDT_BIND", "127.0.0.1")

VOICES_DIR = Path(os.environ.get("MDT_VOICES_DIR") or (Path(os.environ.get("MVG_HOME") or (Path.home() / ".mvg")) / "piper-voices"))
VOICE_MODEL = os.environ.get("MDT_VOICE_MODEL", "en_US-lessac-medium")

CHATTERBOX_ENABLED = os.environ.get("MDT_CHATTERBOX_ENABLED", "1") != "0"
CHATTERBOX_DEVICE = os.environ.get("MDT_CHATTERBOX_DEVICE", "cuda")
# MANDOS (2026-09-30): optional fine-tuned T3 weights loaded over the base
# Turbo model. Unset = stock Turbo. Built by projects/mandos/train.py.
CHATTERBOX_T3_WEIGHTS = os.environ.get("MDT_CHATTERBOX_T3_WEIGHTS", "")
CHATTERBOX_REF_AUDIO = Path(
    os.environ.get("MDT_CHATTERBOX_REF_AUDIO")
    or (Path(os.environ.get("MVG_HOME") or (Path.home() / ".mvg")) / "voices" / "default" / "ref.wav")
)
CHATTERBOX_WARMUP_TEXT = "System check, please disregard."
# Chatterbox Turbo's T3 stage is autoregressive over speech tokens, so
# latency grows with OUTPUT length, not a fixed per-call cost. Measured on
# this 4090 (warm calls, GPU util 15-45% — the
# bottleneck is per-step kernel-launch overhead on a small autoregressive
# batch, not raw compute): ~10 chars -> ~0.7-1.3s, ~107 chars -> ~4.4-5.5s,
# ~168 chars -> ~6.2-6.7s, ~341 chars (a full 60-word /ask reply, the hub's
# own cap) -> ~10.5-14.6s. MVG_DESKTOP_TTS_TIMEOUT is 8s and hub.go's own
# TTS timeout is 10s (config.GetTTSTimeoutSeconds) — a Chatterbox call past
# either doesn't just fall back to Piper, it falls all the way to this
# gateway host's own local Piper (~19-20s), which *also* blows the hub timeout, so
# the result is silence: worse than today's sub-second desktop Piper. This
# cap keeps Chatterbox for the calls it can safely finish and routes
# anything longer straight to Piper instead of gambling on the timeout.
# Decision (2026-09-28): the cloned voice is the default for all
# lengths. 0 = no cap; the gateway and hub timeouts were raised to fit the
# ~15 s worst case instead. Set MDT_CHATTERBOX_MAX_CHARS to restore a cap.
CHATTERBOX_MAX_CHARS = int(os.environ.get("MDT_CHATTERBOX_MAX_CHARS", "0"))

# CUDA-graph T3 decode (t3_fast.py). Built once at model load; any failure
# there or mid-generation falls back to the eager loop below (one stderr line).
# Measured: ~26 ms/token eager -> ~9 ms/token fp32 graph (bench_t3.py).
T3_FAST = os.environ.get("MDT_T3_FAST", "1") != "0"
T3_FAST_DTYPE = os.environ.get("MDT_T3_FAST_DTYPE", "fp32")  # fp32 | bf16 | fp16
T3_FAST_MAX_LEN = int(os.environ.get("MDT_T3_FAST_MAX_LEN", "2048"))

_piper_voice = None
_piper_lock = threading.Lock()
_piper_load_error: str | None = None

_chatterbox_model = None
_chatterbox_lock = threading.Lock()
_chatterbox_load_error: str | None = None


def _load_piper():
    """Imports piper and loads the voice ONCE. Import happens here, not at
    module top, so this file can still be imported (e.g. by a test) under
    an interpreter that doesn't have piper-tts installed."""
    global _piper_voice, _piper_load_error
    try:
        from piper import PiperVoice  # noqa: E402 (deliberately lazy import)
    except Exception as exc:  # pragma: no cover - only hit with a bad venv
        _piper_load_error = f"piper-tts not importable: {exc}"
        print(f"[desktop-tts] {_piper_load_error}", file=sys.stderr, flush=True)
        return

    model_path = VOICES_DIR / f"{VOICE_MODEL}.onnx"
    config_path = VOICES_DIR / f"{VOICE_MODEL}.onnx.json"
    if not model_path.exists():
        _piper_load_error = f"voice model not found: {model_path}"
        print(f"[desktop-tts] {_piper_load_error}", file=sys.stderr, flush=True)
        return

    t0 = time.time()
    _piper_voice = PiperVoice.load(
        str(model_path),
        config_path=str(config_path) if config_path.exists() else None,
    )
    print(f"[desktop-tts] loaded piper {VOICE_MODEL} in {time.time() - t0:.2f}s", flush=True)


def _load_chatterbox():
    """Imports chatterbox-tts and loads Chatterbox Turbo + the agent's cloned
    conditionals ONCE. Both `from_pretrained` (downloads/loads the model)
    and `prepare_conditionals` (embeds the reference audio) are
    multi-second operations — never call either per request. Runs a single
    warmup synthesis afterward so the first real request isn't the one
    that pays for CUDA kernel compilation / lazy allocator warmup."""
    global _chatterbox_model, _chatterbox_load_error
    if not CHATTERBOX_ENABLED:
        _chatterbox_load_error = "disabled (MDT_CHATTERBOX_ENABLED=0)"
        return
    if not CHATTERBOX_REF_AUDIO.exists():
        _chatterbox_load_error = f"reference audio not found: {CHATTERBOX_REF_AUDIO}"
        print(f"[desktop-tts] {_chatterbox_load_error}", file=sys.stderr, flush=True)
        return
    try:
        from chatterbox.tts_turbo import ChatterboxTurboTTS  # noqa: E402
    except Exception as exc:  # pragma: no cover - only hit with a bad venv
        _chatterbox_load_error = f"chatterbox-tts not importable: {exc}"
        print(f"[desktop-tts] {_chatterbox_load_error}", file=sys.stderr, flush=True)
        return

    try:
        t0 = time.time()
        model = ChatterboxTurboTTS.from_pretrained(device=CHATTERBOX_DEVICE)
        print(f"[desktop-tts] loaded chatterbox-turbo in {time.time() - t0:.2f}s", flush=True)
        if CHATTERBOX_T3_WEIGHTS:
            from safetensors.torch import load_file  # noqa: E402
            model.t3.load_state_dict(load_file(CHATTERBOX_T3_WEIGHTS), strict=False)
            model.t3.to(CHATTERBOX_DEVICE).eval()
            print(f"[desktop-tts] loaded fine-tuned T3 from {CHATTERBOX_T3_WEIGHTS}", flush=True)
        t0 = time.time()
        model.prepare_conditionals(str(CHATTERBOX_REF_AUDIO))
        print(
            f"[desktop-tts] cloned agent voice from {CHATTERBOX_REF_AUDIO} "
            f"in {time.time() - t0:.2f}s",
            flush=True,
        )
        _setup_t3_fast(model)
        t0 = time.time()
        model.generate(CHATTERBOX_WARMUP_TEXT)
        print(f"[desktop-tts] chatterbox warmup synthesis in {time.time() - t0:.2f}s", flush=True)
        _chatterbox_model = model
    except Exception as exc:
        _chatterbox_load_error = f"chatterbox load/clone failed: {exc}"
        print(f"[desktop-tts] {_chatterbox_load_error}", file=sys.stderr, flush=True)


def _setup_t3_fast(model) -> None:
    """Captures the CUDA-graph T3 decoder onto `model` (model._mdt_t3_fast)
    and runs one short streamed generation so the first real request does not
    pay for graph/prefill warmup. Never raises: failure leaves the eager path."""
    model._mdt_t3_fast = None
    if not T3_FAST:
        return
    try:
        from t3_fast import try_build
        t0 = time.time()
        dec = try_build(model.t3, device=CHATTERBOX_DEVICE, max_len=T3_FAST_MAX_LEN,
                        dtype_name=T3_FAST_DTYPE)
        if dec is None:
            return
        model._mdt_t3_fast = dec
        n = sum(1 for _ in zip(range(8), _t3_token_iter(model, CHATTERBOX_WARMUP_TEXT)))
        if model._mdt_t3_fast is None:  # the warm run itself fell back
            return
        print(f"[desktop-tts] T3 CUDA-graph decoder ready ({T3_FAST_DTYPE}, max_len={T3_FAST_MAX_LEN}) "
              f"in {time.time() - t0:.2f}s", flush=True)
    except Exception as exc:  # noqa: BLE001
        model._mdt_t3_fast = None
        print(f"[desktop-tts] T3 fast path unavailable, using eager: {exc!r}", file=sys.stderr, flush=True)


def _chatterbox_wav_bytes(text: str) -> bytes:
    if _chatterbox_model is None:
        raise RuntimeError(_chatterbox_load_error or "chatterbox not loaded")
    with _chatterbox_lock:
        wav_tensor = _chatterbox_model.generate(text)
    wav = wav_tensor.squeeze(0).detach().cpu().numpy()
    wav = wav.clip(-1.0, 1.0)
    pcm16 = (wav * 32767.0).astype("int16")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(_chatterbox_model.sr)
        wav_file.writeframes(pcm16.tobytes())
    return buf.getvalue()


def _piper_wav_bytes(text: str) -> bytes:
    if _piper_voice is None:
        raise RuntimeError(_piper_load_error or "piper voice not loaded")
    buf = io.BytesIO()
    with _piper_lock:
        with wave.open(buf, "wb") as wav_file:
            _piper_voice.synthesize_wav(text, wav_file)
    return buf.getvalue()


def synthesize_wav_bytes(text: str) -> tuple[bytes, str]:
    """Returns (wav_bytes, engine). Chatterbox first (the agent's cloned voice)
    for anything short enough to finish inside the TTS timeout budget (see
    CHATTERBOX_MAX_CHARS above), Piper otherwise — either because the text
    is too long, the model isn't loaded, or a specific call raised (CUDA
    OOM, a bad input, etc.)."""
    if _chatterbox_model is not None and (CHATTERBOX_MAX_CHARS <= 0 or len(text) <= CHATTERBOX_MAX_CHARS):
        try:
            return _chatterbox_wav_bytes(text), "chatterbox"
        except Exception as exc:
            print(f"[desktop-tts] chatterbox synth failed, falling back to piper: {exc}",
                  file=sys.stderr, flush=True)
    return _piper_wav_bytes(text), "piper"


# --- sentence-parallel synthesis for long inputs ----------------------------
# MANDOS build (specs/2026-09-28-mandos-streaming.md): "sentence-chunk the
# TTS server itself for any long single input" -- this is the server whose
# own comment above already explains WHY: Chatterbox Turbo's latency scales
# with output length (measured ~0.7-1.3s at ~10 chars up to ~10.5-14.6s at
# ~341 chars), so splitting a long reply into sentences and synthesizing
# them concurrently generally beats one long serial call.

SENTENCE_CHUNK_MIN_CHARS = int(os.environ.get("MDT_SENTENCE_CHUNK_MIN_CHARS", "120"))
SENTENCE_CHUNK_MAX_WORKERS = int(os.environ.get("MDT_SENTENCE_CHUNK_MAX_WORKERS", "4"))


def _concat_wavs(wav_list: list[bytes]) -> bytes:
    """Same contract as gateway.py's _concat_wavs: raises ValueError on any
    format mismatch between chunks rather than risk a garbled/mis-pitched
    concatenation."""
    params = None
    frames = []
    for wav_bytes in wav_list:
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            p = (w.getnchannels(), w.getsampwidth(), w.getframerate())
            if params is None:
                params = p
            elif p != params:
                raise ValueError(f"wav format mismatch: {p} != {params}")
            frames.append(w.readframes(w.getnframes()))
    if params is None:
        raise ValueError("no wav chunks to concatenate")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(params[0])
        out.setsampwidth(params[1])
        out.setframerate(params[2])
        out.writeframes(b"".join(frames))
    return buf.getvalue()


def synthesize_wav_bytes_chunked(text: str) -> tuple[bytes, str]:
    """Sentence-parallel variant of synthesize_wav_bytes(): splits long
    text into sentences, synthesizes them concurrently, and concatenates.
    Falls back to a single serial call for short/single-sentence text, or
    when the concurrent calls produced inconsistent formats (e.g.
    Chatterbox succeeding for one sentence and falling back to Piper,
    which runs at a different sample rate, for another)."""
    if len(text) < SENTENCE_CHUNK_MIN_CHARS:
        return synthesize_wav_bytes(text)
    sentences = segment_full_text(text)
    if len(sentences) <= 1:
        return synthesize_wav_bytes(text)
    with ThreadPoolExecutor(max_workers=min(SENTENCE_CHUNK_MAX_WORKERS, len(sentences))) as ex:
        results = list(ex.map(synthesize_wav_bytes, sentences))
    engines = {engine for _, engine in results}
    try:
        combined = _concat_wavs([wav for wav, _ in results])
    except ValueError as exc:
        print(f"[desktop-tts] chunked concat mismatch, falling back to serial: {exc}",
              file=sys.stderr, flush=True)
        return synthesize_wav_bytes(text)
    engine_label = results[0][1] if len(engines) == 1 else "mixed:" + "+".join(sorted(engines))
    return combined, engine_label


# --- token-window streaming (POST /tts/stream) ------------------------------
# Decision (2026-09-30): priority is audio that STARTS sooner, not a
# faster model. Profile (RTX 4090, warm, median of 3): T3 token generation is
# 92-96% of a /tts call (~26 ms/token, ~38 tokens/s, kernel-launch bound) and
# S3Gen+vocoder is a flat ~0.2-0.25 s however long the text is. Audio plays
# at 25 tokens/s, so T3 outruns playback ~1.5x: if the first chunk can be
# emitted after a handful of tokens, the rest stays ahead of the listener.
#
# Approach: re-implement T3.inference_turbo's sampling loop as a generator
# (same processors, same sampling -> same voice), and every N tokens run
# S3Gen on a window [context tokens | new tokens | lookahead tokens]. Only the
# "new" slice of that window's audio is emitted (context absorbs the window-
# start fade, lookahead absorbs the no-future-context end effects), plus a
# short overlap that is crossfaded with the next window's head so the join is
# inaudible. A sentence is one T3 run (as /tts already does for long text).

STREAM_ENABLED = os.environ.get("MDT_STREAM_ENABLED", "1") != "0"
STREAM_FIRST_TOKENS = int(os.environ.get("MDT_STREAM_FIRST_TOKENS", "12"))
STREAM_CHUNK_TOKENS = int(os.environ.get("MDT_STREAM_CHUNK_TOKENS", "30"))
STREAM_CONTEXT_TOKENS = int(os.environ.get("MDT_STREAM_CONTEXT_TOKENS", "8"))
STREAM_LOOKAHEAD_TOKENS = int(os.environ.get("MDT_STREAM_LOOKAHEAD_TOKENS", "6"))
STREAM_OVERLAP_SAMPLES = int(os.environ.get("MDT_STREAM_OVERLAP_SAMPLES", "720"))
STREAM_WATERMARK = os.environ.get("MDT_STREAM_WATERMARK", "1") != "0"
# Pacing of windows after the first (see WindowPacer): "lead" sizes each window
# from the measured buffered-audio lead, "fixed" uses RAMP then CHUNK tokens.
STREAM_PACING = os.environ.get("MDT_STREAM_PACING", "lead")
# fixed mode: window sizes after the first, e.g. "16,20,24" -> 12,16,20,24,30,30..
STREAM_RAMP_TOKENS = os.environ.get("MDT_STREAM_RAMP_TOKENS", "")
# lead mode: the listener prebuffer the schedule protects, the safety margin
# kept on top of each window's expected arrival, the smallest window allowed
# (each window costs a flat S3Gen decode, so tiny windows lose lead), and the
# size used when the lead cannot cover even that (0 = the max chunk), and the
# throughput priors used until this process has measured its own.
STREAM_ASSUMED_PREBUFFER_MS = int(os.environ.get("MDT_STREAM_ASSUMED_PREBUFFER_MS", "300"))
STREAM_MARGIN_MS = int(os.environ.get("MDT_STREAM_MARGIN_MS", "150"))
STREAM_MIN_CHUNK_TOKENS = int(os.environ.get("MDT_STREAM_MIN_CHUNK_TOKENS", "16"))
STREAM_BEHIND_CHUNK_TOKENS = int(os.environ.get("MDT_STREAM_BEHIND_CHUNK_TOKENS", "0"))
STREAM_PRIOR_T3_TOKS_PER_S = float(os.environ.get("MDT_STREAM_PRIOR_T3_TPS", "45"))
STREAM_PRIOR_DECODE_S = float(os.environ.get("MDT_STREAM_PRIOR_DECODE_MS", "230")) / 1000.0
SAMPLES_PER_TOKEN = 960  # 24 kHz / 25 speech tokens per second
SPEECH_VOCAB = 6561  # ids >= this are special (stop etc.), dropped like generate() does
T3_MAX_GEN_LEN = 1000


def parse_ramp(spec: str) -> list[int]:
    """"16,20,24" -> [16, 20, 24]; junk and non-positive entries are dropped."""
    out = []
    for part in (spec or "").split(","):
        part = part.strip()
        if part.isdigit() and int(part) > 0:
            out.append(int(part))
    return out


class WindowPacer:
    """Decides how many new tokens the next window covers.

    The first window is STREAM_FIRST_TOKENS (time-to-first-byte). Every later
    window i costs  size / t3_rate + decode  of wall time (T3 and S3Gen run
    serially) and yields  size / 25  seconds of audio, so a fixed size either
    outruns a listener who only has a small prebuffer (the second window
    stalled: ~0.5 s of audio against ~1 s of work) or wastes lead on tiny
    windows. "lead" mode picks, per window, the largest size (capped at
    `max_chunk`, floored at `min_chunk`) whose expected arrival still leaves
    `margin_s` of audio in the listener's buffer, using the measured T3 rate
    and decode time (EMAs) and the audio emitted so far. "fixed" mode just
    walks first, ramp..., chunk, chunk... .

    The clock is injected so the logic is testable."""

    SECONDS_PER_TOKEN = 1.0 / 25.0

    def __init__(self, first: int, max_chunk: int, *, mode: str = "lead",
                 ramp: list[int] | None = None, prebuffer_s: float = 0.3,
                 margin_s: float = 0.15, min_chunk: int = 16,
                 t3_rate: float = 45.0, decode_s: float = 0.23, behind_chunk: int | None = None,
                 clock=time.monotonic):
        self.first = max(1, first)
        self.max_chunk = max(1, max_chunk)
        self.mode = mode
        self.ramp = list(ramp or [])
        self.prebuffer_s = prebuffer_s
        self.margin_s = margin_s
        self.min_chunk = max(1, min(min_chunk, self.max_chunk))
        self.behind_chunk = max(1, behind_chunk if behind_chunk else self.max_chunk)
        self.t3_rate = t3_rate
        self.decode_s = decode_s
        self.clock = clock
        self.index = 0  # windows handed out so far
        self.t_first = None  # when the first audio was emitted
        self.audio_s = 0.0  # audio emitted so far (seconds)
        self._t_mark = None  # when the previous window finished decoding

    def next_size(self) -> int:
        i = self.index
        self.index += 1
        if i == 0:
            return self.first
        if self.mode != "lead" or self.t_first is None:
            return self.ramp[i - 1] if i - 1 < len(self.ramp) else self.max_chunk
        lead = self.t_first + self.prebuffer_s + self.audio_s - self.clock()
        size = int(self.t3_rate * (lead - self.decode_s - self.margin_s))
        if size < self.min_chunk:
            # Not enough lead for even a small window: this one arrives late
            # whatever its size, so take the size that buys the most audio
            # per unit of decode overhead and rebuild the lead fastest.
            return self.behind_chunk
        return min(self.max_chunk, size)

    def window_started(self):
        """Call when the T3 loop starts generating toward a window."""
        if self._t_mark is None:
            self._t_mark = self.clock()

    def window_done(self, new_tokens: int, decode_s: float, audio_s: float):
        """Call right after a window's audio is emitted. new_tokens is how
        many tokens T3 produced for it (including lookahead carried in)."""
        now = self.clock()
        if self.t_first is None:
            self.t_first = now
        self.audio_s += audio_s
        if self._t_mark is not None and new_tokens > 0:
            t3_s = max(1e-3, now - self._t_mark - decode_s)
            self.t3_rate = 0.6 * self.t3_rate + 0.4 * (new_tokens / t3_s)
        self.decode_s = 0.6 * self.decode_s + 0.4 * decode_s
        self._t_mark = now


def _t3_token_iter(model, text: str, cancel: threading.Event | None = None):
    """Speech-token generator for `text`. Uses the CUDA-graph decoder when the
    model carries one (model._mdt_t3_fast), else the eager loop. If the fast
    path raises (prefill, or mid-generation) it is disabled for the process,
    one line goes to stderr, and generation continues eagerly from the tokens
    already yielded -- the eager prefill is simply extended with them, so the
    stream never repeats or drops a token."""
    dec = getattr(model, "_mdt_t3_fast", None)
    if dec is None:
        yield from _t3_token_iter_eager(model, text, cancel)
        return
    done: list[int] = []
    try:
        for tok in _t3_token_iter_fast(model, dec, text, cancel):
            done.append(tok)
            yield tok
        return
    except Exception as exc:  # noqa: BLE001 - GeneratorExit is not an Exception
        model._mdt_t3_fast = None
        print(f"[desktop-tts] T3 fast path failed after {len(done)} tokens, "
              f"falling back to eager: {exc!r}", file=sys.stderr, flush=True)
    yield from _t3_token_iter_eager(model, text, cancel, prefix=done)


def _t3_token_iter_fast(model, dec, text: str, cancel: threading.Event | None = None):
    import torch
    import torch.nn.functional as F
    from transformers import (LogitsProcessorList, RepetitionPenaltyLogitsProcessor,
                              TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper)
    from chatterbox.tts_turbo import punc_norm

    t3 = model.t3
    procs = LogitsProcessorList()
    procs.append(TemperatureLogitsWarper(0.8))
    procs.append(TopKLogitsWarper(1000))
    procs.append(TopPLogitsWarper(0.95))
    procs.append(RepetitionPenaltyLogitsProcessor(1.2))
    text_tokens = model.tokenizer(punc_norm(text), return_tensors="pt", padding=True,
                                  truncation=True).input_ids.to(model.device)
    start = t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
    stop = t3.hp.stop_speech_token
    with torch.inference_mode():
        embeds, _ = t3.prepare_input_embeds(t3_cond=model.conds.t3, text_tokens=text_tokens,
                                            speech_tokens=start, cfg_weight=0.0)
        out = t3.tfmr(inputs_embeds=embeds, use_cache=True)
        logits = t3.speech_head(out[0][:, -1:])
        # First token: the same eager processors as before (once per sentence);
        # the graph takes over from the second token.
        nxt = torch.multinomial(F.softmax(procs(start, logits[:, -1, :]), dim=-1), num_samples=1)
        tok = int(nxt)
        dec.begin(out.past_key_values, tok)
        del out
        for _ in range(T3_MAX_GEN_LEN):
            if tok == stop:
                return
            yield tok
            if cancel is not None and cancel.is_set():
                return
            tok = dec.step()
        if tok != stop:
            yield tok


def _t3_token_iter_eager(model, text: str, cancel: threading.Event | None = None,
                         prefix: list[int] | None = None):
    """Generator twin of ChatterboxTurboTTS.generate()'s T3 half
    (t3.inference_turbo): identical sampling settings and logits processors,
    but yields each speech token as a python int as it is produced.
    `prefix` (fast-path fallback only) = tokens already yielded; they are fed
    back as speech context and not yielded again."""
    import torch
    import torch.nn.functional as F
    from transformers import (LogitsProcessorList, RepetitionPenaltyLogitsProcessor,
                              TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper)
    from chatterbox.tts_turbo import punc_norm

    t3 = model.t3
    temperature, top_k, top_p, rep = 0.8, 1000, 0.95, 1.2  # generate() defaults
    procs = LogitsProcessorList()
    procs.append(TemperatureLogitsWarper(temperature))
    procs.append(TopKLogitsWarper(top_k))
    procs.append(TopPLogitsWarper(top_p))
    procs.append(RepetitionPenaltyLogitsProcessor(rep))

    text_tokens = model.tokenizer(punc_norm(text), return_tensors="pt", padding=True,
                                  truncation=True).input_ids.to(model.device)
    start = t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
    with torch.inference_mode():
        speech = start
        if prefix:
            speech = torch.cat([start, torch.tensor([prefix], dtype=torch.long, device=start.device)], dim=1)
        embeds, _ = t3.prepare_input_embeds(t3_cond=model.conds.t3, text_tokens=text_tokens,
                                            speech_tokens=speech, cfg_weight=0.0)
        out = t3.tfmr(inputs_embeds=embeds, use_cache=True)
        past = out.past_key_values
        logits = t3.speech_head(out[0][:, -1:])
        # With a prefix, rep-penalty history is the tokens so far, as in the loop below.
        hist = torch.tensor([prefix], dtype=torch.long, device=start.device) if prefix else start
        nxt = torch.multinomial(F.softmax(procs(hist, logits[:, -1, :]), dim=-1), num_samples=1)
        generated = [nxt]
        if prefix:
            generated = [hist, nxt]
        stop = t3.hp.stop_speech_token
        for _ in range(T3_MAX_GEN_LEN):
            tok = int(nxt)
            if tok == stop:
                return
            yield tok
            if cancel is not None and cancel.is_set():
                return
            out = t3.tfmr(inputs_embeds=t3.speech_emb(nxt), past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = t3.speech_head(out[0])
            ids = torch.cat(generated, dim=1)
            processed = procs(ids, logits[:, -1, :])
            if torch.all(processed == -float("inf")):
                return
            nxt = torch.multinomial(F.softmax(processed, dim=-1), num_samples=1)
            generated.append(nxt)
        tok = int(nxt)
        if tok != stop:
            yield tok


def _decode_window(model, tokens: list[int], final: bool):
    """S3Gen + vocoder over a token window -> float32 numpy audio at model.sr.
    `final` appends the same 3 silence tokens generate() does."""
    import torch
    toks = list(tokens)
    if final:
        toks += [model_sil_token()] * 3
    t = torch.tensor(toks, dtype=torch.long, device=model.device)
    wav, _ = model.s3gen.inference(speech_tokens=t, ref_dict=model.conds.gen, n_cfm_timesteps=2)
    wav = wav.squeeze(0).detach().cpu().numpy()
    if STREAM_WATERMARK:
        wav = model.watermarker.apply_watermark(wav, sample_rate=model.sr)
    return wav


def model_sil_token() -> int:
    from chatterbox.tts_turbo import S3GEN_SIL
    return S3GEN_SIL


class ChunkBlender:
    """Joins overlapping decoded windows. Each pushed piece covers
    [its start, its end + overlap); the held-back tail of the previous piece
    covers the same span as the new piece's head, and the two are crossfaded
    (raised cosine) so a per-window difference in the vocoder's excitation
    phase never shows up as a click."""

    def __init__(self, overlap: int):
        import numpy as np
        self.ov = overlap
        self.tail = None
        self._np = np
        k = np.arange(overlap, dtype="float32")
        self.fade_in = (0.5 - 0.5 * np.cos(np.pi * (k + 0.5) / overlap)).astype("float32")
        self.fade_out = 1.0 - self.fade_in

    def push(self, piece, final: bool):
        np = self._np
        if self.tail is not None:
            n = min(len(self.tail), len(piece))
            head = self.tail[:n] * self.fade_out[:n] + piece[:n] * self.fade_in[:n]
            piece = np.concatenate([head, piece[n:]])
        if final or len(piece) <= self.ov:
            self.tail = None if final else piece
            return piece if final else piece[:0]
        self.tail = piece[-self.ov:]
        return piece[:-self.ov]


def _pcm16(wav) -> bytes:
    return (wav.clip(-1.0, 1.0) * 32767.0).astype("int16").tobytes()


def _make_pacer() -> WindowPacer:
    return WindowPacer(
        STREAM_FIRST_TOKENS, STREAM_CHUNK_TOKENS, mode=STREAM_PACING,
        ramp=parse_ramp(STREAM_RAMP_TOKENS), prebuffer_s=STREAM_ASSUMED_PREBUFFER_MS / 1000.0,
        margin_s=STREAM_MARGIN_MS / 1000.0, min_chunk=STREAM_MIN_CHUNK_TOKENS, behind_chunk=STREAM_BEHIND_CHUNK_TOKENS,
        t3_rate=STREAM_PRIOR_T3_TOKS_PER_S, decode_s=STREAM_PRIOR_DECODE_S)


def _stream_sentence_pcm(model, text: str, cancel: threading.Event | None = None):
    """Yields PCM16 byte chunks for one sentence as tokens become available."""
    spt = SAMPLES_PER_TOKEN
    ctx, la = STREAM_CONTEXT_TOKENS, STREAM_LOOKAHEAD_TOKENS
    blender = ChunkBlender(STREAM_OVERLAP_SAMPLES)
    pacer = _make_pacer()
    toks: list[int] = []
    emitted = 0
    n_seen = 0
    boundary = pacer.next_size()
    pacer.window_started()
    for tok in _t3_token_iter(model, text, cancel):
        if tok >= SPEECH_VOCAB:
            continue
        toks.append(tok)
        if len(toks) >= boundary + la:
            start = max(0, emitted - ctx)
            t_dec = time.monotonic()
            wav = _decode_window(model, toks[start:boundary + la], final=False)
            dec_s = time.monotonic() - t_dec
            lo = (emitted - start) * spt
            hi = (boundary - start) * spt + STREAM_OVERLAP_SAMPLES
            out = blender.push(wav[lo:hi], final=False)
            pacer.window_done(len(toks) - n_seen, dec_s, len(out) / model.sr)
            n_seen = len(toks)
            emitted = boundary
            boundary += pacer.next_size()
            if len(out):
                yield _pcm16(out)
    if len(toks) > emitted or emitted == 0:
        start = max(0, emitted - ctx)
        wav = _decode_window(model, toks[start:], final=True)
        out = blender.push(wav[(emitted - start) * spt:], final=True)
        if len(out):
            yield _pcm16(out)
    elif blender.tail is not None and len(blender.tail):
        yield _pcm16(blender.tail)


def stream_pcm_chunks(text: str, cancel: threading.Event | None = None):
    """Yields PCM16 chunks for `text`, sentence by sentence (same split rule
    as /tts). Holds the chatterbox lock one sentence at a time."""
    model = _chatterbox_model
    if model is None:
        raise RuntimeError(_chatterbox_load_error or "chatterbox not loaded")
    sentences = [text]
    if len(text) >= SENTENCE_CHUNK_MIN_CHARS:
        sentences = segment_full_text(text) or [text]
    for sentence in sentences:
        if cancel is not None and cancel.is_set():
            return
        with _chatterbox_lock:
            yield from _stream_sentence_pcm(model, sentence, cancel)


def _wav_stream_header(sr: int) -> bytes:
    """44-byte PCM16 mono WAV header with the 'unknown length' sentinel
    (0xFFFFFFFF) in both size fields, the standard streaming-WAV form."""
    import struct
    return (b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16) +
            b"data" + struct.pack("<I", 0xFFFFFFFF))


class Handler(BaseHTTPRequestHandler):
    server_version = "LominDesktopTTS/2.0"

    def log_message(self, *_args):
        pass

    def _send_json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if self.path == "/health":
            ok = _piper_voice is not None or _chatterbox_model is not None
            self._send_json(200 if ok else 503, {
                "status": "ok" if ok else "error",
                "chatterbox_loaded": _chatterbox_model is not None,
                "chatterbox_error": _chatterbox_load_error,
                "piper_voice": VOICE_MODEL,
                "piper_loaded": _piper_voice is not None,
                "piper_error": _piper_load_error,
            })
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/tts", "/tts/stream"):
            self._send_json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else b""
        try:
            req = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            req = {}
        text = (req.get("text") or "").strip()
        if not text:
            self._send_json(400, {"error": "text is required"})
            return
        if self.path == "/tts/stream" and self._stream(text, req.get("format", "wav")):
            return
        try:
            wav_bytes, engine = synthesize_wav_bytes_chunked(text)
        except Exception as exc:
            self._send_json(502, {"error": f"synthesis failed: {exc}"})
            return
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav_bytes)))
        self.send_header("X-TTS-Engine", engine)
        self.end_headers()
        try:
            self.wfile.write(wav_bytes)
        except (BrokenPipeError, ConnectionResetError):
            pass


    def _stream(self, text: str, fmt: str) -> bool:
        """POST /tts/stream. Returns True when the response was handled
        (streamed, or failed after headers went out); False means "nothing
        streamable" and the caller serves the ordinary /tts response, so a
        client that accepts audio/wav works whether or not Chatterbox is up.

        Wire format: HTTP/1.0-style, no Content-Length, EOF ends the body
        (same convention as gateway.py's /ask). Default body is a WAV whose
        header carries the 0xFFFFFFFF unknown-length sentinel followed by
        PCM16 mono; {"format": "pcm"} sends bare s16le (rate in the
        X-Sample-Rate header). Headers are sent with the FIRST audio chunk, so
        a failure before any audio still falls back to a normal /tts reply."""
        model = _chatterbox_model
        if not STREAM_ENABLED or model is None:
            return False
        q: queue.Queue = queue.Queue(maxsize=8)
        cancel = threading.Event()

        def produce():
            try:
                for chunk in stream_pcm_chunks(text, cancel):
                    q.put(chunk)
                    if cancel.is_set():
                        return
                q.put(None)
            except Exception as exc:
                q.put(exc)

        threading.Thread(target=produce, daemon=True).start()
        item = q.get()
        if item is None or isinstance(item, Exception):
            if isinstance(item, Exception):
                print(f"[desktop-tts] stream failed before first chunk, falling back: {item}",
                      file=sys.stderr, flush=True)
            return False
        raw = fmt == "pcm"
        self.send_response(200)
        self.send_header("Content-Type", "audio/L16" if raw else "audio/wav")
        self.send_header("X-TTS-Engine", "chatterbox-stream")
        self.send_header("X-Sample-Rate", str(model.sr))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            if not raw:
                self.wfile.write(_wav_stream_header(model.sr))
            while item is not None:
                if isinstance(item, Exception):
                    print(f"[desktop-tts] stream aborted mid-body: {item}", file=sys.stderr, flush=True)
                    break
                self.wfile.write(item)
                self.wfile.flush()
                item = q.get()
        except (BrokenPipeError, ConnectionResetError):
            cancel.set()
            while not q.empty():  # unblock a producer stuck on a full queue
                q.get_nowait()
        return True


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    # Piper loads in under a second and gates nothing external, so it loads
    # synchronously before the server binds. Chatterbox's from_pretrained()
    # calls snapshot_download(), which contacts huggingface.co on every
    # start (even with the weights cached, it does a metadata check) and
    # took ~5 minutes on a cold cache — binding 9097 only after that would
    # leave the desktop hop dead on every restart while a generic-Piper
    # fallback sat unused a few lines above. Load it in a daemon thread
    # instead: Piper serves every request until synthesize_wav_bytes()
    # sees _chatterbox_model flip from None to loaded.
    _load_piper()
    server = Server((BIND, PORT), Handler)
    threading.Thread(target=_load_chatterbox, daemon=True).start()
    print(f"[desktop-tts] listening on {BIND}:{PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
