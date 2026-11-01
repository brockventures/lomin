"""WAV helpers and audio normalization (moved out of gateway.py, ERED LOMIN P1b).

Stdlib only: array, wave, sys. normalize_wav() never logs; the safe wrapper
that does (_safe_normalize_wav) stays in gateway.py."""

import array
import sys
import wave
from io import BytesIO


# --- WAV helpers -----------------------------------------------------------

def pcm_to_wav(pcm_bytes: bytes, sample_rate: int, channels: int = 1, sample_width: int = 2) -> bytes:
    """Wrap raw signed-16-bit little-endian PCM in a WAV container."""
    if len(pcm_bytes) % (channels * sample_width):
        pcm_bytes = pcm_bytes[: len(pcm_bytes) - (len(pcm_bytes) % (channels * sample_width))]
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sample_width)
        w.setframerate(sample_rate)
        w.writeframes(pcm_bytes)
    return buf.getvalue()


# --- Audio normalization (ONE-FORMAT) ---------------------------------------
# The ear wants ONE audio format per turn: the engine can change
# sentence to sentence within a single streamed reply (the desktop Chatterbox
# clone at 24 kHz, Piper at 22.05 kHz or another rate),
# so a caller that concatenates or plays sentences back to back can otherwise
# see the sample rate/channels/width change mid-turn. normalize_wav()
# converts any WAV to a single fixed wire format: 24000 Hz, mono, 16-bit PCM.
#
# Pure stdlib — `audioop` is removed in Python 3.13 (which the gateway host runs), and
# this module stays stdlib-only apart from the already-guarded `requests`
# import for /ambient's Jev path, so numpy is not used here even though it's
# already a dependency of desktop_tts.py/desktop_embed.py on the desktop
# side. Linear-interpolation resampling and integer-averaging downmix are
# good enough for spoken-word TTS audio; this is not meant for music.

AUDIO_TARGET_RATE = 24000
AUDIO_TARGET_CHANNELS = 1
AUDIO_TARGET_SAMPWIDTH = 2


def _pcm_to_mono_int16(raw: bytes, channels: int, sampwidth: int) -> array.array:
    """Returns a mono `array.array("h")` of 16-bit signed samples decoded
    from `raw` PCM bytes at `channels`/`sampwidth`. Supports the sample
    widths WAV actually uses in this fleet: 1 (unsigned), 2 (signed, the
    common case), 3 (24-bit signed, packed), 4 (32-bit signed)."""
    if sampwidth == 1:
        # 8-bit WAV PCM is unsigned 0..255, unlike every wider width.
        unsigned = array.array("B", raw)
        samples = array.array("h", [(b - 128) * 256 for b in unsigned])
    elif sampwidth == 2:
        samples = array.array("h", raw)
        if sys.byteorder == "big":
            samples.byteswap()
    elif sampwidth == 3:
        vals = []
        for i in range(0, len(raw) - 2, 3):
            v = raw[i] | (raw[i + 1] << 8) | (raw[i + 2] << 16)
            if v & 0x800000:
                v -= 0x1000000
            vals.append(max(-32768, min(32767, v >> 8)))
        samples = array.array("h", vals)
    elif sampwidth == 4:
        wide = array.array("i", raw)
        if sys.byteorder == "big":
            wide.byteswap()
        samples = array.array("h", [max(-32768, min(32767, v >> 16)) for v in wide])
    else:
        raise ValueError(f"unsupported WAV sample width: {sampwidth} bytes")

    if channels <= 1:
        return samples
    n_frames = len(samples) // channels
    mono = array.array("h", bytes(2 * n_frames))
    for i in range(n_frames):
        frame = samples[i * channels:(i + 1) * channels]
        mono[i] = sum(frame) // channels
    return mono


def _resample_linear(samples: array.array, src_rate: int, dst_rate: int) -> array.array:
    """Linear-interpolation resample of mono int16 `samples` from `src_rate`
    to `dst_rate`. Good enough for spoken sentences; not a high-quality
    resampler."""
    src_len = len(samples)
    if src_rate == dst_rate or src_len == 0:
        return samples
    dst_len = max(1, round(src_len * dst_rate / src_rate))
    out = array.array("h", bytes(2 * dst_len))
    ratio = (src_len - 1) / (dst_len - 1) if dst_len > 1 else 0.0
    for i in range(dst_len):
        pos = i * ratio
        i0 = int(pos)
        i1 = min(i0 + 1, src_len - 1)
        frac = pos - i0
        out[i] = int(round(samples[i0] * (1 - frac) + samples[i1] * frac))
    return out


def normalize_wav(
    wav_bytes: bytes,
    target_rate: int = AUDIO_TARGET_RATE,
    target_channels: int = AUDIO_TARGET_CHANNELS,
    target_sampwidth: int = AUDIO_TARGET_SAMPWIDTH,
) -> bytes:
    """Converts any WAV to `target_rate` Hz / `target_channels` / a
    `target_sampwidth`-byte-per-sample PCM WAV. Pass-through fast path:
    returns `wav_bytes` unchanged when the input already matches. Raises
    (ValueError, wave.Error, EOFError, ...) on malformed input or an
    unsupported source sample width — callers that must never drop a
    sentence over a normalization failure should catch and fall back to the
    original bytes (see gateway._safe_normalize_wav)."""
    if target_channels != 1 or target_sampwidth != 2:
        raise ValueError("normalize_wav only supports mono 16-bit PCM targets")

    with wave.open(BytesIO(wav_bytes), "rb") as w:
        channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        framerate = w.getframerate()
        nframes = w.getnframes()
        raw = w.readframes(nframes)

    if (channels, sampwidth, framerate) == (target_channels, target_sampwidth, target_rate):
        return wav_bytes

    mono = _pcm_to_mono_int16(raw, channels, sampwidth)
    resampled = _resample_linear(mono, framerate, target_rate)
    if sys.byteorder == "big":
        resampled = array.array("h", resampled)
        resampled.byteswap()
    pcm_bytes = resampled.tobytes()

    buf = BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(target_channels)
        out.setsampwidth(target_sampwidth)
        out.setframerate(target_rate)
        out.writeframes(pcm_bytes)
    return buf.getvalue()
