"""side_channel.py — Side-Channel Artifact Webhook, Transcript Sink & Auto-Mic Reopen.

Provides:
- reply_expects_reply(text: str) -> bool:
    Detects if a reply ends in a question (indicating the microphone should reopen).
    Trims trailing whitespace, trailing quotes, and trailing Unicode emoji/joiners.
- forward_artifacts_to_webhook(raw_reply: str, spoken_reply: str, webhook_url: str = '', http_poster=None) -> tuple[str, bool]:
    Detects URLs or triple-backtick code blocks in raw_reply and posts them to
    MVG_ARTIFACT_WEBHOOK_URL or webhook_url. Appends 'I sent the link to Discord.'
    to the spoken reply if not already cued.
- emit_turn_transcript(turn_record: dict, sink_url: str = '', log_path: str = '', http_poster=None) -> bool:
    Emits completed turn records to MVG_TRANSCRIPT_SINK_URL via HTTP and/or
    appends to MVG_TRANSCRIPT_LOG_PATH in JSONL format.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
import urllib.error
import urllib.request
from typing import Any, Callable

# Regex to detect URLs and code blocks in replies
_URL_RE = re.compile(r"https?://[^\s)>\]\"'`]+", re.IGNORECASE)
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")

# Spoken cues indicating the user has already been verbally notified
_SPOKEN_CHAT_CUES = (
    "discord",
    "sent the link",
    "sent it to",
    "sent that over",
    "check discord",
    "posted the link",
    "dropped it in",
)


def _is_trimmable(ch: str) -> bool:
    """Returns True if the character is whitespace, quote, or trailing emoji/joiner."""
    if ch.isspace():
        return True
    if ch in ('"', "'", "“", "”", "‘", "’", "`", "«", "»", "‹", "›"):
        return True
    if ch in ("\u200d", "\u200c", "\ufe0e", "\ufe0f", "\u200b", "\ufeff"):
        return True
    cat = unicodedata.category(ch)
    if cat in ("Pi", "Pf"):  # punctuation initial/final quote
        return True
    if cat in ("So", "Sk", "Cf"):  # symbols, modifiers, format
        return ch not in ("?", "❓", "❔")
    if cat == "Mn" and ord(ch) in (0xFE0E, 0xFE0F, 0x20E3):
        return True
    code = ord(ch)
    if 0x1F300 <= code <= 0x1FAFF or 0x2600 <= code <= 0x27BF:
        return ch not in ("?", "❓", "❔")
    return False


def reply_expects_reply(text: str) -> bool:
    """Determine whether a reply ends with a question, expecting a follow-up answer.

    Trims trailing whitespace, trailing quotes, and trailing Unicode emoji/joiners.
    Returns True if the resulting string ends with '?' (or Unicode question ornament).
    """
    if not text or not isinstance(text, str):
        return False
    s = text.rstrip()
    while s and _is_trimmable(s[-1]):
        s = s[:-1].rstrip()
    return s.endswith("?") or s.endswith(("❓", "❔"))


def _default_http_post(url: str, payload: dict, timeout_s: float = 5.0) -> bool:
    """Default HTTP POST using urllib.request."""
    try:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Lomin-Gateway/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def _invoke_poster(
    poster: Callable[..., Any],
    url: str,
    payload: dict,
    timeout_s: float = 5.0,
) -> bool:
    """Invoke a custom http_poster with varying parameter signatures."""
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Lomin-Gateway/1.0",
    }
    try:
        try:
            res = poster(url, payload)
        except TypeError:
            try:
                res = poster(url, headers=headers, payload=payload, timeout_s=timeout_s)
            except TypeError:
                try:
                    res = poster(url, headers, payload, timeout_s)
                except TypeError:
                    res = poster(url, headers=headers, payload=payload)
    except Exception:
        return False

    if res is False:
        return False
    return True


def forward_artifacts_to_webhook(
    raw_reply: str,
    spoken_reply: str,
    webhook_url: str = "",
    http_poster: Callable[..., Any] | None = None,
) -> tuple[str, bool]:
    """Inspect raw_reply for URLs or code blocks, and forward to webhook when configured.

    If artifacts are found and webhook_url (or MVG_ARTIFACT_WEBHOOK_URL) is configured:
    - Posts the raw reply as JSON payload to the webhook.
    - Appends 'I sent the link to Discord.' to spoken_reply if not already cued.
    - Returns (updated_spoken_reply, True).

    Otherwise returns (spoken_reply, False).
    """
    raw_text = (raw_reply or "").strip()
    spoken_text = (spoken_reply or "").strip()

    has_url = bool(_URL_RE.search(raw_text))
    has_code = bool(_CODE_BLOCK_RE.search(raw_text))
    if not (has_url or has_code):
        return spoken_reply, False

    target_url = (webhook_url or os.environ.get("MVG_ARTIFACT_WEBHOOK_URL", "")).strip()
    if not target_url:
        return spoken_reply, False

    payload = {
        "content": raw_text,
        "text": raw_text,
        "raw_reply": raw_text,
    }

    if http_poster is not None:
        posted = _invoke_poster(http_poster, target_url, payload)
    else:
        posted = _default_http_post(target_url, payload)

    if not posted:
        return spoken_reply, False

    # Check if spoken_reply already contains a verbal cue
    lower_spoken = spoken_text.lower()
    is_cued = any(cue in lower_spoken for cue in _SPOKEN_CHAT_CUES)

    if not is_cued:
        cue = "I sent the link to Discord."
        if spoken_text:
            if not spoken_text.endswith((".", "!", "?")):
                spoken_text += "."
            updated_spoken = f"{spoken_text} {cue}"
        else:
            updated_spoken = cue
    else:
        updated_spoken = spoken_reply

    return updated_spoken, True


def emit_turn_transcript(
    turn_record: dict,
    sink_url: str = "",
    log_path: str = "",
    http_poster: Callable[..., Any] | None = None,
) -> bool:
    """Emit turn transcript to HTTP sink and/or JSONL log file.

    Configured via MVG_TRANSCRIPT_SINK_URL and MVG_TRANSCRIPT_LOG_PATH.
    Returns True if at least one configured destination succeeded.
    Returns False if neither is configured or all attempts failed.
    """
    if not isinstance(turn_record, dict) or not turn_record:
        return False

    eff_sink_url = (sink_url or os.environ.get("MVG_TRANSCRIPT_SINK_URL", "")).strip()
    eff_log_path = (log_path or os.environ.get("MVG_TRANSCRIPT_LOG_PATH", "")).strip()

    if not eff_sink_url and not eff_log_path:
        return False

    success = False

    if eff_log_path:
        try:
            parent = os.path.dirname(os.path.abspath(eff_log_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            line = json.dumps(turn_record, ensure_ascii=False, default=str) + "\n"
            with open(eff_log_path, "a", encoding="utf-8") as f:
                f.write(line)
            success = True
        except Exception:
            pass

    if eff_sink_url:
        try:
            if http_poster is not None:
                posted = _invoke_poster(http_poster, eff_sink_url, turn_record)
            else:
                posted = _default_http_post(eff_sink_url, turn_record)
            if posted:
                success = True
        except Exception:
            pass

    return success
