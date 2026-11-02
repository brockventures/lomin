"""side_channel.py — Side-Channel Artifact Webhook, Transcript Sink & Auto-Mic Reopen.

Provides:
- reply_expects_reply(text: str) -> bool:
    Detects if a reply ends in a question (indicating the microphone should reopen).
    Trims trailing whitespace, trailing quotes, and trailing Unicode emoji/joiners.
- forward_artifacts_to_webhook(raw_reply: str, spoken_reply: str, webhook_url: str = '',
                               http_poster=None, sync: bool = False) -> tuple[str, bool]:
    Detects URLs or triple-backtick code blocks in raw_reply and posts them to
    MVG_ARTIFACT_WEBHOOK_URL or webhook_url. Appends 'I sent the link to Discord.'
    to the spoken reply if not already cued.
    Runs asynchronously via a bounded background daemon worker queue by default (sync=False),
    or synchronously when sync=True.
- emit_turn_transcript(turn_record: dict, sink_url: str = '', log_path: str = '',
                       http_poster=None, auth_token: str = '', max_bytes: int | None = None,
                       sync: bool = False) -> bool:
    Emits completed turn records to MVG_TRANSCRIPT_SINK_URL via HTTP and/or
    appends to MVG_TRANSCRIPT_LOG_PATH in JSONL format with 0o600 permissions
    and automatic rotation when size exceeds max_bytes (MVG_TRANSCRIPT_MAX_BYTES).
    Runs asynchronously via a bounded background daemon worker queue by default (sync=False),
    or synchronously when sync=True.
- flush_side_channel_queue(timeout_s: float = 5.0) -> bool:
    Flushes all pending tasks in the background worker queue.
"""

from __future__ import annotations

import inspect
import json
import os
import queue
import re
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
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


def _validate_url(url: str) -> bool:
    """Validate that url uses http or https scheme and has a valid netloc."""
    if not url or not isinstance(url, str):
        return False
    try:
        parsed = urllib.parse.urlsplit(url.strip())
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Enforces http/https scheme restrictions and prevents cross-scheme downgrade on redirects."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new_parsed = urllib.parse.urlsplit(newurl)
        if new_parsed.scheme not in ("http", "https"):
            return None
        old_parsed = urllib.parse.urlsplit(req.full_url)
        if old_parsed.scheme == "https" and new_parsed.scheme != "https":
            return None
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_SAFE_OPENER = urllib.request.build_opener(SafeRedirectHandler())


def _default_http_post(
    url: str,
    payload: dict,
    headers: dict[str, str] | None = None,
    auth_token: str = "",
    timeout_s: float = 5.0,
) -> bool:
    """Default HTTP POST using urllib.request with scheme validation and safe redirect handler."""
    if not _validate_url(url):
        return False

    req_headers = {
        "Content-Type": "application/json",
        "User-Agent": "Lomin-Gateway/1.0",
    }
    if headers:
        req_headers.update(headers)
    if auth_token:
        req_headers["Authorization"] = f"Bearer {auth_token}"

    try:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers=req_headers,
            method="POST",
        )
        with _SAFE_OPENER.open(req, timeout=timeout_s) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def _invoke_poster(
    poster: Callable[..., Any],
    url: str,
    payload: dict,
    headers: dict[str, str] | None = None,
    timeout_s: float = 5.0,
) -> bool:
    """Invoke a custom http_poster using inspect.signature without except TypeError retry loops."""
    if headers is None:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Lomin-Gateway/1.0",
        }
    try:
        sig = inspect.signature(poster)
        params = list(sig.parameters.values())
        param_names = [p.name for p in params]
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
        has_varpos = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)

        kwargs: dict[str, Any] = {}
        if has_varkw or "headers" in param_names:
            kwargs["headers"] = headers
        if has_varkw or "timeout_s" in param_names:
            kwargs["timeout_s"] = timeout_s
        elif "timeout" in param_names:
            kwargs["timeout"] = timeout_s

        args: list[Any] = [url]
        if has_varpos or len(params) >= 2:
            args.append(payload)
        elif "payload" in param_names:
            kwargs["payload"] = payload

        res = poster(*args, **kwargs)
        if res is False:
            return False
        return True
    except Exception:
        return False


def _append_transcript_log(
    log_path: str,
    turn_record: dict,
    max_bytes: int | None = None,
) -> bool:
    """Append a JSONL record to log_path with 0o600 permissions and size cap rotation."""
    try:
        parent = os.path.dirname(os.path.abspath(log_path))
        if parent:
            os.makedirs(parent, exist_ok=True)

        eff_max_bytes = max_bytes if max_bytes is not None else int(
            os.environ.get("MVG_TRANSCRIPT_MAX_BYTES", str(10 * 1024 * 1024))
        )

        line = json.dumps(turn_record, ensure_ascii=False, default=str) + "\n"
        encoded = line.encode("utf-8")

        # Rotate if current size + new line exceeds max_bytes
        if os.path.exists(log_path):
            try:
                cur_size = os.path.getsize(log_path)
                if cur_size + len(encoded) > eff_max_bytes:
                    backup_path = f"{log_path}.1"
                    if os.path.exists(backup_path):
                        try:
                            os.remove(backup_path)
                        except Exception:
                            pass
                    os.replace(log_path, backup_path)
            except Exception:
                pass

        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        fd = os.open(log_path, flags, 0o600)
        try:
            os.chmod(log_path, 0o600)
        except Exception:
            pass
        with open(fd, "a", encoding="utf-8") as f:
            f.write(line)
        return True
    except Exception:
        return False


# Bounded background worker queue
_QUEUE_MAXSIZE = int(os.environ.get("MVG_SIDE_CHANNEL_QUEUE_SIZE", "1000"))
_side_channel_queue: queue.Queue = queue.Queue(maxsize=_QUEUE_MAXSIZE)
_worker_started = False
_worker_lock = threading.Lock()


def _worker_loop() -> None:
    while True:
        task = _side_channel_queue.get()
        if task is None:
            _side_channel_queue.task_done()
            break
        func, args, kwargs = task
        try:
            func(*args, **kwargs)
        except Exception:
            pass
        finally:
            _side_channel_queue.task_done()


def _ensure_worker_started() -> None:
    global _worker_started
    with _worker_lock:
        if not _worker_started:
            t = threading.Thread(target=_worker_loop, daemon=True, name="SideChannelWorker")
            t.start()
            _worker_started = True


def _enqueue_task(func: Callable[..., Any], *args: Any, **kwargs: Any) -> bool:
    _ensure_worker_started()
    try:
        _side_channel_queue.put_nowait((func, args, kwargs))
        return True
    except queue.Full:
        return False


def flush_side_channel_queue(timeout_s: float = 5.0) -> bool:
    """Flush pending side-channel tasks within timeout_s."""
    t_end = time.time() + timeout_s
    with _side_channel_queue.all_tasks_done:
        while _side_channel_queue.unfinished_tasks:
            remaining = t_end - time.time()
            if remaining <= 0:
                return False
            _side_channel_queue.all_tasks_done.wait(remaining)
    return True


flush_queue = flush_side_channel_queue  # Convenience alias


def reset_side_channel_queue() -> None:
    """Drain any pending tasks in queue (for test cleanup)."""
    while not _side_channel_queue.empty():
        try:
            _side_channel_queue.get_nowait()
            _side_channel_queue.task_done()
        except Exception:
            break


def _dispatch_artifact_webhook(
    url: str,
    payload: dict,
    http_poster: Callable[..., Any] | None = None,
) -> bool:
    if http_poster is not None:
        return _invoke_poster(http_poster, url, payload)
    return _default_http_post(url, payload)


def forward_artifacts_to_webhook(
    raw_reply: str,
    spoken_reply: str,
    webhook_url: str = "",
    http_poster: Callable[..., Any] | None = None,
    sync: bool = False,
) -> tuple[str, bool]:
    """Inspect raw_reply for URLs or code blocks, and forward to webhook when configured.

    If artifacts are found and webhook_url (or MVG_ARTIFACT_WEBHOOK_URL) is configured:
    - Appends 'I sent the link to Discord.' to spoken_reply if not already cued.
    - If sync=True, posts synchronously and returns (updated_spoken, True) on success
      or (spoken_reply, False) on failure.
    - If sync=False (default), enqueues the POST to the background queue and returns
      (updated_spoken, True).

    Otherwise returns (spoken_reply, False).
    """
    raw_text = (raw_reply or "").strip()
    spoken_text = (spoken_reply or "").strip()

    has_url = bool(_URL_RE.search(raw_text))
    has_code = bool(_CODE_BLOCK_RE.search(raw_text))
    if not (has_url or has_code):
        return spoken_reply, False

    target_url = (webhook_url or os.environ.get("MVG_ARTIFACT_WEBHOOK_URL", "")).strip()
    if not target_url or not _validate_url(target_url):
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

    payload = {
        "content": raw_text,
        "text": raw_text,
        "raw_reply": raw_text,
    }

    if sync:
        posted = _dispatch_artifact_webhook(target_url, payload, http_poster)
        if not posted:
            return spoken_reply, False
        return updated_spoken, True

    enqueued = _enqueue_task(_dispatch_artifact_webhook, target_url, payload, http_poster)
    if not enqueued:
        return spoken_reply, False
    return updated_spoken, True


def forward_artifacts_to_webhook_sync(
    raw_reply: str,
    spoken_reply: str,
    webhook_url: str = "",
    http_poster: Callable[..., Any] | None = None,
) -> tuple[str, bool]:
    """Synchronous helper for forward_artifacts_to_webhook."""
    return forward_artifacts_to_webhook(
        raw_reply, spoken_reply, webhook_url=webhook_url, http_poster=http_poster, sync=True
    )


def _dispatch_transcript_sync(
    turn_record: dict,
    sink_url: str,
    log_path: str,
    http_poster: Callable[..., Any] | None,
    auth_token: str,
    max_bytes: int | None,
) -> bool:
    success = False

    if log_path:
        if _append_transcript_log(log_path, turn_record, max_bytes=max_bytes):
            success = True

    if sink_url:
        try:
            if http_poster is not None:
                posted = _invoke_poster(http_poster, sink_url, turn_record)
            else:
                posted = _default_http_post(sink_url, turn_record, auth_token=auth_token)
            if posted:
                success = True
        except Exception:
            pass

    return success


def emit_turn_transcript(
    turn_record: dict,
    sink_url: str = "",
    log_path: str = "",
    http_poster: Callable[..., Any] | None = None,
    auth_token: str = "",
    max_bytes: int | None = None,
    sync: bool = False,
) -> bool:
    """Emit turn transcript to HTTP sink and/or JSONL log file.

    Configured via MVG_TRANSCRIPT_SINK_URL and MVG_TRANSCRIPT_LOG_PATH.
    If sync=False (default), enqueues to the daemon worker queue and returns True
    if destinations are valid.
    If sync=True, executes immediately and returns whether at least one destination succeeded.
    """
    if not isinstance(turn_record, dict) or not turn_record:
        return False

    eff_sink_url = (sink_url or os.environ.get("MVG_TRANSCRIPT_SINK_URL", "")).strip()
    eff_log_path = (log_path or os.environ.get("MVG_TRANSCRIPT_LOG_PATH", "")).strip()
    eff_token = (auth_token or os.environ.get("MVG_TRANSCRIPT_AUTH_TOKEN", "")).strip()

    if eff_sink_url and not _validate_url(eff_sink_url):
        eff_sink_url = ""

    if not eff_sink_url and not eff_log_path:
        return False

    if sync:
        return _dispatch_transcript_sync(
            turn_record, eff_sink_url, eff_log_path, http_poster, eff_token, max_bytes
        )

    return _enqueue_task(
        _dispatch_transcript_sync,
        turn_record,
        eff_sink_url,
        eff_log_path,
        http_poster,
        eff_token,
        max_bytes,
    )


def emit_turn_transcript_sync(
    turn_record: dict,
    sink_url: str = "",
    log_path: str = "",
    http_poster: Callable[..., Any] | None = None,
    auth_token: str = "",
    max_bytes: int | None = None,
) -> bool:
    """Synchronous helper for emit_turn_transcript."""
    return emit_turn_transcript(
        turn_record,
        sink_url=sink_url,
        log_path=log_path,
        http_poster=http_poster,
        auth_token=auth_token,
        max_bytes=max_bytes,
        sync=True,
    )
