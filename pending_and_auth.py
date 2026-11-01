"""pending_and_auth.py — Deferred Async Answers & Per-Device Bearer Auth.

Provides:
- BearerTokenVerifier:
    Verifies HTTP 'Authorization: Bearer <token>' headers using constant-time
    comparison (hmac.compare_digest). Supports single shared token or
    per-device mapping (node1:tok1,node2:tok2[,shared_fallback]) from MVG_DEVICE_TOKENS.
    Disabled when MVG_DEVICE_TOKENS is empty or unset.
- has_defer_marker(text: str) -> bool:
    Detects if '[[defer]]' marker is present in text.
- strip_defer_marker(text: str) -> str:
    Strips '[[defer]]' marker(s) and normalizes surrounding whitespace.
- PendingVoiceAnswerStore:
    Thread-safe in-memory store for deferred asynchronous voice replies.
    Supports store(node_id, prompt, answer, topic=''), pop_for_node(node_id),
    and peek_for_node(node_id).
"""

from __future__ import annotations

import collections
import hmac
import os
import re
import threading
import time
import uuid
from typing import Any

# Regex to match [[defer]] case-insensitively with optional internal whitespace
_DEFER_RE = re.compile(r"\[\[\s*defer\s*\]\]", re.IGNORECASE)


def has_defer_marker(text: str | None) -> bool:
    """Return True if text contains the [[defer]] marker."""
    if not text or not isinstance(text, str):
        return False
    return bool(_DEFER_RE.search(text))


def strip_defer_marker(text: str | None) -> str:
    """Strip [[defer]] marker(s) from text and normalize whitespace."""
    if not text or not isinstance(text, str):
        return ""
    cleaned = _DEFER_RE.sub("", text)
    # Normalize multiple whitespace characters per line, preserve newlines
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in cleaned.splitlines()]
    return "\n".join(lines).strip()


class BearerTokenVerifier:
    """Verifies Bearer tokens using constant-time hmac.compare_digest.

    Configuration can be passed directly as a tokens_str or read dynamically from
    MVG_DEVICE_TOKENS environment variable.
    Format: 'node1:tok1,node2:tok2' or single shared token 'shared_secret',
    or mixed 'node1:tok1,fallback_secret'.
    """

    def __init__(self, tokens_str: str | None = None) -> None:
        self._tokens_str = tokens_str

    def _get_raw(self) -> str:
        if self._tokens_str is not None:
            return self._tokens_str.strip()
        return os.environ.get("MVG_DEVICE_TOKENS", "").strip()

    @property
    def is_enabled(self) -> bool:
        """Returns True if tokens are configured, False if empty/disabled."""
        return bool(self._get_raw())

    def _parse_tokens(self) -> tuple[dict[str, str], str | None]:
        raw = self._get_raw()
        if not raw:
            return {}, None
        per_node: dict[str, str] = {}
        shared_fallback: str | None = None
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                node, tok = part.split(":", 1)
                node_clean = node.strip()
                tok_clean = tok.strip()
                if node_clean and tok_clean:
                    per_node[node_clean] = tok_clean
            else:
                shared_fallback = part.strip()
        return per_node, shared_fallback

    def get_expected_token(self, node_id: str = "") -> str | None:
        """Resolve expected token for the given node_id, or shared fallback."""
        per_node, shared_fallback = self._parse_tokens()
        if node_id and node_id in per_node:
            return per_node[node_id]
        return shared_fallback

    @staticmethod
    def _extract_token(auth_header: str | None) -> str | None:
        if not auth_header or not isinstance(auth_header, str):
            return None
        h = auth_header.strip()
        if not h:
            return None
        if h.lower().startswith("bearer "):
            tok = h[7:].strip()
            return tok if tok else None
        return h

    def verify(self, auth_header: str | None, node_id: str = "") -> bool:
        """Verify an Authorization header in constant time.

        If disabled (no tokens configured), always returns True.
        If enabled, extracts Bearer token and verifies using hmac.compare_digest.
        """
        if not self.is_enabled:
            return True

        actual_token = self._extract_token(auth_header)
        expected_token = self.get_expected_token(node_id)

        # Constant-time comparison to prevent timing leak
        if actual_token is None or expected_token is None:
            dummy_a = actual_token.encode("utf-8") if actual_token else b"dummy_pad_token_value_a"
            dummy_b = b"dummy_pad_token_value_b"
            hmac.compare_digest(dummy_a, dummy_b)
            return False

        return hmac.compare_digest(actual_token.encode("utf-8"), expected_token.encode("utf-8"))

    def verify_token(self, token: str | None, node_id: str = "") -> bool:
        """Verify a raw token string against expected token for node_id."""
        return self.verify(f"Bearer {token}" if token else None, node_id)


class PendingVoiceAnswerStore:
    """Thread-safe store for deferred voice answers.

    Holds pending responses per node_id until popped or peeked by client.
    """

    def __init__(self, max_per_node: int = 50) -> None:
        self._lock = threading.Lock()
        self._store: dict[str, collections.deque[dict[str, Any]]] = {}
        self._max_per_node = max(1, max_per_node)

    def store(self, node_id: str, prompt: str, answer: str, topic: str = "") -> dict[str, Any]:
        """Store a pending answer for node_id and return the stored record."""
        clean_node = (node_id or "").strip()
        record: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "node_id": clean_node,
            "prompt": str(prompt or ""),
            "answer": str(answer or ""),
            "topic": str(topic or ""),
            "created_at": time.time(),
        }
        with self._lock:
            q = self._store.setdefault(clean_node, collections.deque(maxlen=self._max_per_node))
            q.append(record)
        return dict(record)

    def pop_for_node(self, node_id: str) -> dict[str, Any] | None:
        """Pop and return the oldest pending answer for node_id, or None if empty."""
        clean_node = (node_id or "").strip()
        with self._lock:
            q = self._store.get(clean_node)
            if not q:
                return None
            item = q.popleft()
            if not q:
                self._store.pop(clean_node, None)
            return dict(item)

    def peek_for_node(self, node_id: str) -> dict[str, Any] | None:
        """Return the oldest pending answer for node_id without removing it, or None."""
        clean_node = (node_id or "").strip()
        with self._lock:
            q = self._store.get(clean_node)
            if not q:
                return None
            return dict(q[0])

    def count_for_node(self, node_id: str) -> int:
        """Return the count of pending answers for node_id."""
        clean_node = (node_id or "").strip()
        with self._lock:
            return len(self._store.get(clean_node, ()))

    def clear(self, node_id: str = "") -> None:
        """Clear store for node_id or entire store if node_id is empty."""
        clean_node = (node_id or "").strip()
        with self._lock:
            if clean_node:
                self._store.pop(clean_node, None)
            else:
                self._store.clear()


# Default process-wide singletons
_DEFAULT_STORE = PendingVoiceAnswerStore()
_DEFAULT_VERIFIER = BearerTokenVerifier()


def get_pending_store() -> PendingVoiceAnswerStore:
    """Return the global PendingVoiceAnswerStore singleton."""
    return _DEFAULT_STORE


def get_device_token_verifier() -> BearerTokenVerifier:
    """Return the global BearerTokenVerifier singleton."""
    return _DEFAULT_VERIFIER
