"""Text helpers for spoken replies (moved out of gateway.py, ERED LOMIN P1b).

Stdlib only. gateway.py re-imports every name."""

import re


# --- Markdown stripping (for spoken replies) -----------------------------

_MD_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_MD_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_EMPHASIS_RE = re.compile(r"(\*\*\*|\*\*|\*|___|__|_)")
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_MD_BULLET_RE = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)


def strip_markdown(text: str) -> str:
    """Best-effort strip of markdown so a spoken reply doesn't say
    "asterisk asterisk" or read a code block aloud. Not a full parser —
    good enough for the short spoken sentences /ask is asked to produce."""
    if not text:
        return text
    text = _MD_CODE_BLOCK_RE.sub(" ", text)
    text = _MD_INLINE_CODE_RE.sub(r"\1", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _MD_HEADING_RE.sub("", text)
    text = _MD_BULLET_RE.sub("", text)
    text = _MD_EMPHASIS_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()
