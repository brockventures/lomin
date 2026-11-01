#!/usr/bin/env python3
"""sentence_segmenter.py — cuts a growing stream of text into complete
sentences as soon as they're unambiguously finished, for MANDOS streaming
TTS (specs/2026-09-28-mandos-streaming: "generate audio as soon as a
sentence comes in").

Feed it text as it arrives (from a partial brain-call answer, or a full
string in one shot) via `feed()`; it returns the list of sentences that
became complete as a result of that call, in order. Call `flush()` once
the source is exhausted to get whatever's left in the buffer (a reply
that doesn't end in terminal punctuation, or the last sentence when the
final punctuation was never followed by more text/whitespace).

Handles:
  - Abbreviations ("Mr.", "Dr.", "St.", "vs.", "approx.", ...) and single
    initials ("J. R. R. Tolkien") — the dot after them is never treated
    as a sentence boundary.
  - Decimals ("3.14", "v2.5") — a dot flanked by digits on both sides is
    never a boundary.
  - Ellipses ("...", "…") — treated as one terminator, not three separate
    stops.
  - Trailing closing quotes/brackets after the terminator (`He said "no."`)
    — the boundary check looks past them before requiring whitespace.

A terminator is only treated as a completed sentence once at least one
whitespace character after it (and any trailing quote/bracket) is already
in the buffer — otherwise "Mr" could look sentence-final just because the
next chunk (the "." plus more text) hasn't arrived yet. That's exactly
why `flush()` exists: it hands back the tail with no such lookahead once
the caller knows no more text is coming.
"""
from __future__ import annotations

import re
from typing import Optional

# Common abbreviations whose trailing "." must never be read as a sentence
# boundary. Lowercased, no trailing dot.
ABBREVIATIONS = {
    "mr", "mrs", "ms", "mx", "dr", "st", "prof", "sr", "jr", "vs", "etc",
    "approx", "fig", "figs", "inc", "ltd", "co", "corp",
    "dept", "univ", "gov", "rev", "gen", "col", "lt", "cmdr", "capt",
    "sgt", "al", "cf", "viz", "est", "vol", "pp", "ave", "blvd", "rd",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec", "mon", "tue", "wed", "thu", "fri", "sat", "sun",
}

# Early first-clause flush defaults (see SentenceSegmenter early_first).
FIRST_CLAUSE_MIN_CHARS = 20
FIRST_CLAUSE_MIN_WORDS = 4
FIRST_CLAUSE_MAX_WORDS = 12
_CLAUSE_PUNCT = ",;:\u2014"

_CLOSERS = '"\')]}”’›»'
_TERMINATORS = ".!?…"  # includes the single-char ellipsis glyph


def _word_before(buf: str, idx: int) -> str:
    """The run of alnum chars immediately before buf[idx]."""
    start = idx
    while start > 0 and buf[start - 1].isalnum():
        start -= 1
    return buf[start:idx]


def _is_abbrev_or_initial(buf: str, dot_idx: int) -> bool:
    word = _word_before(buf, dot_idx).lower()
    if not word:
        return False
    if word in ABBREVIATIONS:
        return True
    # A single letter immediately before the dot: an initial ("J.") or the
    # first/second dot of a multi-dot abbreviation ("e.g.", "i.e.").
    if len(word) == 1 and word.isalpha():
        return True
    return False


def _is_decimal_point(buf: str, dot_idx: int) -> bool:
    before = dot_idx > 0 and buf[dot_idx - 1].isdigit()
    after = dot_idx + 1 < len(buf) and buf[dot_idx + 1].isdigit()
    return before and after


class SentenceSegmenter:
    """Stateful — feed it successive chunks of a single growing reply."""

    def __init__(self, early_first: bool = False,
                 first_min_chars: int = FIRST_CLAUSE_MIN_CHARS,
                 first_min_words: int = FIRST_CLAUSE_MIN_WORDS,
                 first_max_words: int = FIRST_CLAUSE_MAX_WORDS) -> None:
        self._buf = ""
        # Early first-clause flush (off by default: the plain segmenter only
        # cuts at an unambiguous boundary). With it on, the FIRST cut of the
        # reply may also be made (a) at a terminator that ends the buffer with
        # no trailing whitespace yet, (b) at a comma/colon/semicolon/dash once
        # a minimum length is met, or (c) after first_max_words complete
        # words. Everything after the first cut uses the normal rules.
        self._early_first = early_first
        self._first_min_chars = first_min_chars
        self._first_min_words = first_min_words
        self._first_max_words = first_max_words
        self._emitted = 0

    def feed(self, new_text: str) -> list[str]:
        """Append `new_text` and return any sentences that just became
        complete, in order. May return an empty list."""
        if new_text:
            self._buf += new_text
        sentences: list[str] = []
        while True:
            cut = self._find_boundary(self._buf)
            if self._early_first and self._emitted == 0:
                early = self._find_early_cut(self._buf)
                if early is not None and (cut is None or early < cut):
                    # A full sentence already in hand wins only when it is
                    # short enough to be a good first utterance.
                    if cut is None or len(self._buf[:cut].split()) > self._first_max_words:
                        cut = early
            if cut is None:
                break
            sentence = self._buf[:cut].strip()
            self._buf = self._buf[cut:]
            if sentence:
                sentences.append(sentence)
                self._emitted += 1
        return sentences

    def flush(self) -> Optional[str]:
        """Hand back whatever's left in the buffer (stripped), clearing
        it. Returns None if there's nothing left. Call this once, when
        the source of text is known to be exhausted."""
        tail = self._buf.strip()
        self._buf = ""
        return tail or None

    def pending(self) -> str:
        """Peek at the unflushed buffer without consuming it (diagnostics/tests)."""
        return self._buf

    def _find_early_cut(self, buf: str) -> Optional[int]:
        """Earliest acceptable early cut of the first utterance, or None.
        Never cuts inside a word: a cut needs whitespace after it, or the
        end of the buffer for punctuation that cannot be the start of a
        longer token ("3." / "1," are skipped there: "3.14", "1,000")."""
        n = len(buf)
        best: Optional[int] = None
        # (b) clause punctuation once the lead-in is long enough.
        for i, ch in enumerate(buf):
            if ch not in _CLAUSE_PUNCT:
                continue
            head = buf[:i].strip()
            if len(head) < self._first_min_chars or len(head.split()) < self._first_min_words:
                continue
            at_end = i == n - 1
            if at_end:
                if buf[i - 1].isdigit():
                    continue
            elif not buf[i + 1].isspace():
                continue
            best = i + 1
            break
        # (a) a terminator ending the buffer (whitespace has not arrived yet).
        k = n
        while k > 0 and buf[k - 1] in _CLOSERS:
            k -= 1
        if k > 0 and buf[k - 1] in _TERMINATORS and any(c.isalnum() for c in buf[:k]):
            d = k - 1
            ok = True
            if buf[d] == "." and buf[d - 1:d] != "." and (
                    _is_abbrev_or_initial(buf, d) or (d > 0 and buf[d - 1].isdigit())):
                ok = False
            if ok and best is None:
                best = n
        # (c) too many words with no boundary: cut after the max-th complete word.
        if best is None:
            for count, m in enumerate(re.finditer(r"\S+", buf), 1):
                if count == self._first_max_words and m.end() < n:
                    best = m.end()
                    break
        return best

    @staticmethod
    def _find_boundary(buf: str) -> Optional[int]:
        n = len(buf)
        i = 0
        while i < n:
            ch = buf[i]
            if ch not in _TERMINATORS:
                i += 1
                continue

            # Ellipsis: either the single glyph, or a run of 2+ dots.
            if ch == "…":
                end = i + 1
            elif ch == "." and buf[i:i + 2] == "..":
                end = i
                while end < n and buf[end] == ".":
                    end += 1
            elif ch == ".":
                if _is_abbrev_or_initial(buf, i) or _is_decimal_point(buf, i):
                    i += 1
                    continue
                end = i + 1
            else:  # "!" or "?"
                end = i + 1

            # Skip past any trailing closing quotes/brackets.
            k = end
            while k < n and buf[k] in _CLOSERS:
                k += 1

            if k < n and buf[k].isspace():
                return k + 1  # include the one whitespace char in the cut

            i = end
        return None


def segment_full_text(text: str) -> list[str]:
    """One-shot convenience: split a complete (non-streaming) string into
    sentences, in order, with nothing left over. Used by callers that
    already have the whole reply and just want sentence-parallel TTS."""
    seg = SentenceSegmenter()
    out = seg.feed(text)
    tail = seg.flush()
    if tail:
        out.append(tail)
    return out
