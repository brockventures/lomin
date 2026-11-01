#!/usr/bin/env python3
"""Tests for ./sentence_segmenter.py.

Run: python3 -m pytest tests/test_sentence_segmenter.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentence_segmenter import SentenceSegmenter, segment_full_text  # noqa: E402


def test_basic_two_sentences_one_shot():
    assert segment_full_text("Hello there. How are you?") == [
        "Hello there.", "How are you?"
    ]


def test_flush_returns_tail_without_terminal_punctuation():
    seg = SentenceSegmenter()
    assert seg.feed("This is unfinished") == []
    assert seg.flush() == "This is unfinished"


def test_abbreviation_mr_not_split():
    out = segment_full_text("Mr. Smith went home. He was tired.")
    assert out == ["Mr. Smith went home.", "He was tired."]


def test_abbreviation_st_not_split():
    # "St." is a known abbreviation, so the dot after it is never treated
    # as a boundary -- including the second, truly sentence-final "St."
    # here. That's a deliberate trade-off (see module docstring): never
    # mis-splitting mid-abbreviation costs an occasional missed real
    # boundary, which is the safer failure mode for TTS chunking.
    out = segment_full_text("We live on St. James St. It is quiet.")
    assert out == ["We live on St. James St. It is quiet."]


def test_decimal_not_split():
    out = segment_full_text("Pi is about 3.14 and that's fine. Next sentence.")
    assert out == ["Pi is about 3.14 and that's fine.", "Next sentence."]


def test_version_number_not_split():
    out = segment_full_text("Upgrade to v2.5 today. It fixes bugs.")
    assert out == ["Upgrade to v2.5 today.", "It fixes bugs."]


def test_initials_not_split():
    out = segment_full_text("J. R. R. Tolkien wrote it. It is long.")
    assert out == ["J. R. R. Tolkien wrote it.", "It is long."]


def test_ellipsis_treated_as_one_terminator():
    # "..." followed by whitespace is itself a complete terminator (one
    # boundary, not three) -- it does not require a fourth dot or more
    # text to close it out.
    out = segment_full_text("Well... I'm not sure. Let's see.")
    assert out == ["Well...", "I'm not sure.", "Let's see."]


def test_ellipsis_unicode_glyph():
    out = segment_full_text("Well… maybe. Or not.")
    assert out == ["Well…", "maybe.", "Or not."]


def test_quoted_sentence_end():
    out = segment_full_text('He said "no." Then he left.')
    assert out == ['He said "no."', "Then he left."]


def test_exclamation_and_question():
    out = segment_full_text("Wait! Are you sure? Yes.")
    assert out == ["Wait!", "Are you sure?", "Yes."]


def test_eg_ie_abbreviations_not_split():
    out = segment_full_text("Bring supplies, e.g. water and food. Then go.")
    assert out == ["Bring supplies, e.g. water and food.", "Then go."]


def test_empty_input():
    assert segment_full_text("") == []
    seg = SentenceSegmenter()
    assert seg.feed("") == []
    assert seg.flush() is None


# --------------------------------------------------------------------------
# Streaming behavior: sentences emitted incrementally as text arrives,
# never re-emitted, order preserved, final flush gets the tail.
# --------------------------------------------------------------------------

def test_streaming_emits_sentence_as_soon_as_boundary_seen():
    seg = SentenceSegmenter()
    assert seg.feed("Hello there.") == []  # no trailing space yet -> not confirmed
    assert seg.feed(" How") == ["Hello there."]
    assert seg.feed(" are you?") == []
    assert seg.flush() == "How are you?"


def test_streaming_char_by_char_matches_one_shot():
    text = "First sentence here. Second one follows! Is this the third? Yes it is."
    seg = SentenceSegmenter()
    out: list[str] = []
    for ch in text:
        out.extend(seg.feed(ch))
    tail = seg.flush()
    if tail:
        out.append(tail)
    assert out == segment_full_text(text)


def test_streaming_abbreviation_across_chunk_boundary():
    seg = SentenceSegmenter()
    out = []
    out += seg.feed("We visited Dr")
    out += seg.feed(". Smith yesterday")
    out += seg.feed(". It went well.")
    tail = seg.flush()
    if tail:
        out.append(tail)
    assert out == ["We visited Dr. Smith yesterday.", "It went well."]


def test_streaming_never_reemits_a_sentence():
    seg = SentenceSegmenter()
    seen = []
    for chunk in ["One. ", "Two. ", "Three."]:
        seen += seg.feed(chunk)
    tail = seg.flush()
    if tail:
        seen.append(tail)
    assert seen == ["One.", "Two.", "Three."]
    assert len(seen) == len(set(seen)) or seen.count("Two.") == 1


def test_final_text_parity_join_of_sentences_reconstructs_content():
    text = "Alpha beta gamma. Delta epsilon zeta! Eta theta iota?"
    sentences = segment_full_text(text)
    # Joining sentences with single spaces reconstructs the original text
    # (modulo whitespace normalization) -- no words dropped or duplicated.
    assert " ".join(sentences) == text


# --- early first-clause flush (early_first=True) ---------------------------

def _early():
    return SentenceSegmenter(early_first=True)


def test_default_segmenter_still_waits_for_whitespace_after_terminator():
    seg = SentenceSegmenter()
    assert seg.feed("Sure thing.") == []


def test_early_flushes_terminator_at_end_of_first_partial():
    seg = _early()
    assert seg.feed("Sure thing.") == ["Sure thing."]
    assert seg.pending() == ""


def test_early_flushes_first_clause_at_comma():
    seg = _early()
    out = seg.feed("Well, it looks like the weather is turning, and then some more")
    assert out == ["Well, it looks like the weather is turning,"]
    assert seg.pending().strip() == "and then some more"


def test_early_short_lead_in_does_not_cut_at_comma():
    seg = _early()
    assert seg.feed("Well, it") == []


def test_early_skips_decimal_and_abbreviation_at_buffer_end():
    seg = _early()
    assert seg.feed("The temperature today is about 3.") == []
    seg = _early()
    assert seg.feed("I spoke with Dr.") == []


def test_early_word_cap_cuts_on_complete_word_only():
    seg = _early()
    text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
    out = seg.feed(text)
    assert out == ["one two three four five six seven eight nine ten eleven twelve"]
    assert seg.pending().strip() == "thirteen fourteen"
    # a partial last word is never split
    seg = _early()
    assert seg.feed("one two three four five six seven eight nine ten eleven twelve") == []


def test_early_applies_to_first_cut_only():
    seg = _early()
    assert seg.feed("Sure thing.") == ["Sure thing."]
    # second utterance: back to the strict rules
    assert seg.feed(" Another short one.") == []
    assert seg.flush() == "Another short one."


def test_early_prefers_short_complete_sentence_over_clause_cut():
    seg = _early()
    out = seg.feed("Yes, I can do that for you right now. And")
    assert out == ["Yes, I can do that for you right now."]
