"""Tests for Exercise 01 — String Sleuthing."""

import start


# ---------------------------------------------------------------------------
# normalise
# ---------------------------------------------------------------------------

class TestNormalise:
    def test_strips_and_lowercases(self):
        assert start.normalise("  Hello World  ") == "hello world"

    def test_already_clean(self):
        assert start.normalise("clean") == "clean"

    def test_tabs_and_newlines(self):
        assert start.normalise("\tLine one\n") == "line one"


# ---------------------------------------------------------------------------
# word_count
# ---------------------------------------------------------------------------

class TestWordCount:
    def test_simple(self):
        assert start.word_count("I saw Reeves near the docks") == 6

    def test_extra_whitespace(self):
        assert start.word_count("  too   many   spaces  ") == 3

    def test_empty(self):
        assert start.word_count("") == 0


# ---------------------------------------------------------------------------
# extract_words
# ---------------------------------------------------------------------------

class TestExtractWords:
    def test_lowercases_tokens(self):
        assert start.extract_words("He SAW the DOCKS") == ["he", "saw", "the", "docks"]

    def test_single_word(self):
        assert start.extract_words("Reeves") == ["reeves"]

    def test_empty(self):
        assert start.extract_words("") == []


# ---------------------------------------------------------------------------
# contains_keyword
# ---------------------------------------------------------------------------

class TestContainsKeyword:
    def test_present_exact_case(self):
        assert start.contains_keyword("I saw Reeves at the docks", "Reeves") is True

    def test_present_different_case(self):
        assert start.contains_keyword("I saw REEVES at the docks", "reeves") is True

    def test_absent(self):
        assert start.contains_keyword("I saw nobody", "Reeves") is False


# ---------------------------------------------------------------------------
# censor_word
# ---------------------------------------------------------------------------

class TestCensorWord:
    def test_replaces_case_insensitive(self):
        result = start.censor_word("Reeves was seen at the docks", "Reeves")
        assert result == "[CENSORED] was seen at the docks"

    def test_replaces_all_occurrences(self):
        result = start.censor_word("Reeves met reeves", "Reeves")
        assert result == "[CENSORED] met [CENSORED]"

    def test_no_match(self):
        text = "Nobody was here"
        assert start.censor_word(text, "Reeves") == text


# ---------------------------------------------------------------------------
# initials
# ---------------------------------------------------------------------------

class TestInitials:
    def test_two_names(self):
        assert start.initials("Margaret Hayes") == "M.H."

    def test_three_names(self):
        assert start.initials("James T Kirk") == "J.T.K."

    def test_single_name(self):
        assert start.initials("Reeves") == "R."


# ---------------------------------------------------------------------------
# format_summary
# ---------------------------------------------------------------------------

class TestFormatSummary:
    def test_basic(self):
        result = start.format_summary("Margaret Hayes", "CASE-42", 38)
        assert result == "Witness: Margaret Hayes | Case: CASE-42 | Words: 38"

    def test_zero_words(self):
        result = start.format_summary("Anon", "CASE-1", 0)
        assert result == "Witness: Anon | Case: CASE-1 | Words: 0"


# ---------------------------------------------------------------------------
# truncate
# ---------------------------------------------------------------------------

class TestTruncate:
    def test_no_truncation_needed(self):
        assert start.truncate("short", 10) == "short"

    def test_exact_length(self):
        assert start.truncate("exact", 5) == "exact"

    def test_truncates_with_ellipsis(self):
        assert start.truncate("a]long statement here", 10) == "a]long ..."

    def test_very_short_max(self):
        assert start.truncate("hello", 2) == "he"
