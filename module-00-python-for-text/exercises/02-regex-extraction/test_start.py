"""Tests for Exercise 02 — Regex Extraction."""

import start


# ---------------------------------------------------------------------------
# find_case_ids
# ---------------------------------------------------------------------------

class TestFindCaseIds:
    def test_multiple(self):
        assert start.find_case_ids("CASE-42 and CASE-107") == ["CASE-42", "CASE-107"]

    def test_none(self):
        assert start.find_case_ids("no cases here") == []

    def test_embedded_in_sentence(self):
        ids = start.find_case_ids("Refer to CASE-3 for background.")
        assert ids == ["CASE-3"]


# ---------------------------------------------------------------------------
# find_dates
# ---------------------------------------------------------------------------

class TestFindDates:
    def test_single_date(self):
        assert start.find_dates("Recorded on 1947-03-12.") == ["1947-03-12"]

    def test_multiple_dates(self):
        assert start.find_dates("Between 1947-03-12 and 1948-01-05.") == [
            "1947-03-12",
            "1948-01-05",
        ]

    def test_no_dates(self):
        assert start.find_dates("No dates here.") == []


# ---------------------------------------------------------------------------
# find_redacted
# ---------------------------------------------------------------------------

class TestFindRedacted:
    def test_redacted(self):
        assert start.find_redacted("[REDACTED] told me") == ["[REDACTED]"]

    def test_censored_lowercase(self):
        assert start.find_redacted("[censored] appeared") == ["[censored]"]

    def test_mixed(self):
        markers = start.find_redacted("[REDACTED] and [Censored]")
        assert len(markers) == 2

    def test_none(self):
        assert start.find_redacted("nothing hidden") == []


# ---------------------------------------------------------------------------
# strip_punctuation
# ---------------------------------------------------------------------------

class TestStripPunctuation:
    def test_removes_commas_and_periods(self):
        assert start.strip_punctuation("Hello, world!") == "Hello world"

    def test_keeps_numbers(self):
        assert start.strip_punctuation("Case #42.") == "Case 42"

    def test_already_clean(self):
        assert start.strip_punctuation("no punctuation") == "no punctuation"


# ---------------------------------------------------------------------------
# extract_statement_ids
# ---------------------------------------------------------------------------

class TestExtractStatementIds:
    def test_finds_ids(self):
        assert start.extract_statement_ids("STM-001 and STM-045") == [
            "STM-001",
            "STM-045",
        ]

    def test_none(self):
        assert start.extract_statement_ids("no statement ids") == []


# ---------------------------------------------------------------------------
# mask_names
# ---------------------------------------------------------------------------

class TestMaskNames:
    def test_single_name(self):
        assert start.mask_names("Reeves was here", ["Reeves"]) == "*** was here"

    def test_multiple_names(self):
        result = start.mask_names("Reeves met Hayes", ["Reeves", "Hayes"])
        assert result == "*** met ***"

    def test_case_insensitive(self):
        result = start.mask_names("reeves was HERE", ["Reeves"])
        assert result == "*** was HERE"

    def test_no_match(self):
        assert start.mask_names("Nobody here", ["Reeves"]) == "Nobody here"


# ---------------------------------------------------------------------------
# extract_times
# ---------------------------------------------------------------------------

class TestExtractTimes:
    def test_24_hour(self):
        assert start.extract_times("At 08:15 sharp.") == ["08:15"]

    def test_am_pm(self):
        assert start.extract_times("At 11:30pm.") == ["11:30pm"]

    def test_multiple(self):
        times = start.extract_times("From 08:15 to 11:30pm.")
        assert times == ["08:15", "11:30pm"]

    def test_none(self):
        assert start.extract_times("No time mentioned.") == []


# ---------------------------------------------------------------------------
# count_uppercase_words
# ---------------------------------------------------------------------------

class TestCountUppercaseWords:
    def test_mixed(self):
        assert start.count_uppercase_words("I saw HIM near the DOCKS") == 2

    def test_none(self):
        assert start.count_uppercase_words("all lowercase here") == 0

    def test_single_letter_excluded(self):
        # Single uppercase letters like 'I' should NOT count
        assert start.count_uppercase_words("I went to the DOCKS") == 1
