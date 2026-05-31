"""Tests for Exercise 03 — Corpus Loader."""

from collections import Counter

import start


# ---------------------------------------------------------------------------
# load_statements
# ---------------------------------------------------------------------------

class TestLoadStatements:
    def test_returns_list(self):
        records = start.load_statements()
        assert isinstance(records, list)

    def test_has_records(self):
        records = start.load_statements()
        assert len(records) > 0

    def test_record_has_expected_keys(self):
        records = start.load_statements()
        r = records[0]
        assert "id" in r
        assert "case_id" in r
        assert "raw_text" in r


# ---------------------------------------------------------------------------
# extract_texts
# ---------------------------------------------------------------------------

class TestExtractTexts:
    def test_returns_strings(self):
        records = [{"raw_text": "hello"}, {"raw_text": "world"}]
        result = start.extract_texts(records)
        assert result == ["hello", "world"]

    def test_from_real_data(self):
        records = start.load_statements()
        texts = start.extract_texts(records)
        assert len(texts) == len(records)
        assert all(isinstance(t, str) for t in texts)


# ---------------------------------------------------------------------------
# word_frequencies
# ---------------------------------------------------------------------------

class TestWordFrequencies:
    def test_basic(self):
        freq = start.word_frequencies(["the dog the cat"])
        assert isinstance(freq, Counter)
        assert freq["the"] == 2
        assert freq["dog"] == 1

    def test_lowercases(self):
        freq = start.word_frequencies(["THE The the"])
        assert freq["the"] == 3

    def test_multiple_texts(self):
        freq = start.word_frequencies(["hello world", "hello again"])
        assert freq["hello"] == 2
        assert freq["world"] == 1


# ---------------------------------------------------------------------------
# top_words
# ---------------------------------------------------------------------------

class TestTopWords:
    def test_returns_n(self):
        freq = Counter({"the": 10, "a": 5, "an": 3, "is": 1})
        result = start.top_words(freq, 2)
        assert len(result) == 2
        assert result[0] == ("the", 10)

    def test_default_n(self):
        freq = Counter({f"w{i}": i for i in range(20)})
        result = start.top_words(freq)
        assert len(result) == 10


# ---------------------------------------------------------------------------
# filter_by_case
# ---------------------------------------------------------------------------

class TestFilterByCase:
    def test_filters(self):
        records = [
            {"case_id": "CASE-1", "text": "a"},
            {"case_id": "CASE-2", "text": "b"},
            {"case_id": "CASE-1", "text": "c"},
        ]
        result = start.filter_by_case(records, "CASE-1")
        assert len(result) == 2
        assert all(r["case_id"] == "CASE-1" for r in result)

    def test_no_match(self):
        records = [{"case_id": "CASE-1", "text": "a"}]
        assert start.filter_by_case(records, "CASE-99") == []


# ---------------------------------------------------------------------------
# group_by_witness
# ---------------------------------------------------------------------------

class TestGroupByWitness:
    def test_groups(self):
        records = [
            {"witness": "Alice", "text": "a"},
            {"witness": "Bob", "text": "b"},
            {"witness": "Alice", "text": "c"},
        ]
        groups = start.group_by_witness(records)
        assert isinstance(groups, dict)
        assert len(groups["Alice"]) == 2
        assert len(groups["Bob"]) == 1

    def test_returns_plain_dict(self):
        records = [{"witness": "Alice", "text": "a"}]
        groups = start.group_by_witness(records)
        assert type(groups) is dict


# ---------------------------------------------------------------------------
# unique_case_ids
# ---------------------------------------------------------------------------

class TestUniqueCaseIds:
    def test_returns_set(self):
        records = [
            {"case_id": "CASE-1"},
            {"case_id": "CASE-2"},
            {"case_id": "CASE-1"},
        ]
        result = start.unique_case_ids(records)
        assert isinstance(result, set)
        assert result == {"CASE-1", "CASE-2"}


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_has_expected_keys(self):
        records = start.load_statements()
        s = start.summary(records)
        assert "total" in s
        assert "cases" in s
        assert "witnesses" in s
        assert "top_words" in s

    def test_total_matches_length(self):
        records = start.load_statements()
        s = start.summary(records)
        assert s["total"] == len(records)

    def test_cases_sorted(self):
        records = start.load_statements()
        s = start.summary(records)
        assert s["cases"] == sorted(s["cases"])
