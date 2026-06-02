"""Tests for Capstone Exercise 02 - Case Chatbot."""

from unittest.mock import MagicMock, patch

import pytest

import start


class TestLoadCaseDocuments:
    def test_returns_documents_for_case_42(self):
        docs = start.load_case_documents("CASE-42")
        assert len(docs) >= 1
        assert all({"id", "source", "text"} <= set(d.keys()) for d in docs)

    def test_has_statements_and_archive(self):
        docs = start.load_case_documents("CASE-42")
        sources = {d["source"] for d in docs}
        assert "statement" in sources
        assert "archive" in sources

    def test_empty_for_unknown_case(self):
        docs = start.load_case_documents("CASE-999")
        assert docs == []


class TestBuildTfidfIndex:
    def test_returns_vectorizer_and_matrix(self):
        docs = start.load_case_documents("CASE-42")
        vectorizer, matrix = start.build_tfidf_index(docs)
        assert matrix.shape[0] == len(docs)
        assert hasattr(vectorizer, "transform")


class TestRetrieveContext:
    def test_returns_top_k_documents(self):
        docs = start.load_case_documents("CASE-42")
        vectorizer, matrix = start.build_tfidf_index(docs)
        results = start.retrieve_context("docks Reeves", vectorizer, matrix, docs, top_k=2)
        assert len(results) <= 2
        assert all("text" in d for d in results)

    def test_results_are_from_original_docs(self):
        docs = start.load_case_documents("CASE-42")
        vectorizer, matrix = start.build_tfidf_index(docs)
        results = start.retrieve_context("warehouse pier", vectorizer, matrix, docs, top_k=3)
        doc_ids = {d["id"] for d in docs}
        for r in results:
            assert r["id"] in doc_ids


class TestBuildPrompt:
    def test_contains_question_and_context(self):
        context = [{"id": "STM-001", "source": "statement", "text": "I saw him at the docks."}]
        prompt = start.build_prompt("Who was at the docks?", context, "CASE-42")
        assert "CASE-42" in prompt
        assert "docks" in prompt
        assert "Who was at the docks?" in prompt

    def test_contains_document_ids(self):
        context = [
            {"id": "STM-001", "source": "statement", "text": "Statement one."},
            {"id": "ARC-001", "source": "archive", "text": "Archive one."},
        ]
        prompt = start.build_prompt("test", context, "CASE-42")
        assert "STM-001" in prompt
        assert "ARC-001" in prompt


class TestGenerateAnswer:
    def test_returns_string(self):
        mock_model = MagicMock()
        mock_model.return_value = [{"generated_text": "Reeves was seen at the docks."}]
        answer = start.generate_answer(mock_model, "test prompt")
        assert isinstance(answer, str)
        assert len(answer) > 0


class TestExtractResponseEntities:
    def test_finds_entities(self):
        entities = start.extract_response_entities(
            "Margaret Hayes saw Reeves on Tuesday near London."
        )
        assert isinstance(entities, dict)
        all_names = []
        for names in entities.values():
            all_names.extend(names)
        assert len(all_names) >= 1


class TestValidateEntities:
    def test_grounded_entities(self):
        docs = [{"id": "1", "source": "statement", "text": "Reeves was at the docks."}]
        entities = {"PERSON": ["Reeves"], "LOC": ["Mars"]}
        result = start.validate_entities(entities, docs)
        assert "Reeves" in result["grounded"]
        assert "Mars" in result["ungrounded"]

    def test_empty_entities(self):
        docs = [{"id": "1", "source": "statement", "text": "Some text."}]
        result = start.validate_entities({}, docs)
        assert result["grounded"] == []
        assert result["ungrounded"] == []
