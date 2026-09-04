"""
Unit tests for SQL extraction/sanitization — no live database needed.
Tests that _extract_sql correctly pulls SQL out of messy LLM output, and
critically, that it refuses anything that isn't a read-only SELECT/WITH.
"""
from dotenv import load_dotenv
load_dotenv()

import pytest
from src.sql_rag.sql_rag_chain import _extract_sql

def test_extracts_plain_sql():
    raw = "SELECT COUNT(*) FROM claims WHERE status = 'pending'"
    assert _extract_sql(raw) == raw


def test_strips_markdown_code_fence_with_sql_tag():
    raw = "```sql\nSELECT * FROM claims\n```"
    assert _extract_sql(raw) == "SELECT * FROM claims"


def test_strips_markdown_code_fence_without_tag():
    raw = "```\nSELECT * FROM claims\n```"
    assert _extract_sql(raw) == "SELECT * FROM claims"


def test_strips_leading_explanation_text():
    raw = "Here's the SQL query you need:\nSELECT * FROM claims"
    assert _extract_sql(raw) == "SELECT * FROM claims"


def test_strips_trailing_semicolon():
    raw = "SELECT * FROM claims;"
    assert _extract_sql(raw) == "SELECT * FROM claims"


def test_accepts_with_clause():
    raw = "WITH recent AS (SELECT * FROM claims) SELECT * FROM recent"
    assert _extract_sql(raw) == raw


def test_rejects_drop_table():
    with pytest.raises(ValueError, match="Refusing to execute"):
        _extract_sql("DROP TABLE claims")


def test_rejects_delete():
    with pytest.raises(ValueError, match="Refusing to execute"):
        _extract_sql("DELETE FROM claims WHERE claim_id = 'CLM-2024-1000'")


def test_rejects_update():
    with pytest.raises(ValueError, match="Refusing to execute"):
        _extract_sql("UPDATE claims SET status = 'approved'")


def test_rejects_drop_table_wrapped_in_markdown():
    # A malicious/hallucinated destructive query hidden inside a normal-
    # looking code fence must still be caught - the safety check runs on
    # the EXTRACTED sql, after fence-stripping, not before.
    raw = "```sql\nDROP TABLE claims\n```"
    with pytest.raises(ValueError, match="Refusing to execute"):
        _extract_sql(raw)


def test_rejects_insert():
    with pytest.raises(ValueError, match="Refusing to execute"):
        _extract_sql("INSERT INTO claims (claim_id) VALUES ('FAKE-001')")