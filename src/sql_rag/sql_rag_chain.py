"""
Phase 5: SQL RAG — translates a natural language question into SQL, runs it
against mediassist.db, and turns the result back into a natural language answer.
"""
import re
import sqlite3

from src.llm_client import get_llm_client, LLM_MODEL

DB_PATH = "data/mediassist_data/db/mediassist.db"

# Sentinel the LLM must emit if the question asks for something the schema
# genuinely cannot answer (e.g. "profit", "expenses" - there are no cost/
# revenue columns, only claimed_amount/approved_amount). Without this, the
# model invents a plausible-sounding but meaningless formula and answers
# with false confidence - verified directly: "profit" got computed as
# approved_amount - claimed_amount, which isn't profit in any real sense.
NO_ANSWER_SQL_SENTINEL = "NO_SQL_POSSIBLE"


def _get_schema_description(conn: sqlite3.Connection) -> str:
    cursor = conn.cursor()
    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    schema_parts = []
    for table_name, create_sql in tables:
        schema_parts.append(f"{create_sql};")

        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        for _, col_name, col_type, *_ in columns:
            if "CHAR" in col_type.upper() or "TEXT" in col_type.upper():
                cursor.execute(f"SELECT DISTINCT {col_name} FROM {table_name} LIMIT 25")
                distinct_vals = [r[0] for r in cursor.fetchall()]
                if 1 < len(distinct_vals) <= 15:
                    schema_parts.append(f"-- {table_name}.{col_name} actual values: {distinct_vals}")

    return "\n\n".join(schema_parts)


def _find_latest_date(conn: sqlite3.Connection) -> str | None:
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cursor.fetchall()]

    latest = None
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        for _, col_name, col_type, *_ in cursor.fetchall():
            if "date" in col_name.lower():
                cursor.execute(f"SELECT MAX({col_name}) FROM {table}")
                val = cursor.fetchone()[0]
                if val and (latest is None or val > latest):
                    latest = val
    return latest


def _question_to_sql(question: str, schema: str, reference_date: str | None) -> str:
    """Step 1: translate the natural language question into SQL using an LLM."""
    date_instruction = ""
    if reference_date:
        date_instruction = (
            f"\n\nThis is a historical dataset, not live data. When the question uses "
            f"relative time expressions ('this quarter', 'this month', 'recently', "
            f"'last N days'), treat {reference_date} as 'today' — the most recent date "
            f"actually present in the data — not the real-world current date."
        )
    scope_instruction = (
        f"\n\nIf the question asks for something this schema genuinely cannot answer "
        f"(e.g. profit, expenses, revenue, cost, margin, budget — there are no such "
        f"columns, only claimed_amount and approved_amount for claims), do NOT invent "
        f"a proxy calculation. Instead, respond with EXACTLY this and nothing else: "
        f"{NO_ANSWER_SQL_SENTINEL}"
    )
    client = get_llm_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": (
                "You are a SQLite expert. Given the database schema below, write "
                "a single SQLite query that answers the user's question. Output "
                "ONLY the SQL query — no explanation, no markdown.\n\nSchema:\n" + schema
                + date_instruction + scope_instruction
            )},
            {"role": "user", "content": question},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


def _extract_sql(raw_output: str) -> str:
    """Step 2: clean the raw LLM output down to just the SQL statement."""
    text = raw_output.strip()

    fence_match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    keyword_match = re.search(r"\b(SELECT|WITH)\b", text, re.IGNORECASE)
    if keyword_match:
        text = text[keyword_match.start():]

    sql = text.strip().rstrip(";").strip()

    if not re.match(r"^\s*(SELECT|WITH)\b", sql, re.IGNORECASE):
        raise ValueError(f"Refusing to execute non-SELECT SQL: {sql[:100]}")

    return sql


def _execute_sql(sql: str) -> tuple[list[str], list[tuple]]:
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        return columns, cursor.fetchall()
    finally:
        conn.close()


def _result_to_answer(question: str, sql: str, columns: list[str], rows: list[tuple]) -> str:
    client = get_llm_client()
    result_preview = f"Columns: {columns}\nRows ({len(rows)} total): {rows[:20]}"
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": (
                "You answer questions using SQL query results. Be concise and "
                "directly answer using the data given. Do not mention SQL or "
                "databases in your answer. All monetary amounts in this data "
                "are in Indian Rupees. Always format them with the ₹ symbol "
                "BEFORE the number, using Indian numbering convention (e.g. "
                "₹1,85,000 not ₹185,000; ₹91,000 not ₹91,000.0 or 91,000 ₹)."
            )},
            {"role": "user", "content": f"Question: {question}\n\nResult:\n{result_preview}"},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


def sql_rag_chain(question: str, verbose: bool = False):
    """
    Plain Python function implementing SQL RAG (per assignment spec, 3 steps:
    NL→SQL, clean/extract, execute+NL-answer). Returns the answer string.
    verbose=True returns a dict with sql/columns/rows too, for testing.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        schema = _get_schema_description(conn)
        reference_date = _find_latest_date(conn)
    finally:
        conn.close()

    raw_sql_output = _question_to_sql(question, schema, reference_date)

    if raw_sql_output.strip() == NO_ANSWER_SQL_SENTINEL:
        answer = (
            "I don't have the data needed to answer that — this system tracks "
            "billing claims and equipment maintenance records, but not profit, "
            "expenses, or revenue figures."
        )
        if verbose:
            return {"question": question, "sql": None, "columns": [], "rows": [], "answer": answer}
        return answer

    sql = _extract_sql(raw_sql_output)
    columns, rows = _execute_sql(sql)
    answer = _result_to_answer(question, sql, columns, rows)

    if verbose:
        return {"question": question, "sql": sql, "columns": columns, "rows": rows, "answer": answer}
    return answer