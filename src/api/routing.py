"""Decides whether a question needs SQL RAG (analytical) or Hybrid RAG (document lookup)."""
from src.llm_client import get_llm_client, LLM_MODEL


def classify_question_type(question: str) -> str:
    client = get_llm_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": (
                "Classify the question as exactly one word: 'analytical' or 'document'.\n\n"
                "Answer 'analytical' if the question asks for EITHER: (a) a numeric "
                "aggregation (count, sum, total, average) over the claims or "
                "maintenance_tickets database tables, OR (b) a specific record or its "
                "details from those tables (e.g. the most recent claim, a particular "
                "claim's status/amount, the latest maintenance ticket).\n\n"
                "Answer 'document' for EVERYTHING else, including: clinical protocols, drug "
                "info, nursing procedures, equipment manuals, billing policies/procedures "
                "(as opposed to billing DATA), HR questions (leave, staff conduct), "
                "IT/facilities support questions (VPN, passwords, general database access "
                "requests), or any general or vague question.\n\n"
                "If you are not confident the question needs data from claims/"
                "maintenance_tickets, answer 'document'.\n\n"
                "Examples:\n"
                "'How many billing claims are pending?' -> analytical\n"
                "'What is the total approved amount this quarter?' -> analytical\n"
                "'What was our last bill?' -> analytical\n"
                "'Show me the most recent claim' -> analytical\n"
                "'What is the status of the latest maintenance ticket?' -> analytical\n"
                "'I have an issue with my VPN' -> document\n"
                "'What are the leave policy rules?' -> document\n"
                "'Want to update the database' -> document\n"
                "'Who approves leave for ICU doctors?' -> document\n\n"
                "Answer with only one word: analytical or document."
            )},
            {"role": "user", "content": question},
        ],
        temperature=0,
    )
    result = response.choices[0].message.content.strip().lower()
    return "analytical" if "analytical" in result else "document"