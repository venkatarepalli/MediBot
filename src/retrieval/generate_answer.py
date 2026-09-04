"""Turns the top reranked chunks into a final answer, or refuses if nothing relevant enough was found."""
from src.llm_client import get_llm_client, LLM_MODEL

# Below this cross-encoder score, the best match is too weak to answer from -
# triggers an RBAC-style refusal instead of a low-confidence guess.
# Calibrated from real test cases: legitimate answers scored -4.5 to -7.0
# (technician/infusion-pump, doctor/diabetes), genuinely off-topic content
# scored -8.3 to -10.7 (nurse asking about billing). -7.0 sits in the gap.
RELEVANCE_THRESHOLD = -7.0

# Sentinel the LLM must output verbatim if the context doesn't actually
# answer the question, even when retrieval score passed the threshold above.
# Checking for this exact token is far more reliable than guessing at every
# possible phrasing of "I don't know" the model might produce - and it lets
# both refusal paths (weak retrieval score / LLM itself declining) produce
# the SAME standardized message and correctly hide sources either way.
NO_ANSWER_SENTINEL = "NOT_FOUND"


def _refusal_message(role: str, allowed_collections: list[str]) -> str:
    collections_str = ", ".join(allowed_collections)
    return (
        f"As a {role}, I don't have sufficiently relevant information to answer that. "
        f"I can only answer questions from the {collections_str} collections."
    )


def generate_answer(question: str, role: str, allowed_collections: list[str], top_chunks: list[dict]) -> tuple[str, bool]:
    """Returns (answer, was_refused)."""
    if not top_chunks or top_chunks[0]["rerank_score"] < RELEVANCE_THRESHOLD:
        return _refusal_message(role, allowed_collections), True

    context = "\n\n".join(f"[Source: {c['source_document']} - {c['section_title']}]\n{c['text']}" for c in top_chunks)
    client = get_llm_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": (
                "Answer using ONLY the provided context. If the context does NOT "
                f"contain the answer, respond with EXACTLY this and nothing else: {NO_ANSWER_SENTINEL}. "
                "Otherwise, be concise."
            )},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0,
    )
    answer = response.choices[0].message.content.strip()

    if answer.upper() == NO_ANSWER_SENTINEL:
        return _refusal_message(role, allowed_collections), True

    return answer, False