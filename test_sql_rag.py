from dotenv import load_dotenv
load_dotenv()

from src.sql_rag.sql_rag_chain import sql_rag_chain

QUESTIONS = [
    "How many billing claims are currently pending?",
    "What is the total approved amount for claims from New India Assurance?",
    "How many maintenance tickets are still in progress?",
    "Which equipment category has the most maintenance tickets?",
]

if __name__ == "__main__":
    for q in QUESTIONS:
        print(f"\n{'='*70}\nQ: {q}\n{'='*70}")
        result = sql_rag_chain(q, verbose=True)
        print(f"SQL:  {result['sql']}")
        print(f"Rows: {len(result['rows'])}")
        print(f"Answer: {result['answer']}")