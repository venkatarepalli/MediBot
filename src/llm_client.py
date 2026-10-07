"""Shared Groq client, used by both SQL RAG and the hybrid-RAG answer generator."""
import os
from groq import Groq

LLM_MODEL = "openai/gpt-oss-120b"
_groq_client = None

def get_llm_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _groq_client