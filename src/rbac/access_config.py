"""
Central role <-> collection access mapping.
Used both at ingestion time (to tag each chunk) and at query time (to build the RBAC filter).
"""

COLLECTION_ACCESS = {
    "general":   ["doctor", "nurse", "billing_executive", "technician", "admin"],
    "clinical":  ["doctor", "admin"],
    "nursing":   ["nurse", "doctor", "admin"],
    "billing":   ["billing_executive", "admin"],
    "equipment": ["technician", "admin"],
}

COLLECTIONS = list(COLLECTION_ACCESS.keys())

def get_access_roles(collection: str) -> list[str]:
    if collection not in COLLECTION_ACCESS:
        raise ValueError(f"Unknown collection: {collection}")
    return COLLECTION_ACCESS[collection]

ROLES = {"doctor", "nurse", "billing_executive", "technician", "admin"}

def validate_role(role: str) -> None:
    if role not in ROLES:
        raise ValueError(f"Unknown role: {role}. Must be one of {sorted(ROLES)}")

SQL_RAG_ALLOWED_ROLES = {"billing_executive", "admin"}

def can_use_sql_rag(role: str) -> bool:
    return role in SQL_RAG_ALLOWED_ROLES