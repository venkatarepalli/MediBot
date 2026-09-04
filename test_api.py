from dotenv import load_dotenv
load_dotenv()

import requests

BASE_URL = "http://127.0.0.1:8000"


def login(username: str, password: str) -> dict:
    resp = requests.post(f"{BASE_URL}/login", json={"username": username, "password": password})
    resp.raise_for_status()
    return resp.json()


def chat(token: str, question: str) -> tuple[int, dict]:
    resp = requests.post(
        f"{BASE_URL}/chat",
        headers={"authorization": f"Bearer {token}"},
        json={"question": question},
    )
    return resp.status_code, resp.json()

def health() -> dict:
    resp = requests.get(f"{BASE_URL}/health")
    resp.raise_for_status()
    return resp.json()

def chat_empty_question(token: str) -> tuple[int, dict]:
    resp = requests.post(
        f"{BASE_URL}/chat",
        headers={"authorization": f"Bearer {token}"},
        json={"question": ""},
    )
    return resp.status_code, resp.json()

def get_collections_bad_role() -> tuple[int, dict]:
    resp = requests.get(f"{BASE_URL}/collections/nope")
    return resp.status_code, resp.json()

if __name__ == "__main__":
    print("[Health check]")
    print(health())

    def run(role_label, username, password, question, expect):
        login_result = login(username, password)
        status, result = chat(login_result["token"], question)
        print(f"\n[{role_label}] expect: {expect}")
        print(f"Q: {question}")
        print(f"status={status}  answer={result['answer']}")
        if result["sources"]:
            print(f"sources: {[(s['collection'], s['source_document']) for s in result['sources']]}")

    # --- Doctor: nursing access (new) + equipment blocked (new) ---
    run("Doctor - nursing question", "dr.mehta", "doctor",
        "What is the correct IV cannula size for a paediatric patient under 5kg?",
        "should succeed, sourced from nursing")

    run("Doctor - equipment question (blocked)", "dr.mehta", "doctor",
        "What is the default occlusion pressure alarm setting for the DriveFlow IP-200?",
        "should refuse - doctor has no equipment access")

    # --- Nurse: equipment blocked (new) ---
    run("Nurse - equipment question (blocked)", "nurse.priya", "nurse",
        "What is the default occlusion pressure alarm setting for the DriveFlow IP-200?",
        "should refuse - nurse has no equipment access")

    # --- Billing executive: legit document question (new) + clinical blocked (new) ---
    run("Billing executive - billing document question", "billing.ravi", "billing_executive",
        "What is the pre-authorisation process for cashless claims?",
        "should succeed, sourced from billing")

    run("Billing executive - clinical question (blocked)", "billing.ravi", "billing_executive",
        "What is the first-line treatment for Type 2 Diabetes?",
        "should refuse - billing_executive has no clinical access")

    # --- Technician: clinical blocked (new) + SQL blocked via /chat (new) ---
    run("Technician - clinical question (blocked)", "tech.anand", "technician",
        "What is the first-line treatment for Type 2 Diabetes?",
        "should refuse - technician has no clinical access")

    run("Technician - analytical question (blocked)", "tech.anand", "technician",
        "How many billing claims are currently pending?",
        "should refuse - technician has no SQL RAG access")

    # --- Admin: confirm 'access everything' across clinical/nursing/equipment (new) ---
    run("Admin - clinical question", "admin.sys", "admin",
        "What is the first-line treatment for Type 2 Diabetes?",
        "should succeed, sourced from clinical")

    run("Admin - nursing question", "admin.sys", "admin",
        "What are the hand hygiene moments?",
        "should succeed, sourced from nursing")

    run("Admin - equipment question", "admin.sys", "admin",
        "What is the default occlusion pressure alarm setting for the DriveFlow IP-200?",
        "should succeed, sourced from equipment")

    print("\n[Empty question validation]")
    admin_login = login("admin.sys", "admin")
    status, result = chat_empty_question(admin_login["token"])
    print(f"status={status}")
    print(result)

    print("\n[Bad role validation]")
    status, result = get_collections_bad_role()
    print(f"status={status}")
    print(result)