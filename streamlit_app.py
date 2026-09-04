"""
Streamlit frontend for MediBot.

Talks to the FastAPI backend over HTTP — this file has zero direct RAG/RBAC/
SQL logic; it only calls /login, /chat, /collections/{role} and displays
what comes back. All the actual intelligence lives in the FastAPI backend.
"""
import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

ROLE_AVATARS = {
    "doctor": "🩺",
    "nurse": "👩‍⚕️",
    "billing_executive": "💰",
    "technician": "🔧",
    "admin": "🛡️",
}

RETRIEVAL_BADGE = {
    "hybrid_rag": ("Hybrid RAG", "#3b82f6"),
    "sql_rag": ("SQL RAG", "#8b5cf6"),
}

REFUSAL_MARKERS = ["sufficiently relevant information", "analytical/operational data queries"]

st.set_page_config(page_title="MediBot", page_icon="🏥", layout="wide")

# Trims Streamlit's default top/bottom page padding, which is much larger
# than this compact login form actually needs.
st.markdown(
    """
    <style>
        .block-container { padding-top: 2.5rem; padding-bottom: 1rem; max-width: 480px; }
        div[data-testid="stAppViewContainer"] > .main .block-container { max-width: 480px; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "token" not in st.session_state:
    st.session_state.token = None
    st.session_state.role = None
    st.session_state.username = None
    st.session_state.chat_history = []


def is_refusal(answer: str) -> bool:
    return any(marker in answer for marker in REFUSAL_MARKERS)


def badge_html(label: str, color: str) -> str:
    return (
        f'<span style="background-color:{color}; color:white; padding:2px 10px; '
        f'border-radius:12px; font-size:0.75em; font-weight:600;">{label}</span>'
    )


def login(username: str, password: str) -> bool:
    try:
        resp = requests.post(f"{API_BASE}/login", json={"username": username, "password": password})
        if resp.status_code == 200:
            data = resp.json()
            st.session_state.token = data["token"]
            st.session_state.role = data["role"]
            st.session_state.username = username
            return True
        st.error("Invalid username or password")
        return False
    except requests.exceptions.ConnectionError:
        st.error("Can't reach the MediBot API. Is uvicorn running on port 8000?")
        return False


def get_collections(role: str) -> list[str]:
    resp = requests.get(f"{API_BASE}/collections/{role}")
    resp.raise_for_status()
    return resp.json()["collections"]


def send_chat(question: str) -> dict:
    resp = requests.post(
        f"{API_BASE}/chat",
        headers={"authorization": f"Bearer {st.session_state.token}"},
        json={"question": question},
    )
    resp.raise_for_status()
    return resp.json()


def logout():
    st.session_state.token = None
    st.session_state.role = None
    st.session_state.username = None
    st.session_state.chat_history = []


if not st.session_state.token:
    st.markdown(
        "<h2 style='text-align:center; margin-bottom:0;'>🏥 MediBot</h2>"
        "<p style='text-align:center; color:gray; margin-top:2px; font-size:0.9em;'>"
        "MediAssist Health Network</p>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        demo_accounts = {
            "doctor (dr.mehta)": ("dr.mehta", "doctor"),
            "nurse (nurse.priya)": ("nurse.priya", "nurse"),
            "billing_executive (billing.ravi)": ("billing.ravi", "billing_executive"),
            "technician (tech.anand)": ("tech.anand", "technician"),
            "admin (admin.sys)": ("admin.sys", "admin"),
        }
        choice = st.selectbox("Demo account", ["-- choose --"] + list(demo_accounts.keys()), label_visibility="collapsed", placeholder="Choose a demo account")
        default_user, default_pass = demo_accounts.get(choice, ("", ""))

        with st.form("login_form"):
            username = st.text_input("Username", value=default_user)
            password = st.text_input("Password", value=default_pass, type="password")
            if st.form_submit_button("Log in", width="stretch") and login(username, password):
                st.rerun()

else:
    role = st.session_state.role
    avatar = ROLE_AVATARS.get(role, "👤")

    with st.sidebar:
        st.markdown(f"## {avatar} {st.session_state.username}")
        st.markdown(f"**Role:** `{role}`")
        try:
            collections = get_collections(role)
            st.markdown("**Accessible collections:**")
            chips = " ".join(badge_html(c, "#334155") for c in collections)
            st.markdown(chips, unsafe_allow_html=True)
        except Exception:
            st.warning("Could not load accessible collections")
        st.divider()
        if st.button("Log out", width="stretch"):
            logout()
            st.rerun()

    st.title(f"{avatar} MediBot")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role_type"], avatar=avatar if msg["role_type"] == "user" else "🏥"):
            content = msg["content"]
            if msg["role_type"] == "assistant" and is_refusal(content):
                st.warning(content)
            else:
                st.markdown(content)

            if msg["role_type"] == "assistant":
                label, color = RETRIEVAL_BADGE.get(msg["retrieval_type"], (msg["retrieval_type"], "#64748b"))
                st.markdown(badge_html(label, color), unsafe_allow_html=True)
                if msg["sources"]:
                    with st.expander(f"Sources ({len(msg['sources'])})"):
                        for s in msg["sources"]:
                            with st.container(border=True):
                                st.markdown(f"📄 **{s['source_document']}**")
                                st.caption(f"{s['section_title']} · _{s['collection']}_")

    question = st.chat_input("Ask MediBot a question...")
    if question:
        st.session_state.chat_history.append({"role_type": "user", "content": question})
        with st.spinner("Thinking..."):
            try:
                result = send_chat(question)
                st.session_state.chat_history.append({
                    "role_type": "assistant", "content": result["answer"],
                    "retrieval_type": result["retrieval_type"], "sources": result["sources"],
                })
            except Exception as e:
                st.session_state.chat_history.append({
                    "role_type": "assistant", "content": f"Error: {e}",
                    "retrieval_type": "error", "sources": [],
                })
        st.rerun()