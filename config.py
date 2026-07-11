"""
Centralised config loader.
Reads from st.secrets (Streamlit Cloud) first, then falls back to os.getenv (local .env).
"""

import os
from dotenv import load_dotenv

load_dotenv()

def _get(key: str) -> str | None:
    """Try st.secrets first, then env vars."""
    try:
        import streamlit as st
        value = st.secrets.get(key)
        if value:
            return value
    except Exception:
        pass
    return os.getenv(key)


QDRANT_URL     = _get("QDRANT_URL")
QDRANT_API_KEY = _get("QDRANT_API_KEY")
GROQ_API_KEY   = _get("GROQ_API_KEY")
