"""Configuration"""
import os
from dotenv import load_dotenv

load_dotenv()

# Get API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_API_KEY")

# If not in env, try Streamlit secrets (for Streamlit Cloud)
if not OPENAI_API_KEY:
    try:
        import streamlit as st
        OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
    except:
        pass

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "resumes"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
TOP_K = 20