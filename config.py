"""Configuration for Resume RAG Application"""
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
MAIN_LLM_MODEL = "gpt-4o-mini"

PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "resumes"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 20