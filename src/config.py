# src/config.py
"""
Configuration loader for environment variables (.env).
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Warnings for missing keys
if not QDRANT_URL:
    print("[WARN] QDRANT_URL not found in .env (required for Qdrant Cloud).")
if not QDRANT_API_KEY:
    print("[WARN] QDRANT_API_KEY not found in .env (required for Qdrant Cloud).")
if not GROQ_API_KEY:
    print("[WARN] GROQ_API_KEY not found in .env (Groq generation will not work).")
