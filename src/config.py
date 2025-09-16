
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env at repo root

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Safety checks:
if not QDRANT_URL:
	print("Warning: QDRANT_URL not found in .env (you will need it for Qdrant Cloud).")
if not GROQ_API_KEY:
	print("Warning: GROQ_API_KEY not found in .env (you'll need it for Groq inference).")
