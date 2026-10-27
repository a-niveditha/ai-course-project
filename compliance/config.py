"""All tunable settings in one place."""
import os

# Models
EMBED_MODEL = "gemini-embedding-001"
# Flash models allow only 20 requests/day on the free tier; flash-lite allows far more.
LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

# Chunking
MAX_WORDS = 350       # long sections are split into pieces of this size
OVERLAP_WORDS = 50    # so a sentence on a piece boundary isn't lost

# Retrieval
TOP_K = 3             # passages handed to the model up front
MIN_SCORE = 0.60      # below this, nothing relevant was retrieved -> "No evidence found"

# Agent
MAX_SEARCHES = 3      # tool calls allowed per requirement before it must give a verdict

STATUSES = ["Addressed", "Partially addressed", "Not addressed"]
