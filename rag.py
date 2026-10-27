"""Retrieval: split the policy into chunks, embed them with Gemini, search by cosine similarity.
Everything lives in memory - one policy per run, so no vector database is needed."""
import os
from functools import lru_cache

import numpy as np
from google import genai
from google.genai import types

EMBED_MODEL = "gemini-embedding-001"
CHUNK_WORDS = 300     # free tier allows 100 embedded texts/min, so keep the chunk count low
OVERLAP_WORDS = 50    # so a sentence on a boundary isn't lost
TOP_K = 3
MIN_SCORE = 0.60      # below this, nothing relevant was retrieved -> "No evidence found"


@lru_cache
def get_client():
    # Timeout + capped retries so a rate limit surfaces as an error instead of an endless spinner.
    return genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
        http_options=types.HttpOptions(
            timeout=60_000,
            retry_options=types.HttpRetryOptions(attempts=3, initial_delay=5, max_delay=20),
        ),
    )


def chunk_text(text):
    words = text.split()
    step = CHUNK_WORDS - OVERLAP_WORDS
    return [" ".join(words[i:i + CHUNK_WORDS]) for i in range(0, max(len(words) - OVERLAP_WORDS, 1), step)]


def embed(texts, task_type):
    vectors = []
    for i in range(0, len(texts), 100):  # API accepts up to 100 texts per call
        res = get_client().models.embed_content(
            model=EMBED_MODEL,
            contents=texts[i:i + 100],
            config=types.EmbedContentConfig(task_type=task_type),
        )
        vectors += [e.values for e in res.embeddings]
    v = np.array(vectors)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class PolicyIndex:
    def __init__(self, text):
        self.chunks = chunk_text(text)
        self.vectors = embed(self.chunks, "RETRIEVAL_DOCUMENT")

    def search(self, queries, k=TOP_K):
        """Embeds all queries in one call. Returns, per query, [(chunk, score), ...] best first."""
        results = []
        for q in embed(queries, "RETRIEVAL_QUERY"):
            scores = self.vectors @ q
            top = np.argsort(scores)[::-1][:k]
            results.append([(self.chunks[i], float(scores[i])) for i in top])
        return results
