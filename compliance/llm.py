"""Gemini API access: the shared client and the embedding call."""
import os
from functools import lru_cache

import numpy as np
from google import genai
from google.genai import types

from compliance.config import EMBED_MODEL


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


def embed(texts, task_type):
    """Returns one unit-length vector per text, so a dot product is cosine similarity."""
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
