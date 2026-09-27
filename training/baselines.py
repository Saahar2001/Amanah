from __future__ import annotations

import re
from collections import Counter
from math import sqrt


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.lower())


def lexical_token_similarity(a: str, b: str) -> float:
    ca, cb = Counter(_tokens(a)), Counter(_tokens(b))
    if not ca or not cb:
        return 0.0
    keys=set(ca)|set(cb)
    dot=sum(ca[k]*cb[k] for k in keys)
    na=sqrt(sum(v*v for v in ca.values())); nb=sqrt(sum(v*v for v in cb.values()))
    return float(dot/(na*nb)) if na and nb else 0.0


def cosine_embedding_similarity(a: str, b: str, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2") -> dict:
    try:
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.util import cos_sim
    except ImportError:
        return {"status":"unavailable","reason":"sentence-transformers not installed"}
    model=SentenceTransformer(model_name)
    emb=model.encode([a,b],convert_to_tensor=True)
    return {"status":"ok","score":float(cos_sim(emb[0],emb[1]).item())}
