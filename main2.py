import spacy, random
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from collections import Counter
import psutil, os, uvicorn


app = FastAPI()
nlp = spacy.load("en_core_web_sm")

class TagRequest(BaseModel):
    title: str
    description: str
    top_n: int = 5

class TagResponse(BaseModel):
    tags: List[str]
    
    

@app.get("/memory-usage")
def memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)  # in MB
    return {"memory_usage_mb": round(mem, 2)}

    

@app.post("/suggest-tags", response_model=TagResponse)
def suggest_tags(data: TagRequest):
    text = f"{data.title}\n{data.description}"
    doc = nlp(text)

    # Get phrases
    noun_chunks = [chunk.text.lower().strip() for chunk in doc.noun_chunks]
    entities = [ent.text.lower().strip() for ent in doc.ents if ent.label_ not in ("CARDINAL", "DATE")]
    keywords = [
        token.text.lower()
        for token in doc
        if token.pos_ in ("NOUN", "PROPN", "ADJ")
        and not token.is_stop and token.is_alpha and len(token.text) > 3
    ]

    # Combine and deduplicate
    combined = noun_chunks + entities + keywords
    freq = Counter(combined)

    # Soft sampling: pick more frequent terms with higher probability
    weighted_candidates = [(tag, freq[tag]) for tag in freq]
    population = [tag for tag, score in weighted_candidates for _ in range(score)]

    sampled_tags = list(set(random.sample(population, min(len(population), data.top_n * 2))))
    random.shuffle(sampled_tags)

    return TagResponse(tags=sampled_tags[:data.top_n])


if __name__ == "__main__":
    uvicorn.run("main2:app", host="0.0.0.0", port=8000, reload=True)