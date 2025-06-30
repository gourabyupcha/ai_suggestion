from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from keybert import KeyBERT
from typing import List
import uvicorn, random

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows requests from these origins
    allow_credentials=True,  # Allow cookies and headers like Authorization
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize KeyBERT model with MiniLM (CPU-friendly)
kw_model = KeyBERT(model='paraphrase-MiniLM-L6-v2')

# Request schema
class TagRequest(BaseModel):
    title: str
    description: str
    top_n: int = 5

# Response schema
class TagResponse(BaseModel):
    tags: List[str]

@app.post("/suggest-tags", response_model=TagResponse)
def suggest_tags(data: TagRequest):
    try:
        input_text = f"{data.title}\n{data.description}"
        keywords = kw_model.extract_keywords(input_text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=data.top_n * 2)

        # Shuffle and take diverse sample
        random.shuffle(keywords)
        selected = keywords[:data.top_n]

        tags = [kw[0] for kw in selected]
        return TagResponse(tags=tags)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)