import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from rag_api_compatible import ArabicRAGSystem  # Import your RAG system

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Allow requests from this origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

rag_system = ArabicRAGSystem()

class QueryRequest(BaseModel):
    query: str

def format_for_chatbot(text: str):
    """
    Formats the response for chatbot-friendly display by removing newlines.
    """
    # Remove \n and extra spaces
    cleaned_text = re.sub(r"\s*\n\s*", " ", text).strip()

    # Split into sentences for chatbot-style responses
    sentences = re.split(r"\.\s*", cleaned_text)
    sentences = [s.strip() + "." for s in sentences if s.strip()]  # Ensure proper punctuation

    # Convert each sentence into a separate chatbot message
    messages = [{"role": "bot", "content": s} for s in sentences]

    return {
        "full_response": cleaned_text,  # Full cleaned response
        "messages": messages  # List of chatbot messages
    }

@app.post("/query")
async def query_rag(request: QueryRequest):
    """
    API endpoint for chatbot-style responses.
    """
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG System not initialized")

    try:
        response_text = rag_system.generate_response(request.query)
        structured_response = format_for_chatbot(response_text)

        return {
            "query": request.query,
            "response": structured_response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")
