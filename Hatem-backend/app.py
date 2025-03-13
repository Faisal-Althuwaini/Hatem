import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from rag_api_compatible_v3_2_openai import ArabicRAGSystem  # Import your RAG system

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

def sanitize_text(text: str):
    """
    Cleans the response and replaces newlines with spaces to ensure a clean output.
    """
    # text = re.sub(r"[^\u0600-\u06FF\s.,!?؛،]", "", text)  # Remove non-Arabic characters
    text = text.replace("\n", "\n")  # Replace newlines with spaces
    return text.strip()

@app.post("/query")
async def query_rag(request: QueryRequest):
    """
    API endpoint that returns only the plain response without formatting.
    """
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG System not initialized")

    try:
        # 🔹 Generate response from RAG
        response_text = rag_system.generate_response(request.query)

        # 🔹 Ensure valid JSON response
        cleaned_response = sanitize_text(response_text) if response_text else "لم يتم العثور على معلومات."

        return {"full_response": cleaned_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------
#  Running FastAPI
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
