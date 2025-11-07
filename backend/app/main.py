import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys

# --- Relative Imports ---
from .api.model import ChatRequest, ChatResponse

# --- Our RAG Engine ---
# We now import the *master function*
from .services.rag_pipeline import run_rag_pipeline

# --- Our Ingestion Function ---
from .services.ingestion import run_ingestion, CORPUS_PATH

# Initialize the FastAPI app
app = FastAPI(
    title="Precision-RAG API",
    description="API for the Precision-RAG hybrid search and re-ranking system."
)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# --- LLM Helper Function ---
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

def get_llm(choice: str, api_key: str | None):
    """
    Selects and initializes the correct LLM based on user's choice.
    """
    if choice == "groq":
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise HTTPException(status_code=400, detail="GROQ_API_KEY not set.")
        return ChatGroq(model_name="llama-3.1-8b-instant", temperature=0, groq_api_key=key)
    
    elif choice == "openai":
        if not api_key:
            raise HTTPException(status_code=400, detail="OpenAI API key is required.")
        return ChatOpenAI(model_name="gpt-4o-mini", temperature=0, api_key=api_key)
    
    else:
        raise HTTPException(status_code=400, detail="Invalid LLM choice.")

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Precision-RAG API is running!"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Receives a file from the frontend and saves it to the /corpus directory.
    """
    file_path = os.path.join(CORPUS_PATH, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"message": f"File '{file.filename}' uploaded successfully."}
    except Exception as e:
        print(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")


@app.post("/ingest")
async def ingest_documents():
    """
    Runs the ingestion script to re-index all documents in the /corpus folder.
    """
    try:
        print("Starting /ingest endpoint...")
        # Run the ingestion function
        run_ingestion()
        
        print("Ingestion complete. Triggering server reload...")
        
        # This forces uvicorn --reload to see a change and restart
        with open(__file__, "a") as f:
            f.write(" ")
            
        return {"message": "Ingestion complete. Server is restarting with new data."}

    except Exception as e:
        print(f"Error during ingestion: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@app.post("/chat")
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    The main chat endpoint.
    """
    try:
        # 1. Select the LLM
        llm = get_llm(request.llm_choice, request.api_key)
        
        # 2. Run the full pipeline (this now loads models and releases them)
        answer = run_rag_pipeline(request.query, llm)
        
        return ChatResponse(answer=answer)

    except Exception as e:
        print(f"An error occurred in /chat: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

# This allows running the file directly
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)    