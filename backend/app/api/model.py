from pydantic import BaseModel

class ChatRequest(BaseModel):
    """
    This is the data we expect from the frontend.
    """
    query: str
    llm_choice: str  # e.g., "groq", "openai"
    api_key: str | None = None # Optional API key

class ChatResponse(BaseModel):
    """
    This is the data we will send back to the frontend.
    """
    answer: str
    # We can add sources later if needed
    # sources: list[str]