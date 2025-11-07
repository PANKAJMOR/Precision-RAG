
# Precision-RAG: A Dynamic RAG System 🚀

Precision-RAG is a full-stack, dynamic Retrieval-Augmented Generation (RAG) application. It allows users to upload their own documents, which are then indexed using a sophisticated hybrid search pipeline. The system combines keyword-based (BM25) and semantic (ChromaDB) search, followed by a cross-encoder re-ranking step to provide highly accurate, fact-grounded answers from a selection of LLMs.

This project is built from scratch with a vanilla HTML/CSS/JS frontend and a powerful Python (FastAPI) backend.

## ✨ Key Features

  * **Dynamic Document Upload:** Users can upload new `.pdf` or `.txt` files directly through the web interface.
  * **Persistent Ingestion:** Uploaded files are saved, and the entire document library is re-indexed, providing a persistent and growing knowledge base.
  * **Hybrid Search:** Implements a "best-of-both-worlds" retrieval by combining **BM25** (for keyword matching) and **ChromaDB** (for semantic meaning).
  * **Cross-Encoder Re-ranking:** A `sentence-transformers` cross-encoder model re-ranks the retrieved results for maximum relevance, significantly reducing noise and improving answer quality.
  * **Multi-LLM Support:** Easily switch between different language models, such as the ultra-fast Groq (`llama-3.1-8b-instant`) or powerful paid models like OpenAI's (`gpt-4o-mini`).
  * **Clean API Backend:** Built with **FastAPI**, providing clear, fast, and testable API endpoints.
  * **Lightweight Frontend:** A simple, dependency-free vanilla **HTML, CSS, and JavaScript** frontend that's easy to run and understand.

## 🛠️ Tech Stack

| Area | Technology | Purpose |
| :--- | :--- | :--- |
| **Backend** | Python 3.10+ | Core application language |
| | FastAPI | High-performance web API framework |
| | LangChain | Framework for RAG pipeline orchestration |
| | ChromaDB | Vector database for semantic storage |
| | `rank_bm25` | Library for keyword (sparse) retrieval |
| | `sentence-transformers` | For embeddings and cross-encoder re-ranking |
| **Frontend** | HTML5 | Webpage structure |
| | CSS3 | Styling and layout |
| | JavaScript (ES6+) | Application logic and API communication |
| **Server** | Uvicorn | ASGI server to run FastAPI |

## 📁 Project Structure

```
Precision-RAG/
│
├── .gitignore          # --- IMPORTANT: Keeps secrets & data out of Git ---
├── README.md           # --- You are here ---
│
├── backend/
│   ├── .env            # --- CRITICAL: Holds your secret API keys ---
│   ├── .venv/          # (Ignored) Python virtual environment
│   ├── requirements.txt
│   │
│   └── app/
│       ├── __init__.py
│       ├── main.py         # --- FastAPI app, API endpoints (/upload, /ingest, /chat) ---
│       │
│       ├── api/
│       │   ├── __init__.py
│       │   └── models.py   # Pydantic request/response models
│       │
│       └── services/
│           ├── __init__.py
│           ├── ingestion.py    # --- The "Librarian": Loads, splits, and indexes docs ---
│           └── rag_pipeline.py # --- The "Expert": Runs the RAG chain (hybrid search + rerank) ---
│
├── bm25_index/
│   └── (Ignored)       # --- Generated BM25 index file ---
│
├── corpus/
│   └── (Ignored)       # --- Uploaded .pdf and .txt files are saved here ---
│
├── frontend/
│   ├── index.html      # The main (and only) HTML page
│   │
│   ├── scripts/
│   │   └── main.js     # Handles all frontend logic (upload, chat)
│   │
│   └── styles/
│       └── style.css   # All styling for the application
│
└── vectorstore/
    └── (Ignored)       # --- Generated ChromaDB vector index ---
```

-----

## 🚀 How to Run the Application

Follow these steps to get your Precision-RAG system running locally.

### Prerequisites

  * **Git:** To clone the repository.
  * **Python 3.10+:** To run the backend.
  * **VS Code:** With the **[Live Server](https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer)** extension (the easiest way to run the frontend).

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/Precision-RAG.git
cd Precision-RAG
```

### Step 2: Set Up the Backend

1.  Navigate to the `backend` directory:

    ```bash
    cd backend
    ```

2.  Create and activate a Python virtual environment:

    ```bash
    # Windows
    python -m venv .venv
    .venv\Scripts\activate
    ```

3.  Install all required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

### Step 3: Set Up Environment Keys (Critical)

1.  In the `backend/` folder, create a new file named `.env`.

2.  Paste the following content into it and add your secret keys. You can get your keys from the respective websites.

    ```env
    # Get from https://console.groq.com/
    GROQ_API_KEY=your_groq_api_key_here

    # Get from https://huggingface.co/settings/tokens
    # Required for downloading the re-ranking model
    HUGGING_FACE_HUB_TOKEN=your_hugging_face_token_here
    ```

### Step 4: Run the Backend Server

With your virtual environment still active, run the FastAPI server from the `backend/` directory:

```bash
uvicorn app.main:app --reload --port 8000
```

Your terminal should show that the server is running on `http://127.0.0.1:8000`. Leave this terminal running.

### Step 5: Run the Frontend

1.  Open the entire `Precision-RAG` project folder in VS Code.
2.  In the VS Code file explorer, navigate to `frontend/`.
3.  **Right-click on the `index.html` file.**
4.  Select **"Open with Live Server"**.

Your web browser will automatically open, and you will see the chat application, ready to go\!

## Usage

1.  **Upload a File:** Click "Choose File" and select a `.pdf` or `.txt` file from your computer.
2.  **Process:** Click the "Upload & Process" button. You will see a status message. The backend will:
    a. Save the file to the `/corpus` folder.
    b. Run the ingestion script, deleting all old indexes.
    c. Create new, fresh indexes for all files in the `/corpus` folder.
    d. Restart the server (this is automatic).
3.  **Chat:** Once the server restarts (this takes a few seconds), you can ask questions about *any* of the documents you've uploaded.