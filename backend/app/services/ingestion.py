import os
import pickle
import shutil # Import this to delete directories
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
import sys 

# --- 1. Define Paths ---
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "..", "..", ".."))
    
    CORPUS_PATH = os.path.join(project_root, "corpus")
    VECTORSTORE_PATH = os.path.join(project_root, "vectorstore")
    BM25_INDEX_PATH = os.path.join(project_root, "bm25_index", "bm25_retriever.pkl")
    
    # Ensure output directories exist
    os.makedirs(CORPUS_PATH, exist_ok=True) # Makes sure /corpus exists
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(BM25_INDEX_PATH), exist_ok=True)

except Exception as e:
    print(f"Error setting up paths: {e}")
    sys.exit(1)


# --- 2. NEW: Function to clear old indexes ---
def clear_indexes():
    """
    Deletes the old vectorstore and BM25 index to start fresh.
    """
    print("Clearing old indexes...")
    if os.path.exists(VECTORSTORE_PATH):
        shutil.rmtree(VECTORSTORE_PATH)
        print(f"Removed old vectorstore at {VECTORSTORE_PATH}")
    
    if os.path.exists(BM25_INDEX_PATH):
        os.remove(BM25_INDEX_PATH)
        print(f"Removed old BM25 index at {BM25_INDEX_PATH}")
        
    # Re-create the directories
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(BM25_INDEX_PATH), exist_ok=True)


# --- 3. Simplified Document Loader (PDF and TXT only) ---
def load_documents(corpus_path):
    """
    Loads all .pdf and .txt documents from the corpus directory.
    """
    print(f"Loading documents from: {corpus_path}")
    all_documents = []

    if not os.path.exists(corpus_path):
        print(f"Error: Corpus directory not found at {corpus_path}")
        return []

    for root, dirs, files in os.walk(corpus_path):
        for file in files:
            # Skip hidden files
            if file.startswith('.'):
                continue
                
            file_path = os.path.join(root, file)
            
            try:
                if file.endswith(".pdf"):
                    print(f"Loading PDF: {file}")
                    loader = PyPDFLoader(file_path)
                    all_documents.extend(loader.load())
                
                elif file.endswith(".txt"):
                    print(f"Loading TEXT: {file}")
                    loader = TextLoader(file_path, encoding='utf-8')
                    all_documents.extend(loader.load())
                
                else:
                    pass
            except Exception as e:
                print(f"Failed to load {file}. Error: {e}")
                pass 

    print(f"Loaded {len(all_documents)} documents in total.")
    return all_documents

# --- 4. Split Documents ---
def split_documents(documents):
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")
    return chunks

# --- 5. Create Vectorstore (ChromaDB) ---
def create_vectorstore(chunks, embedding_model):
    print(f"Creating vectorstore at: {VECTORSTORE_PATH}")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model, 
        persist_directory=VECTORSTORE_PATH
    )
    print("Vectorstore created successfully.")
    return vectorstore

# --- 6. Create BM25 Index ---
def create_bm25_index(chunks):
    print("Creating BM25 index...")
    try:
        chunk_texts = [doc.page_content for doc in chunks]
        chunk_metadatas = [doc.metadata for doc in chunks]
        bm25_retriever = BM25Retriever.from_texts(
            texts=chunk_texts,
            metadatas=chunk_metadatas
        )
        print(f"Saving BM25 index to: {BM25_INDEX_PATH}")
        with open(BM25_INDEX_PATH, 'wb') as f:
            pickle.dump(bm25_retriever, f)
        print("BM25 index created and saved successfully.")
        return bm25_retriever
    except Exception as e:
        print(f"Error creating BM25 index: {e}")
        return None

# --- Main Execution (RENAMED from main) ---
def run_ingestion():
    """
    Main function to run the complete ingestion pipeline.
    This is what our API will call.
    """
    print("--- Starting Ingestion Pipeline ---")
    
    # 1. NEW: Clear old indexes first
    clear_indexes()
    
    # 2. Load
    documents = load_documents(CORPUS_PATH)
    if not documents:
        print("No .pdf or .txt documents found in /corpus. Exiting.")
        return

    # 3. Split
    chunks = split_documents(documents)
    if not chunks:
        print("No chunks were created. Exiting.")
        return
        
    # 4. Initialize Embedding Model (ONCE)
    print("Initializing embedding model (all-MiniLM-L6-v2)...")
    try:
        model_name = "all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'}
        )
        print("Embedding model initialized.")
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        return

    # 5. Create and persist Chroma vectorstore
    create_vectorstore(chunks, embeddings)
    
    # 6. Create and save BM25 index
    create_bm25_index(chunks)
    
    print("--- Ingestion Pipeline Finished Successfully ---")

# This part calls the renamed function
if __name__ == "__main__":
    run_ingestion()