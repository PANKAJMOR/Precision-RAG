import os
import pickle

# --- 0. LOAD ENVIRONMENT VARIABLES FIRST ---
from dotenv import load_dotenv
base_dir_for_env = os.path.dirname(os.path.abspath(__file__))
project_root_for_env = os.path.abspath(os.path.join(base_dir_for_env, "..", "..", ".."))
load_dotenv(dotenv_path=os.path.join(project_root_for_env, "backend", ".env"))
# --- End of new code ---

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder 

# --- 1. Define Paths (These are still needed) ---
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, "..", "..", ".."))

VECTORSTORE_PATH = os.path.join(project_root, "vectorstore")
BM25_INDEX_PATH = os.path.join(project_root, "bm25_index", "bm25_retriever.pkl") 

# --- 2. Helper Functions ---
# We make them "private" (with _) as they will only be called by our master function.
def _hybrid_search(query, chroma_retriever, bm25_retriever):
    print(f"Performing hybrid search for: '{query}'")
    chroma_docs = chroma_retriever.invoke(query)
    bm25_docs = bm25_retriever.invoke(query)
    
    all_docs = chroma_docs + bm25_docs
    unique_docs = {}
    for doc in all_docs:
        unique_docs[doc.page_content] = doc
        
    unique_list = list(unique_docs.values())
    print(f"Found {len(unique_list)} unique documents after hybrid search.")
    return unique_list

def _rerank_documents(query_and_docs: dict, cross_encoder_model):
    query = query_and_docs["query"]
    documents = query_and_docs["documents"]
    
    if not documents:
        return []
        
    pairs = [[query, doc.page_content] for doc in documents]
    print(f"Reranking {len(pairs)} documents...")
    scores = cross_encoder_model.predict(pairs)
    
    scored_docs = list(zip(scores, documents))
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    print(f"Returning top 3 reranked documents.")
    return [doc for score, doc in scored_docs[:3]]

def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- 3. Define Prompt (can be global) ---
template = """
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
You must answer *only* based on the context provided.

Context:
{context}

Question:
{question}

Answer:
"""
RAG_PROMPT = PromptTemplate.from_template(template)

# --- 4. NEW: The "Master" RAG Function ---
def run_rag_pipeline(query: str, llm):
    """
    Loads all models, builds the chain, runs it, and returns the answer.
    All file locks are released when this function exits.
    """
    print("--- RAG Pipeline Started (Loading models)... ---")

    # --- A. Load Models ---
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'} 
    )
    
    print("Initializing cross-encoder for re-ranking...")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # --- B. Load Indexes from Disk ---
    print(f"Loading vectorstore from: {VECTORSTORE_PATH}")
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_PATH,
        embedding_function=embeddings
    )
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    print(f"Loading BM25 index from: {BM25_INDEX_PATH}")
    with open(BM25_INDEX_PATH, 'rb') as f:
        bm25_retriever = pickle.load(f)
    bm25_retriever.k = 5
    
    print("Models and indexes loaded.")

    # --- C. Create the Chain ---
    
    # Create lambda functions to pass the loaded retrievers/models
    hybrid_search_lambda = lambda q: _hybrid_search(q, chroma_retriever, bm25_retriever)
    rerank_lambda = lambda x: _rerank_documents(x, cross_encoder)
    
    chain = (
        {
            "documents": RunnableLambda(hybrid_search_lambda), 
            "query": RunnablePassthrough()
        }
        | RunnableLambda(rerank_lambda)
        | RunnableLambda(_format_docs)
    )

    rag_chain = (
        {"context": chain, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    
    print("RAG chain built. Invoking...")

    # --- D. Run the Chain ---
    answer = rag_chain.invoke(query)
    
    # --- E. Cleanup ---
    # All local objects (vectorstore, retrievers, models)
    # go out of scope here. Python's garbage collector will
    # automatically release all file locks.
    del vectorstore, chroma_retriever, bm25_retriever, embeddings, cross_encoder
    print("--- RAG Pipeline Finished. Resources released. ---")
    
    return answer   

# --- 10. (Optional) Test the pipeline ---
if __name__ == "__main__":
    print("\n--- Testing RAG Pipeline ---")
    
    try:
        from langchain_groq import ChatGroq
        
        if not os.environ.get("GROQ_API_KEY"):
            print("GROQ_API_KEY not set in .env file. Skipping live LLM test.")
        elif not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
             print("HUGGING_FACE_HUB_TOKEN not set in .env file. Skipping live LLM test.")
        else:
            print("Initializing Groq LLM (Llama3-8B) for testing...")
            test_llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
                     
            test_query = "What is the main topic of the documents?" # <-- CHANGE THIS
            print(f"\nTest Query: {test_query}")
            response = run_rag_pipeline(test_query,test_llm)
            print(response)

    except ImportError:
        print("langchain_groq or python-dotenv not installed. Skipping live LLM test.")
        print("Please run: pip install langchain-groq python-dotenv")
    except Exception as e:
        print(f"An error occurred during testing: {e}")