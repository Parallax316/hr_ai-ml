# main.py
import streamlit as st
import os
import io
import uuid
import tempfile
import logging
import json
import sys # Import sys for explicit flushing
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from threading import Thread
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Callable # For type hinting

# --- LLM & RAG Imports ---
from llama_cpp import Llama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever
# --- Import for OpenRouter/OpenAI compatible models ---
try:
    from langchain_openai import ChatOpenAI
    LANGCHAIN_OPENAI_INSTALLED = True
except ImportError:
    logging.warning("langchain-openai not installed. OpenRouter evaluation will not work. Run: pip install langchain-openai")
    LANGCHAIN_OPENAI_INSTALLED = False
    ChatOpenAI = None # Define as None if import fails


# --- Re-ranking Imports ---
try:
    from sentence_transformers.cross_encoder import CrossEncoder
    SENTENCE_TRANSFORMERS_INSTALLED = True
except ImportError:
    logging.warning("sentence-transformers not installed. Re-ranking will be skipped. Run: pip install sentence-transformers")
    SENTENCE_TRANSFORMERS_INSTALLED = False
    CrossEncoder = None

# --- Ragas Evaluation Imports ---
try:
    import ragas # Import the main library
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
        answer_correctness,
    )
    from datasets import Dataset
    RAGAS_INSTALLED = True
except ImportError as e:
    logging.warning(f"ragas or datasets library not installed or core import failed ({e}). RAG evaluation endpoint will not work. Run: pip install ragas datasets")
    RAGAS_INSTALLED = False
    ragas = None
    Dataset = None
    evaluate = None
    faithfulness, answer_relevancy, context_recall, context_precision, answer_correctness = [None]*5


# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
MODEL_PATH = "D:\\hr_ai\\hr-finetune-merged.gguf" # Local model for primary RAG
N_CTX = 8192
N_GPU_LAYERS = 35
EMBEDDING_MODEL_NAME = "nomic-embed-text" # Local embeddings
OLLAMA_BASE_URL = "http://localhost:11434"
CHUNK_SIZE = 350
CHUNK_OVERLAP = 75
ANALYSIS_COLLECTION_NAME = "hr_analysis_results_v1"
POLICY_COLLECTION_NAME = "hr_policy_docs_rerank_v1"
UPLOAD_DIR = "./uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- RAG Configuration ---
INITIAL_RETRIEVER_K = 10
RERANK_TOP_N = 2
CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# --- OpenRouter Configuration (for Ragas Eval) ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY") # Get key from environment
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1" # Default OpenRouter URL
RAGAS_EVAL_LLM_MODEL = "openai/gpt-4o-mini" # OpenRouter model identifier


# --- Start: Analysis & Dashboard Imports (Strict) ---

# Initialize function variables globally
run_attrition_analysis: Callable | None = None
run_clustering_analysis: Callable | None = None
generate_basic_dashboard: Callable | None = None
generate_extended_dashboard: Callable | None = None

# Attempt to import the actual functions
try:
    from attrition_analysis import run_attrition_analysis as real_attrition
    run_attrition_analysis = real_attrition # Assign if successful
    logger.info("Successfully imported run_attrition_analysis.")

    from clustering_analysis import run_clustering_analysis as real_clustering
    run_clustering_analysis = real_clustering # Assign if successful
    logger.info("Successfully imported run_clustering_analysis.")

    from hr_dashboard import generate_basic_dashboard as real_basic_dash
    generate_basic_dashboard = real_basic_dash # Assign if successful
    logger.info("Successfully imported generate_basic_dashboard.")

    from hr_dashboard_extended import generate_extended_dashboard as real_extended_dash
    generate_extended_dashboard = real_extended_dash # Assign if successful
    logger.info("Successfully imported generate_extended_dashboard.")

except ImportError as e:
    logger.error(f"CRITICAL: Failed to import one or more analysis/dashboard functions: {e}. "
                 f"The /analyze and /dashboard endpoints will likely fail. "
                 f"Ensure attrition_analysis.py, clustering_analysis.py, hr_dashboard.py, "
                 f"and hr_dashboard_extended.py exist and contain the required functions.")
except Exception as e:
    logger.error(f"CRITICAL: An unexpected error occurred during analysis/dashboard function imports: {e}. "
                 f"The /analyze and /dashboard endpoints may fail.", exc_info=True)

# --- End: Analysis & Dashboard Imports ---


# --- Global Resources ---
logger.info("Loading LOCAL LLM model for RAG...")
llm_model: Llama | None = None # This is your primary local LLM for answering
try:
    llm_model = Llama(model_path=MODEL_PATH, n_ctx=N_CTX, n_gpu_layers=N_GPU_LAYERS, verbose=False)
    logger.info("Local LLM model loaded successfully.")
except Exception as e:
    logger.error(f"Fatal error loading local LLM model: {e}", exc_info=True)

logger.info("Initializing LOCAL embeddings...")
embeddings: OllamaEmbeddings | None = None # Local embeddings for RAG & Ragas
try:
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME, base_url=OLLAMA_BASE_URL)
    embeddings.embed_query("test connectivity")
    logger.info("Local embeddings initialized successfully.")
except Exception as e:
    logger.error(f"Fatal error initializing local embeddings: {e}", exc_info=True)

# --- Initialize Cross-Encoder Model ---
cross_encoder = None
if SENTENCE_TRANSFORMERS_INSTALLED and CrossEncoder:
    logger.info(f"Initializing CrossEncoder model: {CROSS_ENCODER_MODEL}...")
    try:
        cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        logger.info("CrossEncoder initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing CrossEncoder: {e}. Re-ranking will be skipped.", exc_info=True)
else:
     logger.warning("Re-ranking skipped as sentence-transformers is not installed.")

# --- REMOVED Global Ragas Configuration Block ---


# --- In-Memory State Management ---
app_state = { "policy_retriever": None, "analysis_retriever": None, "policy_file_info": None, "data_file_info": None }

# --- FastAPI App ---
app = FastAPI(title="HR AI Assistant Backend")

# --- CORS Middleware ---
origins = ["http://localhost", "http://localhost:3000", "http://localhost:5173"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Pydantic Models ---
class ChatMessage(BaseModel): role: str; content: str
class ChatRequest(BaseModel): prompt: str

# --- Helper Functions ---
def save_uploaded_file(upload_file: UploadFile) -> str:
    # (Implementation as before)
    try:
        _, ext = os.path.splitext(upload_file.filename)
        temp_file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{ext}")
        with open(temp_file_path, "wb") as buffer: content = upload_file.file.read(); buffer.write(content)
        logger.info(f"Saved '{upload_file.filename}' to '{temp_file_path}'")
        return temp_file_path
    except Exception as e: logger.error(f"Save error: {e}", exc_info=True); raise HTTPException(status_code=500, detail="Save error")
    finally:
        if hasattr(upload_file, 'file') and hasattr(upload_file.file, 'close'): upload_file.file.close()

def rerank_documents(query: str, retrieved_docs: list[Document], top_n: int) -> list[Document]:
    # (Implementation as before)
    global cross_encoder
    if not retrieved_docs or not cross_encoder: return retrieved_docs[:top_n]
    logger.info(f"Re-ranking {len(retrieved_docs)} docs for query '{query[:50]}...'")
    try:
        pairs = [[query, doc.page_content] for doc in retrieved_docs]
        scores = cross_encoder.predict(pairs, show_progress_bar=False)
        scored_docs = sorted(zip(scores, retrieved_docs), key=lambda x: x[0], reverse=True)
        reranked_docs = [doc for score, doc in scored_docs[:top_n]]
        logger.info(f"Selected top {len(reranked_docs)} docs after re-ranking.")
        return reranked_docs
    except Exception as e: logger.error(f"Re-ranking error: {e}", exc_info=True); return retrieved_docs[:top_n]

def process_and_store_pdf(file_path: str, collection_name: str):
    # (Implementation as before - uses INITIAL_RETRIEVER_K)
    global app_state
    logger.info(f"Processing PDF: {file_path} (Chunk: {CHUNK_SIZE}/{CHUNK_OVERLAP}, Initial k: {INITIAL_RETRIEVER_K})")
    if not embeddings: logger.error("Embeddings unavailable."); return False
    try:
        loader = PyPDFLoader(file_path); documents = loader.load()
        if not documents: logger.warning(f"No docs in PDF: {file_path}"); return False
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        docs = text_splitter.split_documents(documents)
        if not docs: logger.warning(f"No chunks: {file_path}"); return False
        logger.info(f"Creating/updating store '{collection_name}' ({len(docs)} chunks)")
        safe_collection_name = "".join(c for c in collection_name if c.isalnum() or c in ('_', '-'))
        vectorstore = Chroma.from_documents(docs, embeddings, collection_name=safe_collection_name)
        app_state["policy_retriever"] = vectorstore.as_retriever(search_kwargs={'k': INITIAL_RETRIEVER_K})
        logger.info(f"Stored policy retriever (Initial k={INITIAL_RETRIEVER_K}).")
        return True
    except Exception as e: logger.error(f"PDF processing error: {e}", exc_info=True); app_state["policy_retriever"] = None; return False
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as remove_e:
                logger.warning(f"Could not remove temp PDF {file_path}: {remove_e}")


def store_analysis_results(analysis_docs: list[Document], collection_name: str):
    # (Implementation as before - uses INITIAL_RETRIEVER_K)
    global app_state
    if not embeddings: logger.error("Embeddings unavailable."); app_state["analysis_retriever"] = None; return False
    if not analysis_docs: logger.warning("No analysis docs."); app_state["analysis_retriever"] = None; return False
    logger.info(f"Storing {len(analysis_docs)} analysis results in '{collection_name}'")
    try:
        safe_collection_name = "".join(c for c in collection_name if c.isalnum() or c in ('_', '-'))
        vectorstore = Chroma.from_documents(analysis_docs, embeddings, collection_name=safe_collection_name)
        app_state["analysis_retriever"] = vectorstore.as_retriever(search_kwargs={'k': INITIAL_RETRIEVER_K})
        logger.info(f"Stored analysis retriever (Initial k={INITIAL_RETRIEVER_K}).")
        return True
    except Exception as e: logger.error(f"Analysis results storage error: {e}", exc_info=True); app_state["analysis_retriever"] = None; return False

# --- Updated stream_llm_response with Re-ranking ---
def stream_llm_response(prompt: str, retriever: BaseRetriever, system_prompt_template: str):
    # (Implementation as before - performs re-ranking, uses local llm_model for generation)
    logger.info(f"Generating RAG response for prompt: '{prompt[:75]}...'")
    global cross_encoder, llm_model # Uses local llm_model here
    if not llm_model or not retriever: logger.error(f"LLM/Retriever unavailable."); yield json.dumps({"error": "LLM or Retriever not available."}) + "\n"; return
    initial_docs = []; context = "No relevant context found."
    try: initial_docs = retriever.invoke(prompt); logger.info(f"Initial retrieval got {len(initial_docs)} docs.")
    except Exception as e: logger.error(f"Retrieval error: {e}", exc_info=True); yield json.dumps({"error": "Context retrieval failed."}) + "\n"; return
    final_docs = rerank_documents(query=prompt, retrieved_docs=initial_docs, top_n=RERANK_TOP_N)
    if final_docs: context = "\n\n---\n\n".join([f"Context Chunk {i+1}:\n{doc.page_content}" for i, doc in enumerate(final_docs)])
    else: logger.warning("No docs after re-ranking.")
    full_prompt_history = ""
    try: system_prompt = system_prompt_template.format(context=context); full_prompt_history = f"{system_prompt}<|user|>{prompt}<|end|>\n<|assistant|>"
    except Exception as e: logger.error(f"Prompt format error: {e}", exc_info=True); yield json.dumps({"error": "Prompt format error."}) + "\n"; return
    logger.info(f"--- Full Prompt Sent to LOCAL LLM (Re-ranked Context, {len(final_docs)} docs) ---")
    try:
        # Uses the main local llm_model for streaming the final answer
        stream = llm_model.create_completion(full_prompt_history, max_tokens=1024, temperature=0.5, top_p=0.9, stop=["</s>", "<|user|>", "<|end|>", "<|context|>"], stream=True)
        response_started = False
        for output in stream:
            if 'choices' in output and len(output['choices']) > 0 and 'text' in output['choices'][0]:
                chunk = output['choices'][0]['text']
                if not response_started: yield json.dumps({"type": "response_start"}) + "\n"; response_started = True
                yield chunk
        if response_started: yield json.dumps({"type": "response_end"}) + "\n"
        else: logger.warning("LLM stream empty.")
    except Exception as e: error_msg = str(e).replace('"', '\\"'); logger.error(f"Streaming error: {error_msg}", exc_info=True); yield json.dumps({"error": f"LLM streaming error: {error_msg}"}) + "\n"

# --- Function to Generate Full RAG Answer (for Evaluation) ---
def generate_rag_answer(prompt: str, retriever: BaseRetriever, system_prompt_template: str) -> Dict[str, Any]:
    # (Implementation as before - performs re-ranking, uses local llm_model for generation)
    logger.info(f"Generating non-streaming RAG answer for eval: '{prompt[:75]}...'")
    global cross_encoder, llm_model # Uses local llm_model here
    if not llm_model or not retriever: return {"answer": "[Error: LLM/Retriever unavailable]", "contexts": []}
    initial_docs = []; contexts_list = []; context = "No relevant context found."
    try: initial_docs = retriever.invoke(prompt)
    except Exception as e: logger.error(f"Eval - Retrieval error: {e}"); return {"answer": "[Error: Retrieval failed]", "contexts": []}
    final_docs = rerank_documents(query=prompt, retrieved_docs=initial_docs, top_n=RERANK_TOP_N)
    contexts_list = [doc.page_content for doc in final_docs]
    if final_docs: context = "\n\n---\n\n".join([f"Context Chunk {i+1}:\n{doc.page_content}" for i, doc in enumerate(final_docs)])
    full_prompt_history = ""
    try: system_prompt = system_prompt_template.format(context=context); full_prompt_history = f"{system_prompt}<|user|>{prompt}<|end|>\n<|assistant|>"
    except Exception as e: logger.error(f"Eval - Prompt format error: {e}"); return {"answer": "[Error: Prompt format failed]", "contexts": contexts_list}
    answer = "[Error: LLM generation failed]"
    try:
        # Uses the main local llm_model for generating the answer for evaluation
        response = llm_model.create_completion(full_prompt_history, max_tokens=512, temperature=0.3, top_p=0.8, stop=["</s>", "<|user|>", "<|end|>"], stream=False)
        if response and 'choices' in response and len(response['choices']) > 0: answer = response['choices'][0]['text'].strip()
    except Exception as e: logger.error(f"Eval - LLM generation error: {e}", exc_info=True)
    return {"answer": answer, "contexts": contexts_list, "question": prompt}

# --- API Endpoints ---
@app.get("/")
def read_root(): return {"message": "HR AI Assistant Backend is running."}

@app.post("/upload/policy")
async def upload_policy_pdf(file: UploadFile = File(...)):
    # (Implementation as before - calls process_and_store_pdf which uses k=10)
    global app_state
    logger.info(f"Received policy file upload request: {file.filename}")
    if not file.filename.lower().endswith(".pdf"): raise HTTPException(status_code=400, detail="Invalid file type.")
    logger.info("Clearing previous policy state.")
    app_state["policy_retriever"] = None
    app_state["policy_file_info"] = {"filename": file.filename, "status": "saving"}
    temp_file_path = ""
    try:
        temp_file_path = save_uploaded_file(file)
        app_state["policy_file_info"]["status"] = "processing"
        success = process_and_store_pdf(temp_file_path, POLICY_COLLECTION_NAME) # Uses k=10 now
        if success:
            app_state["policy_file_info"]["status"] = "processed"; logger.info(f"Policy PDF '{file.filename}' processed (Re-ranking setup).")
            return JSONResponse(content={ "message": "Policy PDF processed (Re-ranking setup).", "filename": file.filename, "status": "ready" })
        else:
            app_state["policy_file_info"]["status"] = "error"; raise HTTPException(status_code=500, detail="Failed to process policy PDF.")
    except Exception as e:
        logger.error(f"Error during policy upload: {e}", exc_info=True); app_state["policy_file_info"]["status"] = "error"
        if temp_file_path and os.path.exists(temp_file_path):
             try:
                 os.remove(temp_file_path)
             except Exception as rem_e: # Correct indentation and specific exception logging
                 logger.warning(f"Could not remove temp file {temp_file_path} on error: {rem_e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@app.post("/upload/data")
async def upload_data_csv(file: UploadFile = File(...)):
    # (Implementation as before - calls store_analysis_results which uses k=10)
    global app_state
    logger.info(f"Received data file upload request: {file.filename}")
    if not file.filename.lower().endswith(".csv"): raise HTTPException(status_code=400, detail="Invalid file type. CSV only.")
    if app_state is None: raise HTTPException(status_code=500, detail="App state error.")
    current_data_info = app_state.get("data_file_info")
    previous_filename = current_data_info.get("filename") if current_data_info else None
    previous_filepath = current_data_info.get("path") if current_data_info else None
    should_clear_state = (previous_filename and previous_filename != file.filename) or not previous_filename or (previous_filename == file.filename)
    if should_clear_state:
        logger.info("Clearing previous analysis state.")
        app_state["analysis_retriever"] = None
        if previous_filepath and os.path.exists(previous_filepath):
            try: os.remove(previous_filepath); logger.info(f"Removed old data file: {previous_filepath}")
            except Exception as e: logger.warning(f"Could not remove old data file {previous_filepath}: {e}")
    temp_file_path = ""
    try:
        temp_file_path = save_uploaded_file(file)
        app_state["data_file_info"] = { "filename": file.filename, "path": temp_file_path, "analysis_status": "uploaded" }
        logger.info(f"Data CSV '{file.filename}' uploaded to '{temp_file_path}'.")
        return JSONResponse(content={ "message": "Data CSV uploaded.", "filename": file.filename, "status": "uploaded" })
    except Exception as e:
        logger.error(f"Error during data upload: {e}", exc_info=True)
        if temp_file_path and os.path.exists(temp_file_path):
             try:
                 os.remove(temp_file_path)
             except Exception as rem_e: # Correct indentation and specific exception logging
                 logger.warning(f"Could not remove temp file {temp_file_path} on error: {rem_e}")
        app_state["data_file_info"] = None
        raise HTTPException(status_code=500, detail=f"Unexpected data upload error: {e}")

@app.post("/analyze")
async def run_analysis():
    global app_state
    logger.info("Received request to run analysis.")
    if app_state is None or not app_state.get("data_file_info"): raise HTTPException(status_code=400, detail="No data file uploaded.")
    data_info = app_state["data_file_info"]; data_path = data_info.get("path"); data_filename = data_info.get("filename", "Unknown file")
    if not data_path or not os.path.exists(data_path): raise HTTPException(status_code=404, detail=f"Data file '{data_filename}' not found.")

    # --- Start: Check if analysis functions were imported ---
    if run_attrition_analysis is None or run_clustering_analysis is None:
        logger.error("Analysis functions (attrition/clustering) were not imported successfully. Cannot run analysis.")
        raise HTTPException(status_code=501, detail="Analysis functions not available due to import errors.")
    # --- End: Check ---

    data_info["analysis_status"] = "running"; app_state["analysis_retriever"] = None
    logger.info(f"Starting data analysis for: {data_filename}...")
    analysis_docs = []; results_summary = {}
    try:
        # Call Analysis Functions (Now checked for existence)
        logger.info("Running attrition analysis...")
        attrition_results = run_attrition_analysis(data_path)
        results_summary['attrition'] = attrition_results.get('summary', 'Error/No Summary')
        if findings := attrition_results.get("key_findings", []):
             for finding in findings: analysis_docs.append(Document(page_content=finding, metadata={"source": "attrition_analysis"}))

        logger.info("Running clustering analysis...")
        clustering_results = run_clustering_analysis(data_path)
        results_summary['clustering'] = clustering_results.get('summary', 'Error/No Summary')
        if clusters := clustering_results.get("clusters", []):
             for i, desc in enumerate(clusters): analysis_docs.append(Document(page_content=desc, metadata={"source": "clustering_analysis", "cluster_id": i}))

        if not analysis_docs: logger.warning("No analysis results generated.")
        logger.info("Storing analysis results..."); success_storing = store_analysis_results(analysis_docs, ANALYSIS_COLLECTION_NAME) # Uses k=10
        if success_storing:
            data_info["analysis_status"] = "completed"; logger.info("Analysis completed and results stored.")
            return JSONResponse(content={ "message": "Analysis completed.", "status": "completed", "summary": results_summary })
        else: data_info["analysis_status"] = "error_storing"; raise HTTPException(status_code=500, detail="Failed to store analysis results.")
    except Exception as e: # Catch errors during function execution
        logger.error(f"Error during analysis execution: {e}", exc_info=True)
        data_info["analysis_status"] = "error_running"
        raise HTTPException(status_code=500, detail=f"Analysis execution error: {e}")


@app.post("/chat/policy")
async def chat_policy_endpoint(request: ChatRequest):
    # (Implementation as before - calls stream_llm_response which now does re-ranking)
    logger.info(f"Received policy chat request: '{request.prompt[:75]}...'")
    if app_state is None: raise HTTPException(status_code=500, detail="App state error.")
    policy_retriever = app_state.get("policy_retriever")
    if not policy_retriever: raise HTTPException(status_code=400, detail="Policy not processed.")
    policy_system_prompt = """<|system|>You are an expert HR Assistant providing guidance to an HR representative (the user). Your primary goal is to outline the specific steps the **HR representative** must take to handle the situation described by the user, based *primarily* on the provided **re-ranked policy document context**: {context}. Always address the HR representative directly and detail *their* required actions according to the policy. If the context is relevant, cite it. If not, state the policy doesn't cover it and provide general HR best practice steps for the HR representative. IMPORTANT: When the user asks about 'meal breaks' (typically unpaid, longer breaks for meals), carefully distinguish this from 'rest breaks' (typically paid, shorter breaks). Prioritize information explicitly discussing meal periods if available in the context.<|end|>
<|context|>Relevant Re-ranked Policy Information:
{context}
<|end|>"""
    return StreamingResponse(stream_llm_response(request.prompt, policy_retriever, policy_system_prompt), media_type="application/x-ndjson")

@app.post("/chat/analysis")
async def chat_analysis_endpoint(request: ChatRequest):
    # (Implementation as before - calls stream_llm_response which now does re-ranking)
    logger.info(f"Received analysis chat request: '{request.prompt[:75]}...'")
    if app_state is None: raise HTTPException(status_code=500, detail="App state error.")
    analysis_retriever = app_state.get("analysis_retriever")
    if not analysis_retriever: raise HTTPException(status_code=400, detail="Analysis not run/ready.")
    analysis_system_prompt = """<|system|>You are an expert HR Analyst AI assistant... {context}.<|end|><|context|>Analysis Results Context:\n{context}<|end|>""" # Truncated
    return StreamingResponse(stream_llm_response(request.prompt, analysis_retriever, analysis_system_prompt), media_type="application/x-ndjson")

# --- Start: Ragas Evaluation Endpoint (Passing Raw Objects) ---

# Define a sample test dataset (Expand this significantly!)
eval_dataset_dict = {
    'question': [
        "An employee has worked for the organization for 14 months and has worked 1,300 hours in the last 12 months. Their spouse needs care due to a serious health condition. Are they eligible for FMLA/CFRA leave? Explain why or why not based on the policy.",
        "A non-exempt employee works 10 hours on Monday. Are they entitled to one or two meal breaks? Explain based on the policy.",
        "What is the maximum vacation accrual rate per year for long-term employees?",
        "Is personal use of the company email allowed?",
    ],
    'ground_truth': [
        "Yes, the employee is eligible for FMLA/CFRA leave. Eligibility requires working for the organization for at least 12 months and having worked at least 1,250 hours in the preceding 12 months. This employee meets both criteria (14 months, 1,300 hours). Caring for a spouse with a serious health condition is a permissible reason for leave under the policy.",
        "The employee is entitled to two unpaid 30 [or 60] minute meal breaks. The first is required because they worked more than 5 hours, and the second is required because they worked more than 10 hours (starting no later than the end of the 10th hour).",
        "The maximum vacation accrual rate is 6.15 hours biweekly, for a maximum of 20 days per year, for employees with 11 or more years of continuous service.",
        "Excessive personal use is grounds for reprimand, implying limited, non-excessive personal use might be tolerated, but the policy primarily states electronic systems are for business use. It also prohibits using email to solicit for non-business matters.",
    ]
}

@app.post("/evaluate_rag")
async def evaluate_rag_endpoint():
    """Runs the predefined test dataset through the RAG pipeline and evaluates using Ragas."""
    global llm_model, embeddings # Use global models

    # Check required libraries and components
    if not RAGAS_INSTALLED: raise HTTPException(status_code=501, detail="Ragas/datasets not installed.")
    if not app_state.get("policy_retriever"): raise HTTPException(status_code=400, detail="Policy not processed.")
    if not embeddings: raise HTTPException(status_code=503, detail="Embeddings not loaded for Ragas.")
    if not LANGCHAIN_OPENAI_INSTALLED or not ChatOpenAI: raise HTTPException(status_code=501, detail="langchain-openai not installed.")
    if not OPENROUTER_API_KEY: raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not set.")

    logger.info(f"Starting RAG evaluation with {len(eval_dataset_dict['question'])} questions...")
    policy_retriever = app_state["policy_retriever"]
    policy_system_prompt = """<|system|>You are an expert HR Assistant... {context}.<|end|><|context|>Relevant Uploaded Policy Information:\n{context}<|end|>""" # Same prompt as chat

    results_for_eval = []
    for i, question in enumerate(eval_dataset_dict['question']):
        logger.info(f"Generating answer for eval question {i+1}...")
        rag_output = generate_rag_answer(question, policy_retriever, policy_system_prompt)
        eval_item = { "question": question, "answer": rag_output.get("answer", "[Error]"), "contexts": rag_output.get("contexts", []), "ground_truth": eval_dataset_dict['ground_truth'][i] }
        results_for_eval.append(eval_item)

    try:
        hf_dataset = Dataset.from_list(results_for_eval)
        logger.info("Dataset prepared for Ragas evaluation.")
    except Exception as e: logger.error(f"Error preparing Ragas dataset: {e}", exc_info=True); raise HTTPException(status_code=500, detail="Ragas dataset prep error.")

    metrics = [ faithfulness, answer_relevancy, context_recall, context_precision, answer_correctness ]
    active_metrics = [m for m in metrics if m is not None]
    if not active_metrics: raise HTTPException(status_code=501, detail="No Ragas metrics loaded.")

    logger.info(f"Running Ragas evaluation with metrics: {[m.name for m in active_metrics]}...")
    try:
        # --- Instantiate OpenRouter client and pass raw objects to evaluate ---
        logger.info(f"Instantiating OpenRouter client for Ragas: {RAGAS_EVAL_LLM_MODEL}")
        openrouter_llm_client = ChatOpenAI(
            model_name=RAGAS_EVAL_LLM_MODEL,
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base=OPENROUTER_BASE_URL,
            temperature=0.0
        )
        if not embeddings: raise HTTPException(status_code=500, detail="Local embeddings object not available for Ragas.")

        logger.info("Passing raw LangChain objects (ChatOpenAI client, OllamaEmbeddings) to Ragas evaluate.")
        result_scores = evaluate(
            hf_dataset,
            metrics=active_metrics,
            llm=openrouter_llm_client, # Pass the raw ChatOpenAI client
            embeddings=embeddings      # Pass the raw OllamaEmbeddings object
        )
        # --- End passing raw objects ---

        logger.info("Ragas evaluation completed.")
        logger.info(f"Evaluation Scores: {result_scores}")
        # --- Start: FIX for KeyError: 0 ---
        # Convert Ragas result (often a Dataset object or dict-like) to a plain dictionary
        if hasattr(result_scores, 'items'):
             scores_dict = dict(result_scores.items())
        elif isinstance(result_scores, dict):
             scores_dict = result_scores
        else:
             try: # Attempt conversion if it behaves like a Hugging Face Dataset row/dict
                 scores_dict = result_scores.to_dict()
             except AttributeError:
                 logger.warning("Could not convert Ragas result to dict directly. Returning raw object representation.")
                 scores_dict = {"ragas_result": str(result_scores)} # Fallback
        # --- End: FIX for KeyError: 0 ---
        return JSONResponse(content=scores_dict)

    except Exception as e:
        logger.error(f"Error during Ragas evaluation: {e}", exc_info=True)
        error_str = str(e)
        if "api_key" in error_str.lower() or "authentication" in error_str.lower():
             error_detail = f"Authentication error during Ragas evaluation: {e}. Check OPENROUTER_API_KEY."
        elif "does not support" in error_str or "expected type" in error_str or "Could not load" in error_str:
             error_detail = f"Model/Embedding compatibility error during Ragas evaluation: {e}. Ragas v{ragas.__version__ if ragas else 'Unknown'} might not support direct use of these LangChain objects. Check Ragas documentation for custom model configuration."
        else:
             error_detail = f"Error during Ragas evaluation: {e}. Check Ragas setup."
        raise HTTPException(status_code=500, detail=error_detail)

# --- End: Ragas Evaluation Endpoint ---


# --- Dashboard & Health Check Endpoints (Keep as before) ---
@app.get("/dashboard/basic")
async def get_basic_dashboard():
    logger.info("Request for basic dashboard.")
    if app_state is None or not app_state.get("data_file_info"): raise HTTPException(status_code=400, detail="No data.")
    data_info = app_state["data_file_info"]; data_path = data_info.get("path")
    if not data_path or not os.path.exists(data_path): raise HTTPException(status_code=404, detail="Data file not found.")

    # --- Start: Check if dashboard function was imported ---
    if generate_basic_dashboard is None:
        logger.error("generate_basic_dashboard function not imported successfully.")
        raise HTTPException(status_code=501, detail="Basic dashboard function not available due to import error.")
    # --- End: Check ---

    logger.info(f"Generating basic dashboard...")
    fig = None
    try:
        fig = generate_basic_dashboard(data_path) # Uses globally defined function (checked above)
        if not fig: raise HTTPException(status_code=500, detail="Failed to generate basic dashboard.")
        buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=150, bbox_inches='tight'); plt.close(fig); buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e: logger.error(f"Dashboard error: {e}", exc_info=True); raise HTTPException(status_code=500, detail=f"{e}")

@app.get("/dashboard/extended")
async def get_extended_dashboard():
    logger.info("Request for extended dashboard.")
    if app_state is None or not app_state.get("data_file_info"): raise HTTPException(status_code=400, detail="No data.")
    data_info = app_state["data_file_info"]; data_path = data_info.get("path")
    if not data_path or not os.path.exists(data_path): raise HTTPException(status_code=404, detail="Data file not found.")

    # --- Start: Check if dashboard function was imported ---
    if generate_extended_dashboard is None:
        logger.error("generate_extended_dashboard function not imported successfully.")
        raise HTTPException(status_code=501, detail="Extended dashboard function not available due to import error.")
    # --- End: Check ---

    logger.info(f"Generating extended dashboard...")
    fig = None
    try:
        fig = generate_extended_dashboard(data_path) # Uses globally defined function (checked above)
        if not fig: raise HTTPException(status_code=500, detail="Failed to generate extended dashboard.")
        buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=150, bbox_inches='tight'); plt.close(fig); buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e: logger.error(f"Dashboard error: {e}", exc_info=True); raise HTTPException(status_code=500, detail=f"{e}")

@app.get("/health")
def health_check():
    health_status = {"status": "ok"}
    if not llm_model: health_status["llm_status"] = "error: not loaded"
    if not embeddings: health_status["embedding_status"] = "error: not initialized"
    if SENTENCE_TRANSFORMERS_INSTALLED and not cross_encoder: health_status["cross_encoder_status"] = "error: not loaded"
    if not RAGAS_INSTALLED: health_status["ragas_status"] = "error: not installed"
    # Check analysis/dashboard import status
    analysis_imports_ok = all([run_attrition_analysis, run_clustering_analysis])
    dashboard_imports_ok = all([generate_basic_dashboard, generate_extended_dashboard])
    if not analysis_imports_ok: health_status["analysis_scripts"] = "error: import failed"
    if not dashboard_imports_ok: health_status["dashboard_scripts"] = "error: import failed"
    if analysis_imports_ok and dashboard_imports_ok: health_status["external_scripts"] = "ok (imports attempted)"

    health_status["ragas_config"] = "local_to_endpoint"
    return JSONResponse(content=health_status)

# --- Optional: Run directly for development ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server directly for development...")
    # Set reload=False if experiencing issues with global model loading during reloads
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

