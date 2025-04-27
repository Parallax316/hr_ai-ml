# HR AI Assistant with RAG and Data Analysis

## Download link for my fine-tuned llm - 
  https://drive.google.com/file/d/1Oj1G34m9aylIrjor3gEM2rRUl2fvo3U5/view?usp=sharing

## Overview

This project implements an interactive AI Assistant designed for Human Resources (HR) representatives. It features a FastAPI backend providing API endpoints and a React frontend for the user interface. The assistant leverages Large Language Models (LLMs) – specifically a **custom microsoft phi-4 mini instruct model fine-tuned by the project author on HR Q&A data using PEFT/LoRA techniques and converted to GGUF f16 format** – combined with Retrieval-Augmented Generation (RAG) and Cross-Encoder Re-ranking to provide context-aware assistance based on company documents and dynamic employee data analysis.

The application allows HR users to:

1.  **Chat with Policy Documents:** Upload a company policy document (e.g., Employee Handbook PDF). The AI Assistant uses RAG with re-ranking to retrieve relevant sections and provide answers grounded in the document via a chat interface.
2.  **Analyze Employee Data:** Upload employee data (CSV format) to trigger backend Machine Learning analysis scripts (e.g., Attrition Prediction, Clustering). The results are stored and embedded.
3.  **Gain Data Insights:** Interact with the AI Assistant via chat to get summaries and insights derived *directly* from the results of the employee data analysis, also using RAG with re-ranking.
4.  **Visualize Data:** View basic and extended dashboards generated from the employee data.
5.  **Evaluate RAG Performance:** An endpoint (`/evaluate_rag`) allows running a predefined test set against the policy RAG pipeline and calculating performance metrics using the Ragas library.


## Screenshots - 

![{23D5311E-3EC0-4BA4-995D-05007A9A0C5B}](https://github.com/user-attachments/assets/66f200e0-e6d6-48ea-8bdf-582ab74fb373)

![{B587E897-4CC4-4990-956E-98F3A63C64EB}](https://github.com/user-attachments/assets/1bd279bf-5966-4f44-a459-99fa47f5eb67)

![{B18E5809-C239-4C84-938D-B3B16E2D9975}](https://github.com/user-attachments/assets/2fdc8398-9e86-4c86-a967-742fb0ad28f2)

![{AD659E2A-F9AA-4378-8B03-74FEB56F56D6}](https://github.com/user-attachments/assets/264abe55-de1c-40ca-bd53-f79e1758b4cf)


## Architecture

* **Backend:** Python using FastAPI, serving API endpoints for document/data upload, analysis triggering, chat streaming, dashboard generation, and RAG evaluation. Handles core RAG logic, LLM interaction, and data processing.
* **Frontend:** React (likely Next.js based on file structure seen previously) using TypeScript. Provides the user interface for uploads, chat interactions, and displaying results/dashboards. Interacts with the FastAPI backend via API calls.

## Features

* **React-based UI:** Modern frontend for user interaction.
* **FastAPI Backend:** Robust API for handling requests and background tasks.
* **Custom Fine-Tuned LLM:** Utilizes a microsoft phi-4 mini model fine-tuned specifically on HR Q&A data using PEFT/LoRA techniques and converted to GGUF f16 format for efficient local execution via `llama-cpp-python`.
* **Policy Document RAG:**
    * Upload PDF policy documents.
    * Document processing (Recursive Chunking: 350/75) via LangChain.
    * Vector embedding creation using Ollama (`nomic-embed-text`).
    * In-memory vector storage using ChromaDB.
    * **Advanced Retrieval:** Initial retrieval (`k=10`) followed by Cross-Encoder re-ranking (`top_n=2`) using `sentence-transformers`.
    * LLM interaction grounded in re-ranked policy context.
    * Streaming responses (`ndjson`) for chat.
* **Data Analysis RAG:**
    * Upload employee data via CSV.
    * Trigger execution of Python ML scripts (`attrition_analysis.py`, `clustering_analysis.py`).
    * Analysis results are processed, embedded (Ollama), and stored in ChromaDB.
    * Uses the same re-ranking retrieval strategy for Q&A based on analysis results.
* **Data Visualization:** Endpoints to generate and display Matplotlib dashboards as images.
* **RAG Evaluation:** Dedicated API endpoint (`/evaluate_rag`) using the `Ragas` library to assess retrieval and generation quality against a predefined test set and ground truths, using OpenRouter (`gpt-4o-mini`) as the evaluation LLM.

## Technology Stack

* **Backend Framework:** FastAPI
* **Frontend Framework:** React / Next.js , TypeScript
* **LLM (Primary):** Microsoft phi-4-mini-instruct (Fine-tuned on HR Q&A using PEFT/LoRA, converted to GGUF f16 format)
* **LLM Runner:** `llama-cpp-python`
* **Embeddings:** `nomic-embed-text` via Ollama
* **Vector Store:** ChromaDB (in-memory)
* **Orchestration/RAG:** LangChain (`langchain`, `langchain-community`, `langchain-chroma`, `langchain-ollama`, `langchain-openai`)
* **Re-ranking:** `sentence-transformers` (Cross-Encoder)
* **RAG Evaluation:** `ragas`, `datasets`
* **Evaluation LLM:** OpenRouter (`gpt-4o-mini`) via `langchain-openai`
* **PDF Processing:** `pypdf`
* **Data Analysis:** Python, Pandas, Scikit-learn (in analysis scripts)
* **Environment:** Python 3.x, Virtual Environment (`venv`)

## Setup and Installation

1.  **Clone Repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```
2.  **Backend Setup:**
    * Navigate to the backend directory (if separate).
    * Create and activate a Python virtual environment:
        ```bash
        python -m venv .venv
        # Windows PowerShell: .\.venv\Scripts\Activate.ps1
        # macOS/Linux: source .venv/bin/activate
        ```
    * Install backend dependencies:
        ```bash
        pip install -r requirements.txt
        # Ensure requirements.txt includes: fastapi uvicorn python-multipart llama-cpp-python langchain langchain-community langchain-chroma langchain-ollama langchain-openai langchain-experimental pypdf pandas scikit-learn sentence-transformers ragas datasets matplotlib python-dotenv openai # Added openai
        ```
    * Set up Ollama (install, pull `nomic-embed-text`, ensure server is running).
    * Place your fine-tuned LLM GGUF file (e.g., `hr-finetune-merged.gguf`) and update `MODEL_PATH` in `main.py`.
    * Place your analysis scripts (`attrition_analysis.py`, etc.) where `main.py` can import them.
    * Set the `OPENROUTER_API_KEY` environment variable for Ragas evaluation:
        ```powershell
        # Windows PowerShell:
        $env:OPENROUTER_API_KEY='your-openrouter-key'
        # macOS/Linux:
        # export OPENROUTER_API_KEY='your-openrouter-key'
        ```
3.  **Frontend Setup:**
    * Navigate to the frontend directory (if separate).
    * Install Node.js dependencies:
        ```bash
        npm install
        ```

## Running the Application

1.  **Start Backend Server:**
    * Activate the backend virtual environment.
    * Set the `OPENROUTER_API_KEY` environment variable.
    * Run the FastAPI server (from the backend directory):
        ```bash
        python main.py
        # OR using uvicorn directly:
        # uvicorn main:app --host 0.0.0.0 --port 8000
        ```
        *(Note: Running without `--reload` is recommended due to global models)*
2.  **Start Frontend Development Server:**
    * Navigate to the frontend directory.
    * Run the React development server:
        ```bash
        npm run dev
        ```
3.  **Interact:**
    * Open the frontend URL (usually `http://localhost:3000`) in your browser.
    * Upload policy PDF / data CSV and interact via the UI.
    * To run evaluation (optional): Send a POST request to `http://localhost:8000/evaluate_rag` using `curl` or Postman after uploading the policy PDF.

## RAG Evaluation Results (Sample)

The `/evaluate_rag` endpoint uses Ragas with `gpt-4o-mini` as the judge LLM and local embeddings against a small test dataset. Initial results with the Re-ranking RAG setup yielded scores like:

* **faithfulness:** 0.4214 *(Low - Indicates LLM often deviates from provided context, potentially due to fine-tuning)*
* **answer_relevancy:** 0.9510 *(High - Answers are on-topic for the questions)*
* **context_recall:** 0.6042 *(Moderate - Retriever sometimes misses necessary info)*
* **context_precision:** 0.7500 *(Good - Re-ranked context is mostly relevant)*
* **answer_correctness:** 0.5472 *(Moderate - Reflects issues in faithfulness & recall)*

**(Note:** These scores are based on a small, hardcoded sample dataset and placeholder ground truths in `main.py`. Expand the dataset and refine ground truths for more meaningful results.)*

## Testing Summary (Based on Debugging)

Manual testing with specific queries against the Re-ranking RAG configuration revealed:

* **PASS:** FMLA/CFRA Eligibility (Q18) - Re-ranking successfully retrieved the correct eligibility criteria, fixing earlier errors.
* **PASS:** At-Will Policy (Q21) - Retrieved relevant context.
* **PASS:** Holidays Policy (Q22) - Retrieved relevant context.
* **PASS:** Bereavement Leave Policy (Q23) - Retrieved relevant context.
* **FAIL:** Meal Breaks vs. Rest Breaks (Q19) - Consistently retrieved context about *rest breaks* even when asked about *meal breaks*. Re-ranking did not resolve this specific confusion.

## Key Learnings & Observations

* **RAG Effectiveness:** RAG significantly grounds LLM answers in provided documents/data. Re-ranking improved retrieval relevance over simpler methods.
* **RAG Tuning is Iterative:** Simple chunking/`k` selection were insufficient. Re-ranking fixed some issues but not all (e.g., meal/rest break confusion). Further tuning (prompting, semantic chunking) is needed for specific challenging queries.
* **Fine-tuning Impact:** Using a model fine-tuned on HR Q&A ( Microsoft phi-4-mini-instruct via PEFT/LoRA) provides strong domain knowledge and conversational ability. However, this can sometimes lead the model to answer based on its internal training rather than strictly adhering to the retrieved RAG context, negatively impacting `faithfulness` scores. Balancing fine-tuning goals with the need for context adherence in RAG is crucial.
* **Evaluation is Crucial:** Quantitative evaluation using Ragas highlighted the low `faithfulness` score, pinpointing a key area for improvement that wasn't solely a retrieval problem.

## Potential Future Improvements

* **Improve RAG for Meal/Rest Breaks:**
    * Implement stricter **Prompt Engineering** in `main.py`'s system prompt to force the LLM to differentiate between meal and rest breaks based on the context provided.
    * Revisit **Semantic Chunking** with parameter tuning specifically for the policy document, aiming to better isolate these sections, possibly combined with re-ranking.
* **Improve Faithfulness:** Continue refining system prompts to emphasize context adherence strictly. Experiment with lower LLM temperature settings during generation. Add examples to the fine-tuning data that demonstrate answering *only* from provided context.
* **Improve Context Recall:** Experiment with `RERANK_TOP_N = 3` (or more) after improving `faithfulness`, to provide more context.
* **Expand Evaluation Dataset:** Add more diverse questions and accurate ground truths to `eval_dataset_dict` in `main.py` for more reliable Ragas scores.
* **Persistent Vector Stores:** Optionally configure ChromaDB to persist to disk (`persist_directory`).
* **Configuration File:** Move settings like `MODEL_PATH`, collection names, `OPENROUTER_API_KEY` etc., from `main.py` to a `.env` file or `config.yaml`.
* **UI Enhancements:** Display RAG evaluation results, allow configuration changes via UI.
* **Agentic Behavior:** Explore LangChain agents for more complex workflows.


## Project Author - 
Simarpreet Singh




