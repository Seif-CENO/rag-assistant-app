# 👷 OSHA Safety & PPE Auditor (RAG + YOLOv8)

## Overview
An enterprise-grade, full-stack AI application that fuses Computer Vision (YOLOv8) with Retrieval-Augmented Generation (RAG). This system acts as an automated safety auditor. Users can upload worksite images, and the system will detect objects in the scene and query an OSHA-grounded local LLM to verify safety compliance.

## Architecture Diagram
```text
[User] -> (Streamlit Frontend) -> [Uploads Image & Question]
                                      |
                                      v
                              (FastAPI Backend)
                                /            \
                [YOLOv8 Vision Model]     [Chroma Vector Store]
                  (Detects Objects)       (Retrieves OSHA Chunks)
                                \            /
                                 v          v
                        [Local Llama3 via Ollama]
                                      |
                                      v
                     [Grounded Answer + Citations returned]
```

## Tech Stack
* **AI/Models:** Ollama (Llama 3), YOLOv8 (Ultralytics), HuggingFace Embeddings (`all-MiniLM-L6-v2`)
* **Backend:** Python 3.12, FastAPI, LangChain, ChromaDB
* **Frontend:** Streamlit, Requests

## Project Structure
* `data/` - Contains the raw OSHA PDF guidelines and worksite images. *(Excluded from git to protect repository size)*
* `data/vector_store/` - Persisted ChromaDB embeddings. *(Excluded from git)*
* `notebooks/` - `rag_pipeline.ipynb` containing data parsing, chunking, and evaluation.
* `backend/` - FastAPI application serving the RAG and Vision endpoints.
* `frontend/` - Streamlit application providing the chat UI.

## Domain & Data Description
This project focuses on Workplace Safety and PPE compliance. The text corpus consists of official OSHA guidelines regarding employer PPE responsibilities, hazard protection, and safety protocols. The vision component analyzes images of construction workers to detect the presence or absence of required equipment.

## Setup Instructions

### 1. Prerequisites
* Python 3.10+
* Ollama installed and running locally with the Llama 3 model (`ollama pull llama3`)

### 2. Backend Setup
```bash
python -m venv .venv
pip install -r backend/requirements.txt
cd backend
uvicorn app.main:app --reload
```

### 3. Frontend Setup
In a new terminal (with the same virtual environment activated):
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

## Environment Variables
Create a `.env` file in the `frontend/` directory:
| Variable | Description | Example |
| :--- | :--- | :--- |
| `API_BASE_URL` | The URL of the FastAPI backend | `http://localhost:8000` |

## API Reference (cURL Example)
Test the vision-augmented RAG endpoint:
```bash
curl -X 'POST' \
  'http://localhost:8000/query-vision' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'question="Is this worker following safety protocols?"' \
  -F 'file=@data/images/s-l400.jpeg;type=image/jpeg'
```

## Evaluation Results
| Question | Retrieved Source | Answer Correct? | Grounded/Hallucinated? |
| :--- | :--- | :--- | :--- |
| Who pays for PPE? | Handout 2 (Page 0) | Yes | Grounded |
| What protection is required for head injuries? | Handout 2 (Page 0) | Yes | Grounded |
| How to prevent struck-by accidents? | Hazards QC (Page 0) | Yes | Grounded |
| When should workers wear respiratory protection? | PPE Factsheet (Page 1) | Yes | Grounded |

*(Failure cases mitigated by enforcing standard context limits and overriding default YOLO classes with specific PPE weights).*