from fastapi import APIRouter, File, UploadFile, Form
from app.schemas.query import QueryRequest, QueryResponse
from app.services.retrieval import generate_rag_response, generate_vision_rag_response

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    answer, sources = generate_rag_response(request.question)
    return QueryResponse(answer=answer, sources=sources)

@router.post("/query-vision", response_model=QueryResponse)
def query_vision_rag(question: str = Form(...), file: UploadFile = File(...)):
    image_bytes = file.file.read()
    answer, sources, detections = generate_vision_rag_response(question, image_bytes)
    return QueryResponse(answer=answer, sources=sources, detections=detections)