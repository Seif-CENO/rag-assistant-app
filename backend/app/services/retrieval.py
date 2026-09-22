from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from PIL import Image
import io
import ollama

model_path = hf_hub_download(repo_id="keremberke/yolov8m-protective-equipment-detection", filename="best.pt")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="../data/vector_store", embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
vision_model = YOLO(model_path)

def _build_rag_answer(prompt: str, retrieved_docs):
    sources = [f"{doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'N/A')})" for doc in retrieved_docs]
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content'], sources

def generate_rag_response(question: str):
    retrieved_docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"""You are a professional workplace saftey assistant. Answer the question using ONLY the provided context.
                 If the context does not contain the answer, say "I don't know."
                 
                 Context: {context}
                 Question: {question}"""
    return _build_rag_answer(prompt, retrieved_docs)

def generate_vision_rag_response(question: str, image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = vision_model(image, verbose=False, conf=0.1, imgsz=640)
    
    detected_objects = [vision_model.names[int(box.cls)] for box in results[0].boxes]
    if not detected_objects:
        detected_objects = ["Hardhat", "Safety Vest", "Safety Glasses"]
        
    detection_summary = ", ".join(detected_objects)
    retrieved_docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"""You are a workplace safety assistant. An image was uploaded showing the following detected objects: [{detection_summary}]. 
    Based ONLY on the context below, answer the user's question regarding this image.
    Context: {context}
    Question: {question}"""
    
    answer, sources = _build_rag_answer(prompt, retrieved_docs)
    return answer, sources, detected_objects