from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from ultralytics import YOLO
from PIL import Image
import io
import ollama

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="../data/vector_store", embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
vision_model = YOLO('yolov8n.pt')

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
    image = Image.open(io.BytesIO(image_bytes))
    results = vision_model(image, verbose=False)
    detected_objects = [vision_model.names[int(box.cls)] for box in results[0].boxes]
    detection_summary = ", ".join(detected_objects) if detected_objects else "No objects detected"

    retrieved_docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"""You are a professional workplace saftey assistant. An image was uploaded showing the following detected objects: [{detection_summary}].
                 Based ONLY on the context below, answer the user's questions regarding this image.
                 Context: {context}
                 Question: {question}"""
    answer, sources = _build_rag_answer(prompt, retrieved_docs)
    return answer, sources, detected_objects