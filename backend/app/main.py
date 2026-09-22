from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import query

app = FastAPI(title="OSHA Safety RAG Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
app.include_router(query.router)

@app.get("/health")
def health_check():
    return {"status": "healthy"}