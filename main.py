# ============================================
# FILE: main.py - FastAPI Backend
# ============================================
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path
import uvicorn
import os
import shutil
from datetime import datetime
from session_logger import session_logger
from llm import gemini_chat

# Get base directory
BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Gemini RAG Chatbot API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files with absolute path
app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static"
)

# Templates with absolute path
templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)

os.makedirs("./data", exist_ok=True)
os.makedirs("./vector_store", exist_ok=True)

class ChatRequest(BaseModel):
    message: str
    use_rag: bool = True
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    is_rag: bool
    success: bool
    session_id: str

class SessionCreateResponse(BaseModel):
    session_id: str
    created_at: str

class SessionListResponse(BaseModel):
    sessions: List[Dict[str, Any]]
    total: int

class DocUploadResponse(BaseModel):
    success: bool
    message: str
    filename: Optional[str] = None

class FeaturesModel(BaseModel):
    hybrid_search: bool
    reranking: bool
    mmr: bool

class RAGStatusResponse(BaseModel):
    has_vector_store: bool
    document_count: int
    rag_chain_initialized: bool
    active_sessions: int
    features: FeaturesModel

class HistoryItem(BaseModel):
    role: str
    parts: List[str]

class ChatHistoryResponse(BaseModel):
    history: List[HistoryItem]
    session_id: str


# ============================================
# SERVE FRONTEND HTML
# ============================================
@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


# ============================================
# API ENDPOINTS
# ============================================

@app.get("/api")
def api_info():
    return {
        "message": "Gemini RAG Chatbot API with Multi-Session Support",
        "version": "3.0.0",
        "features": [
            "Multi-Session Management",
            "Arabic Word Document RAG",
            "Hybrid Search (BM25 + Dense)",
            "Gemini Integration"
        ]
    }

@app.post("/session/create", response_model=SessionCreateResponse)
async def create_session(session_id: Optional[str] = None):
    try:
        new_session_id = gemini_chat.create_session(session_id)
        return SessionCreateResponse(
            session_id=new_session_id,
            created_at=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/list", response_model=SessionListResponse)
async def list_sessions():
    try:
        sessions = gemini_chat.get_all_sessions()
        return SessionListResponse(
            sessions=sessions,
            total=len(sessions)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    try:
        success = gemini_chat.delete_session(session_id)
        if success:
            return {"message": f"Session {session_id} deleted successfully", "success": True}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/session/{session_id}/reset")
async def reset_session(session_id: str):
    try:
        success = gemini_chat.reset_chat(session_id)
        if success:
            return {"message": f"Session {session_id} reset successfully", "success": True}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{session_id}/history", response_model=ChatHistoryResponse)
async def get_session_history(session_id: str):
    try:
        history = gemini_chat.get_chat_history(session_id)
        
        formatted_history = []
        for item in history:
            role = "user" if item.role == "user" else "assistant"
            parts = []
            for part in item.parts:
                if hasattr(part, 'text'):
                    parts.append(part.text)
                elif isinstance(part, str):
                    parts.append(part)
            
            if parts:
                formatted_history.append(HistoryItem(role=role, parts=parts))
        
        return ChatHistoryResponse(
            history=formatted_history,
            session_id=session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/session/cleanup")
async def cleanup_sessions():
    try:
        gemini_chat.cleanup_sessions()
        active_sessions = gemini_chat.get_all_sessions()
        return {
            "message": "Session cleanup completed",
            "active_sessions": len(active_sessions),
            "success": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if not request.message or request.message.strip() == "":
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        session_id = request.session_id
        if not session_id:
            session_id = gemini_chat.create_session()
        
        result = gemini_chat.get_response(
            message=request.message.strip(),
            session_id=session_id,
            use_rag=request.use_rag
        )
        
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag/status", response_model=RAGStatusResponse)
async def get_rag_status():
    try:
        status = gemini_chat.get_rag_status()
        
        if "features" not in status:
            status["features"] = {
                "hybrid_search": False,
                "reranking": False,
                "mmr": False
            }
        
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-docx")
async def upload_docx(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(('.docx', '.doc')):
            raise HTTPException(status_code=400, detail="File must be a Word document (.docx or .doc)")
        
        file_path = f"./data/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        success, message = gemini_chat.process_new_docx(file_path)
        
        if success:
            return DocUploadResponse(
                success=True,
                message=message,
                filename=file.filename
            )
        else:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=500, detail=message)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/process-directory")
async def process_directory():
    try:
        gemini_chat._initialize_rag()
        status = gemini_chat.get_rag_status()
        
        return {
            "success": True,
            "message": f"Processed directory. Documents in vector store: {status['document_count']}",
            "status": status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/list")
async def list_documents():
    try:
        documents = []
        if os.path.exists("./data"):
            for filename in os.listdir("./data"):
                if filename.lower().endswith(('.docx', '.doc')):
                    file_path = os.path.join("./data", filename)
                    file_size = os.path.getsize(file_path)
                    documents.append({
                        "filename": filename,
                        "size_kb": round(file_size / 1024, 2),
                        "uploaded": datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
                    })
        
        return {
            "success": True,
            "documents": documents,
            "count": len(documents)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/logs/sessions")
async def get_session_logs(lines: int = 100):
    try:
        log_file = "./logs/sessions.log"
        if not os.path.exists(log_file):
            return {"logs": [], "message": "No session logs found"}
        
        with open(log_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        return {
            "success": True,
            "logs": [line.strip() for line in recent_lines],
            "total_lines": len(recent_lines)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    rag_status = gemini_chat.get_rag_status()
    return {
        "status": "healthy",
        "service": "gemini-rag-chatbot-api",
        "version": "3.0.0",
        "rag_initialized": rag_status["rag_chain_initialized"],
        "documents_loaded": rag_status["document_count"],
        "active_sessions": rag_status["active_sessions"]
    }

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    
    # Get port from environment variable (for Azure) or default to 8000
    port = int(os.getenv("PORT", 8000))
    
    print(f"🚀 Starting IDSC Assistant on port {port}")
    print(f"🌐 Access the application at: http://localhost:{port}")
    
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)