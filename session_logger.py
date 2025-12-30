# ============================================
# FILE 4: session_logger.py
# ============================================
import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any
import json

class SessionLogger:
    """Centralized logger for session activities"""
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.session_logger = self._setup_logger(
            'session_activity',
            os.path.join(log_dir, 'sessions.log')
        )
        
        self.chat_logger = self._setup_logger(
            'chat_activity', 
            os.path.join(log_dir, 'chats.log')
        )
        
        self.error_logger = self._setup_logger(
            'errors',
            os.path.join(log_dir, 'errors.log'),
            level=logging.ERROR
        )
        
        self.rag_logger = self._setup_logger(
            'rag_activity',
            os.path.join(log_dir, 'rag.log')
        )
    
    def _setup_logger(self, name: str, log_file: str, level=logging.INFO):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        if logger.handlers:
            return logger
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_session_created(self, session_id: str, metadata: Optional[Dict] = None):
        msg = f"SESSION_CREATED | ID: {session_id[:8]}..."
        if metadata:
            msg += f" | Metadata: {json.dumps(metadata, ensure_ascii=False)}"
        self.session_logger.info(msg)
    
    def log_session_deleted(self, session_id: str, reason: str = "manual"):
        self.session_logger.info(
            f"SESSION_DELETED | ID: {session_id[:8]}... | Reason: {reason}"
        )
    
    def log_session_reset(self, session_id: str):
        self.session_logger.info(
            f"SESSION_RESET | ID: {session_id[:8]}..."
        )
    
    def log_session_expired(self, session_id: str, inactive_minutes: float):
        self.session_logger.warning(
            f"SESSION_EXPIRED | ID: {session_id[:8]}... | "
            f"Inactive: {inactive_minutes:.1f} minutes"
        )
    
    def log_chat_message(self, session_id: str, role: str, message_preview: str, 
                         use_rag: bool = False, sources_count: int = 0):
        preview = message_preview[:100] + "..." if len(message_preview) > 100 else message_preview
        msg = (
            f"CHAT_MESSAGE | Session: {session_id[:8]}... | "
            f"Role: {role} | RAG: {use_rag}"
        )
        if sources_count > 0:
            msg += f" | Sources: {sources_count}"
        msg += f" | Preview: {preview}"
        self.chat_logger.info(msg)
    
    def log_rag_search(self, session_id: str, query_preview: str, 
                       results_count: int, search_method: str):
        preview = query_preview[:100] + "..." if len(query_preview) > 100 else query_preview
        self.rag_logger.info(
            f"RAG_SEARCH | Session: {session_id[:8]}... | "
            f"Method: {search_method} | Results: {results_count} | "
            f"Query: {preview}"
        )
    
    def log_document_processed(self, filename: str, chunks_count: int, 
                               success: bool = True):
        status = "SUCCESS" if success else "FAILED"
        self.rag_logger.info(
            f"DOC_PROCESSED | File: {filename} | "
            f"Chunks: {chunks_count} | Status: {status}"
        )
    
    def log_error(self, operation: str, error: Exception, 
                  session_id: Optional[str] = None):
        msg = f"ERROR | Operation: {operation}"
        if session_id:
            msg += f" | Session: {session_id[:8]}..."
        msg += f" | Error: {str(error)}"
        self.error_logger.error(msg, exc_info=True)
    
    def log_cleanup(self, expired_count: int, total_count: int):
        self.session_logger.info(
            f"SESSION_CLEANUP | Expired: {expired_count} | "
            f"Remaining: {total_count - expired_count}"
        )
    
    def log_system_status(self, status_info: Dict[str, Any]):
        self.session_logger.info(
            f"SYSTEM_STATUS | Documents: {status_info.get('document_count', 0)} | "
            f"Active Sessions: {status_info.get('active_sessions', 0)} | "
            f"RAG Enabled: {status_info.get('has_vector_store', False)}"
        )
    
    def get_session_stats(self, hours: int = 24) -> Dict[str, Any]:
        return {
            "hours": hours,
            "note": "Implement log parsing for detailed statistics"
        }


session_logger = SessionLogger()

