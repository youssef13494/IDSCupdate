# ============================================
# FILE 3: session_manager.py
# ============================================
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
import google.generativeai as genai
from dataclasses import dataclass, field
from session_logger import session_logger


@dataclass
class ChatSession:
    """Represents a single chat session"""
    session_id: str
    chat: Any
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    message_count: int = 0
    
    def update_activity(self):
        self.last_activity = datetime.now()
        self.message_count += 1


class SessionManager:
    """Manages multiple chat sessions"""
    
    def __init__(self, model):
        self.model = model
        self.sessions: Dict[str, ChatSession] = {}
        self.session_timeout_minutes = 60
        
        session_logger.session_logger.info("SessionManager initialized")
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        if session_id in self.sessions:
            self.sessions[session_id].update_activity()
            session_logger.log_session_created(
                session_id, 
                metadata={"status": "existing", "reused": True}
            )
            return session_id
        
        try:
            chat = self.model.start_chat(history=[])
            session = ChatSession(
                session_id=session_id,
                chat=chat
            )
            self.sessions[session_id] = session
            
            session_logger.log_session_created(
                session_id,
                metadata={
                    "status": "new",
                    "total_sessions": len(self.sessions)
                }
            )
            
            print(f"✅ Created new session: {session_id}")
            return session_id
            
        except Exception as e:
            session_logger.log_error("create_session", e, session_id)
            raise
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        session = self.sessions.get(session_id)
        if session:
            session.update_activity()
        return session
    
    def delete_session(self, session_id: str, reason: str = "manual") -> bool:
        if session_id in self.sessions:
            try:
                session = self.sessions[session_id]
                
                session_logger.log_session_deleted(session_id, reason)
                session_logger.session_logger.info(
                    f"Session {session_id[:8]}... stats - "
                    f"Messages: {session.message_count}, "
                    f"Duration: {(datetime.now() - session.created_at).total_seconds() / 60:.1f} minutes"
                )
                
                del self.sessions[session_id]
                print(f"🗑️ Deleted session: {session_id}")
                return True
                
            except Exception as e:
                session_logger.log_error("delete_session", e, session_id)
                return False
        return False
    
    def cleanup_expired_sessions(self):
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            time_diff = (current_time - session.last_activity).total_seconds() / 60
            if time_diff > self.session_timeout_minutes:
                expired_sessions.append((session_id, time_diff))
        
        for session_id, inactive_minutes in expired_sessions:
            session_logger.log_session_expired(session_id, inactive_minutes)
            self.delete_session(session_id, reason="expired")
        
        if expired_sessions:
            session_logger.log_cleanup(len(expired_sessions), len(self.sessions) + len(expired_sessions))
            print(f"🧹 Cleaned up {len(expired_sessions)} expired sessions")
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        return [
            {
                "session_id": session_id,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "message_count": session.message_count
            }
            for session_id, session in self.sessions.items()
        ]
    
    def get_session_count(self) -> int:
        return len(self.sessions)
    
    def get_chat_history(self, session_id: str) -> List:
        session = self.get_session(session_id)
        if session:
            return session.chat.history
        return []
    
    def reset_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            try:
                old_count = self.sessions[session_id].message_count
                self.sessions[session_id].chat = self.model.start_chat(history=[])
                self.sessions[session_id].message_count = 0
                self.sessions[session_id].update_activity()
                
                session_logger.log_session_reset(session_id)
                session_logger.session_logger.info(
                    f"Session {session_id[:8]}... reset - "
                    f"Previous messages: {old_count}"
                )
                
                print(f"🔄 Reset session: {session_id}")
                return True
                
            except Exception as e:
                session_logger.log_error("reset_session", e, session_id)
                return False
        return False

