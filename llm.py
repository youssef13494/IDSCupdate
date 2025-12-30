# ============================================
# FILE 2: llm.py
# ============================================
import os
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from rag import arabic_processor
from session_manager import SessionManager
from session_logger import session_logger

load_dotenv()

class GeminiRAGChat:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.session_manager = SessionManager(self.model)
        self.llm = None
        self.vector_store = None
        
        session_logger.session_logger.info("GeminiRAGChat initializing...")
        self._initialize_rag()
        session_logger.session_logger.info("GeminiRAGChat initialized successfully")
    
    def _initialize_rag(self):
        try:
            arabic_processor.initialize()
            
            collection_info = arabic_processor.get_collection_info()
            if collection_info["count"] == 0:
                print("No vector store found. Processing Word documents in data directory...")
                session_logger.rag_logger.info("No vector store found, processing data directory")
                documents = arabic_processor.process_directory("./data")
                if documents:
                    arabic_processor.create_vector_store(documents)
                    session_logger.rag_logger.info(f"Vector store created with {len(documents)} documents")
                else:
                    print("No Word documents found in data directory")
                    session_logger.rag_logger.warning("No Word documents found in data directory")
            else:
                if not collection_info.get("has_bm25", False):
                    print("BM25 index not found. Creating from existing documents...")
                    session_logger.rag_logger.info("Creating BM25 index from existing documents")
                    vector_store = arabic_processor.load_vector_store()
                    if vector_store:
                        all_points = arabic_processor.client.scroll(
                            collection_name=arabic_processor.collection_name,
                            limit=10000,
                            with_payload=True,
                            with_vectors=False
                        )[0]
                        
                        if all_points:
                            from langchain_core.documents import Document
                            all_documents = [
                                Document(
                                    page_content=point.payload.get('page_content', ''),
                                    metadata=point.payload.get('metadata', {})
                                )
                                for point in all_points
                            ]
                            arabic_processor._create_bm25_index(all_documents)
                            session_logger.rag_logger.info(f"BM25 index created with {len(all_documents)} documents")
            
            self.vector_store = arabic_processor.load_vector_store()
            
            if self.vector_store:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=os.getenv("GEMINI_API_KEY"),
                    temperature=0.1,
                    convert_system_message_to_human=True
                )
                
                collection_info = arabic_processor.get_collection_info()
                print("✅ RAG system initialized successfully")
                print(f"📊 Hybrid Search: {collection_info.get('has_bm25', False)}")
                print(f"📚 Documents: {collection_info['count']}")
                
                session_logger.log_system_status({
                    "document_count": collection_info['count'],
                    "has_vector_store": True,
                    "has_bm25": collection_info.get('has_bm25', False),
                    "active_sessions": 0
                })
                
        except Exception as e:
            print(f"Error initializing RAG: {str(e)}")
            session_logger.log_error("initialize_rag", e)
            self.vector_store = None

    def create_session(self, session_id: Optional[str] = None) -> str:
        return self.session_manager.create_session(session_id)
    
    def get_response(self, message: str, session_id: str, use_rag: bool = True) -> Dict[str, Any]:
        try:
            session_logger.log_chat_message(
                session_id, 
                "user", 
                message, 
                use_rag=use_rag
            )
            
            session = self.session_manager.get_session(session_id)
            if not session:
                session_id = self.session_manager.create_session(session_id)
                session = self.session_manager.get_session(session_id)
            
            if use_rag and self.vector_store and self.llm:
                print("\n" + "="*50)
                print(f"🚀 HYBRID SEARCH - Session: {session_id[:8]}...")
                print("="*50)
                
                relevant_docs = arabic_processor.search_documents(
                    query=message,
                    k=40
                )
                
                session_logger.log_rag_search(
                    session_id,
                    message,
                    len(relevant_docs),
                    "Hybrid Search"
                )
                
                if not relevant_docs:
                    response = session.chat.send_message(message)
                    
                    session_logger.log_chat_message(
                        session_id,
                        "assistant",
                        response.text,
                        use_rag=False,
                        sources_count=0
                    )
                    
                    return {
                        "response": response.text,
                        "sources": [],
                        "is_rag": False,
                        "success": True,
                        "session_id": session_id
                    }
                
                context = "\n\n".join([
                    f"[مصدر {i+1} - {doc.metadata.get('source', 'Unknown')} - صفحة {doc.metadata.get('page', 'غير محدد')}]\n{doc.page_content}"
                    for i, doc in enumerate(relevant_docs)
                ])
                
                prompt_template = """أنت مساعد ذكي يتحدث العربية بطلاقة. استخدم المعلومات التالية للإجابة على السؤال بدقة.

قواعد مهمة:
- أجب باللغة العربية فقط
- استخدم فقط المعلومات الموجودة في السياق
- إذا لم تجد الإجابة في السياق، قل "لا أجد معلومات كافية للإجابة"
- كن دقيقاً ومختصراً
- اذكر المصدر عند الاقتباس

السياق:
{context}

السؤال: {question}

الإجابة:"""

                prompt = PromptTemplate(
                    template=prompt_template,
                    input_variables=["context", "question"]
                )
                
                formatted_prompt = prompt.format(context=context, question=message)
                response = self.llm.invoke(formatted_prompt)
                
                source_docs = []
                for doc in relevant_docs:
                    source_docs.append({
                        "content": doc.page_content[:400] + "...",
                        "source": doc.metadata.get("source", "Unknown"),
                        "chunk": doc.metadata.get("chunk", 0),
                        "page": doc.metadata.get("page", "غير محدد")
                    })
                
                session_logger.log_chat_message(
                    session_id,
                    "assistant",
                    response.content,
                    use_rag=True,
                    sources_count=len(source_docs)
                )
                
                print("="*50)
                print(f"✅ Response generated with {len(source_docs)} sources")
                print("="*50 + "\n")
                
                return {
                    "response": response.content,
                    "sources": source_docs,
                    "is_rag": True,
                    "success": True,
                    "session_id": session_id
                }
            else:
                response = session.chat.send_message(message)
                
                session_logger.log_chat_message(
                    session_id,
                    "assistant",
                    response.text,
                    use_rag=False
                )
                
                return {
                    "response": response.text,
                    "sources": [],
                    "is_rag": False,
                    "success": True,
                    "session_id": session_id
                }
                
        except Exception as e:
            print(f"❌ Error in get_response: {str(e)}")
            session_logger.log_error("get_response", e, session_id)
            return {
                "response": f"عذراً، حدث خطأ: {str(e)}",
                "sources": [],
                "is_rag": False,
                "success": False,
                "session_id": session_id
            }
    
    def process_new_docx(self, docx_path: str, metadata: Optional[dict] = None):
        try:
            session_logger.rag_logger.info(f"Processing document: {os.path.basename(docx_path)}")
            
            documents = arabic_processor.process_docx(docx_path, metadata)
            if documents:
                if self.vector_store:
                    self.vector_store.add_documents(documents)
                    
                    all_points = arabic_processor.client.scroll(
                        collection_name=arabic_processor.collection_name,
                        limit=10000,
                        with_payload=True,
                        with_vectors=False
                    )[0]
                    
                    if all_points:
                        from langchain_core.documents import Document
                        all_documents = [
                            Document(
                                page_content=point.payload.get('page_content', ''),
                                metadata=point.payload.get('metadata', {})
                            )
                            for point in all_points
                        ]
                        arabic_processor._create_bm25_index(all_documents)
                    
                    session_logger.log_document_processed(
                        os.path.basename(docx_path),
                        len(documents),
                        success=True
                    )
                    
                    return True, f"✅ Added {len(documents)} chunks from {os.path.basename(docx_path)}"
                else:
                    arabic_processor.create_vector_store(documents)
                    self._initialize_rag()
                    
                    session_logger.log_document_processed(
                        os.path.basename(docx_path),
                        len(documents),
                        success=True
                    )
                    
                    return True, f"✅ Created vector store with {len(documents)} chunks"
            else:
                session_logger.log_document_processed(
                    os.path.basename(docx_path),
                    0,
                    success=False
                )
                return False, "❌ No text extracted from Word document"
                
        except Exception as e:
            session_logger.log_error("process_new_docx", e)
            session_logger.log_document_processed(
                os.path.basename(docx_path),
                0,
                success=False
            )
            return False, f"❌ Error processing Word document: {str(e)}"
    
    def get_chat_history(self, session_id: str):
        return self.session_manager.get_chat_history(session_id)
    
    def reset_chat(self, session_id: str) -> bool:
        return self.session_manager.reset_session(session_id)
    
    def delete_session(self, session_id: str) -> bool:
        return self.session_manager.delete_session(session_id)
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        return self.session_manager.get_all_sessions()
    
    def cleanup_sessions(self):
        self.session_manager.cleanup_expired_sessions()
    
    def get_rag_status(self):
        collection_info = arabic_processor.get_collection_info()
        has_vector_store = collection_info["count"] > 0
        has_bm25 = collection_info.get("has_bm25", False)
        
        status = {
            "has_vector_store": has_vector_store,
            "document_count": collection_info["count"],
            "has_bm25": has_bm25,
            "rag_chain_initialized": self.vector_store is not None,
            "active_sessions": self.session_manager.get_session_count(),
            "features": {
                "hybrid_search": has_bm25 and has_vector_store,
                "reranking": False,
                "mmr": False
            }
        }
        
        if self.session_manager.get_session_count() > 0:
            session_logger.log_system_status(status)
        
        return status

gemini_chat = GeminiRAGChat()
