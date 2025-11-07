import os
import sys
from typing import List
from langchain.schema import Document
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from src.models.llm import get_llm, get_embeddings
from src.services.vector_store import VectorStoreService
from src.utils.document_loader import DocumentLoader
from src.agents.router_graph import AgentRouterGraph  # ← Thay đổi import


load_dotenv()

class AIWorkshopChatbot:
    """Chatbot sử dụng RAG với Azure OpenAI và LangChain"""

    def __init__(self):
        """Khởi tạo chatbot"""
        try:
            required_env_vars = [
                'AZURE_OPENAI_API_KEY',
                'AZURE_OPENAI_ENDPOINT', 
                'AZURE_OPENAI_DEPLOYMENT_NAME'
            ]
            
            missing_vars = [var for var in required_env_vars if not os.getenv(var)]
            if missing_vars:
                raise ValueError(f"Thiếu biến môi trường: {', '.join(missing_vars)}")
            
            # Đọc cấu hình use_unstructured từ .env
            use_unstructured = os.getenv('USE_UNSTRUCTURED', 'false').lower() == 'true'
            
            self.llm = get_llm(streaming=True)
            self.vector_service = VectorStoreService()
            self.document_loader = DocumentLoader(use_unstructured=use_unstructured)
            self.router = AgentRouterGraph(vector_service=self.vector_service)  # ← LangGraph Router
            self.user_messages_only = []
            
            # Sử dụng approach mới của LangChain
            self.chat_history = ChatMessageHistory()
            
            # Tạo prompt template với history
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", "Bạn là trợ lý AI thông minh, thân thiện và hữu ích."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}")
            ])
            
            # Tạo chain với message history
            self.chain = self.prompt | self.llm
            self.conversation = RunnableWithMessageHistory(
                self.chain,
                lambda session_id: self.chat_history,
                input_messages_key="input",
                history_messages_key="history"
            )
            
            self.conversation_history = []
            
            try:
                self.vector_service.load_vector_store()
                print("✅ Đã load vector store")
            except Exception:
                print("⚠️ Chưa có vector store, đang kiểm tra documents...")
                self.auto_load_documents()
                
        except Exception as e:
            print(f"❌ Lỗi khởi tạo: {str(e)}")
            sys.exit(1)

    def auto_load_documents(self):
        """Tự động load và tạo vector store từ documents"""
        folder_path = "./data/documents"
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print("📁 Đã tạo thư mục data/documents")
            return
        
        documents = self.document_loader.load_documents_from_folder(folder_path)
        
        if documents:
            print(f"📚 Đang tạo vector store từ {len(documents)} tài liệu...")
            self.vector_service.create_vector_store(documents)
            print("✅ Đã tạo vector store thành công")
        else:
            print("📂 Không có tài liệu. Thêm file vào data/documents")

    def load_documents_from_folder(self, folder_path: str = "./data/documents"):
        try:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                return []
                
            documents = self.document_loader.load_documents_from_folder(folder_path)
            
            if documents:
                self.vector_service.create_vector_store(documents)
                print(f"📚 Đã tải {len(documents)} tài liệu")
                
            return documents
            
        except Exception as e:
            print(f"❌ Lỗi tải tài liệu: {str(e)}")
            return []

    def get_context_from_query(self, query: str, k: int = 3) -> str:
        try:
            if not self.vector_service.vector_store:
                return ""
                
            relevant_docs = self.vector_service.similarity_search(query, k=k)
            if relevant_docs:
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                return f"Thông tin tham khảo:\n{context}\n\n"
            return ""
            
        except Exception as e:
            if "key_model_access_denied" in str(e):
                print(f"❌ Lỗi embedding model: {os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME', 'không tìm thấy')}")
            return ""

    def _build_conversation_context(self) -> str:
        """Tạo context từ lịch sử hội thoại"""
        if not self.chat_history.messages:
            return ""
        
        # Lấy 6 tin nhắn gần nhất (3 cặp hỏi-đáp)
        recent_messages = self.chat_history.messages[-6:]
        
        context_parts = []
        for msg in recent_messages:
            role = "Bệnh nhân" if msg.type == "human" else "Bác sĩ"
            context_parts.append(f"{role}: {msg.content}")
        
        return "\n".join(context_parts)

    def _build_user_messages_only(self) -> str:
        """Tạo context CHỈ từ tin nhắn của USER"""
        if not self.chat_history.messages:
            return ""
        
        recent_messages = self.chat_history.messages[-6:]
        user_messages = []
        
        for msg in recent_messages:
            if msg.type == "human":
                user_messages.append(msg.content)
        
        return " ".join(user_messages)

    def chat(self, user_input: str) -> str:
        """Xử lý chat với RAG và streaming"""
        try:
            # Lưu tin nhắn user gốc
            self.user_messages_only.append(user_input)
            
            conversation_context = self._build_conversation_context()
            user_only_context = " ".join(self.user_messages_only[-6:])  # 6 tin nhắn gần nhất
            
            print(f"🔍 USER ONLY CONTEXT: '{user_only_context}'")
            
            routing_result = self.router.route(
                user_input, 
                conversation_context,
                user_only_context
            )
            
            # Cập nhật system prompt
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", routing_result.get("system_prompt", "Bạn là trợ lý AI thông minh, thân thiện và hữu ích.")),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}")
            ])
            
            self.chain = self.prompt | self.llm
            self.conversation = RunnableWithMessageHistory(
                self.chain,
                lambda session_id: self.chat_history,
                input_messages_key="input",
                history_messages_key="history"
            )
            
            full_input = routing_result["prompt"]
            
            # Stream response
            full_response = ""
            for chunk in self.conversation.stream(
                {"input": full_input},
                config={"configurable": {"session_id": "default"}}
            ):
                if hasattr(chunk, 'content'):
                    content = chunk.content
                    full_response += content
                    # Print từng chunk ra màn hình
                    print(content, end='', flush=True)
            
            print()  # Xuống dòng sau khi stream xong
            
            self.conversation_history.append({
                "user": user_input,
                "assistant": full_response,
                "intent": routing_result["intent"]
            })
            
            return full_response
            
        except Exception as e:
            return f"Lỗi: {str(e)}"

    def clear_memory(self):
        """Xóa lịch sử hội thoại"""
        self.chat_history.clear()
        self.conversation_history = []
        self.user_messages_only = []  # ← Xóa cả user_messages_only
        print("✅ Đã xóa lịch sử hội thoại")

    def get_stats(self) -> str:
        doc_count = 0
        if self.vector_service.vector_store:
            try:
                doc_count = len(self.vector_service.vector_store.get()['ids']) if hasattr(self.vector_service.vector_store, 'get') else "N/A"
            except:
                doc_count = "N/A"
                
        return f"""
📊 Thống kê:
• Tài liệu: {doc_count}
• Cuộc hội thoại: {len(self.conversation_history)}
• Model: {os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'N/A')}
"""


def print_welcome():
    print("\n🤖 AI WORKSHOP - RAG CHATBOT")
    print("Lệnh: /exit, /clear, /help, /load, /stats\n")


def main():
    try:
        print_welcome()
        chatbot = AIWorkshopChatbot()
        chatbot.load_documents_from_folder()

        while True:
            try:
                user_input = input("👤 Bạn: ").strip()
                if not user_input:
                    continue

                if user_input.lower() in ['/exit', '/quit', 'exit', 'quit']:
                    print("👋 Tạm biệt!")
                    break
                elif user_input.lower() in ['/clear', 'clear']:
                    os.system('cls' if os.name == 'nt' else 'clear')
                    chatbot.clear_memory()
                    print_welcome()
                    continue
                elif user_input.lower() in ['/help', 'help']:
                    print_welcome()
                    continue
                elif user_input.lower() in ['/reload', 'reload']:
                    print("\n🔄 Đang tải lại tài liệu...")
                    documents = chatbot.load_documents_from_folder()
                    continue
                elif user_input.lower() in ['/stats', 'stats']:
                    print(chatbot.get_stats())
                    continue

                print("\n🤖 AI: ", end="", flush=True)
                chatbot.chat(user_input)
                print()

            except KeyboardInterrupt:
                print("\n👋 Tạm biệt!")
                break
            except Exception as e:
                print(f"❌ Lỗi: {str(e)}")

    except Exception as e:
        print(f"❌ Lỗi khởi động: {str(e)}")


if __name__ == "__main__":
    main()

