import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from src.models.llm import get_llm
from src.services.vector_store import VectorStoreService
from src.utils.document_loader import DocumentLoader
from src.agents.router_graph import AgentRouterGraph  # ← Thay đổi import

load_dotenv()

st.set_page_config(
    page_title="AI Workshop - Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS để làm đẹp UI
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 0.5rem 0;
    }
    .progress-section {
        font-size: 0.85rem;
        margin: 0.5rem 0;
    }
    .progress-section h3 {
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: #1a1a1a;
        font-weight: 500;
    }
    .info-box.medical {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .info-box.doctor {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .info-box.medicine {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .warning-box {
        background-color: #fff9c4;
        border: 2px solid #f57c00;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        color: #e65100;
        font-weight: 600;
    }
    .warning-box strong {
        color: #bf360c;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_chatbot():
    """Khởi tạo chatbot"""
    try:
        use_unstructured = os.getenv('USE_UNSTRUCTURED', 'false').lower() == 'true'
        
        llm = get_llm(streaming=True)
        vector_service = VectorStoreService()
        document_loader = DocumentLoader(use_unstructured=use_unstructured)
        router = AgentRouterGraph(vector_service=vector_service)  # ← LangGraph Router
        
        chat_history = ChatMessageHistory()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Bạn là trợ lý y tế AI thông minh, thân thiện và hữu ích."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        chain = prompt | llm
        conversation = RunnableWithMessageHistory(
            chain,
            lambda session_id: chat_history,
            input_messages_key="input",
            history_messages_key="history"
        )
        
        try:
            vector_service.load_vector_store()
        except:
            folder_path = "./data/documents"
            if os.path.exists(folder_path):
                documents = document_loader.load_documents_from_folder(folder_path)
                if documents:
                    vector_service.create_vector_store(documents)
            
        return llm, vector_service, document_loader, router, chat_history, conversation
    except Exception as e:
        st.error(f"❌ Lỗi khởi tạo: {str(e)}")
        return None, None, None, None, None, None

def build_conversation_context(chat_history) -> str:
    """Tạo context từ lịch sử hội thoại"""
    if not chat_history or not chat_history.messages:
        return ""
    
    recent_messages = chat_history.messages[-6:]
    context_parts = []
    for msg in recent_messages:
        role = "Bệnh nhân" if msg.type == "human" else "Bác sĩ"
        context_parts.append(f"{role}: {msg.content}")
    
    return "\n".join(context_parts)

def load_documents():
    """Tải tài liệu từ thư mục"""
    folder_path = "./data/documents"
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return []
        
    documents = st.session_state.document_loader.load_documents_from_folder(folder_path)
    
    if documents:
        st.session_state.vector_service.create_vector_store(documents)
        return documents
    return []

def get_intent_icon_and_color(intent: str):
    """Lấy icon và màu theo intent"""
    intent_map = {
        "medical_consultation": ("🏥", "medical", "Tư vấn y tế"),
        "doctor_recommendation": ("👨‍⚕️", "doctor", "Gợi ý bác sĩ"),
        "medicine_inquiry": ("💊", "medicine", "Tư vấn thuốc"),
        "general_chat": ("💬", "general", "Trò chuyện")
    }
    return intent_map.get(intent, ("💬", "general", "Trò chuyện"))

def display_conversation_step():
    """Hiển thị bước hội thoại hiện tại"""
    num_messages = len(st.session_state.messages)
    
    if num_messages == 0:
        current_step = 0
    elif num_messages <= 2:
        current_step = 1
    elif num_messages <= 4:
        current_step = 2
    else:
        current_step = 3
    
    steps = ["🏁 Bắt đầu", "💬 Tư vấn", "👨‍⚕️ Gợi ý", "💊 Điều trị"]
    
    cols = st.columns(4)
    for i, step_name in enumerate(steps):
        with cols[i]:
            if i == current_step:
                st.write(f"**{step_name}**")
                st.progress(1.0)
            elif i < current_step:
                st.write(f"✅ {step_name}")
                st.progress(1.0)
            else:
                st.write(f"⚪ {step_name}")
                st.progress(0.0)

def main():
    st.markdown('<h1 class="main-header">🏥 AI Medical Assistant</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "user_messages_only" not in st.session_state:
        st.session_state.user_messages_only = []
    
    if "chatbot_initialized" not in st.session_state:
        with st.spinner("🚀 Đang khởi tạo AI Medical Assistant..."):
            llm, vector_service, document_loader, router, chat_history, conversation = init_chatbot()
            if llm:
                st.session_state.llm = llm
                st.session_state.vector_service = vector_service
                st.session_state.document_loader = document_loader
                st.session_state.router = router
                st.session_state.chat_history = chat_history
                st.session_state.conversation = conversation
                st.session_state.chatbot_initialized = True
            else:
                st.stop()
    
    # Sidebar
    with st.sidebar:
        if st.button("🗑️ Xóa lịch sử chat", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.session_state.chat_history.clear()
            st.session_state.user_messages_only = []
            st.rerun()
        
        st.markdown("---")
        st.header("⚙️ Cài đặt hệ thống")
        
        with st.expander("ℹ️ Thông tin hệ thống & Hành động", expanded=False):
            loader_mode = "Auto-detection" if os.getenv('USE_UNSTRUCTURED', 'false').lower() == 'true' else "Custom"
            st.info(f"📋 Loader: **{loader_mode}**")
            st.info(f"🤖 Model: **{os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'N/A')}**")
            st.info(f"💬 Tin nhắn: **{len(st.session_state.messages)}**")
            
            st.markdown("---")
            
            if st.button("📚 Tải lại tài liệu", use_container_width=True):
                with st.spinner("Đang tải tài liệu..."):
                    documents = load_documents()
                    if documents:
                        st.success(f"✅ Đã tải {len(documents)} tài liệu")
                    else:
                        st.info("📂 Không tìm thấy tài liệu mới")
        
        st.markdown("---")
        st.subheader("💡 Hướng dẫn sử dụng")
        st.markdown("""
        **Cách sử dụng:**
        1. 🗣️ Mô tả triệu chứng của bạn
        2. 👨‍⚕️ Hỏi về bác sĩ phù hợp
        3. 💊 Tư vấn về thuốc điều trị
        
        **Lưu ý:**
        - ⚠️ Thông tin chỉ mang tính tham khảo
        - 🏥 Luôn tham khảo bác sĩ trước khi điều trị
        - 💊 Không tự ý dùng thuốc
        """)
    
    # Main content
    st.markdown('<div class="progress-section">', unsafe_allow_html=True)
    st.markdown("#### 📊 Tiến trình tư vấn")
    display_conversation_step()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input
    prompt = st.chat_input("💬 Nhập câu hỏi của bạn...")
    
    if prompt:
        # Lưu tin nhắn gốc của user
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Đang trả lời..."):
                try:
                    conversation_context = build_conversation_context(st.session_state.chat_history)
                    
                    # QUAN TRỌNG: Chỉ lấy lịch sử TRƯỚC câu hỏi hiện tại
                    # Không bao gồm prompt hiện tại khi check symptoms
                    user_only_context = " ".join(st.session_state.user_messages_only[-6:])
                    
                    # Append sau khi đã lấy context
                    st.session_state.user_messages_only.append(prompt)
                    
                    routing_result = st.session_state.router.route(
                        prompt, 
                        conversation_context,
                        user_only_context
                    )
                    
                    icon, box_class, intent_name = get_intent_icon_and_color(routing_result["intent"])
                    
                    # Chỉ hiển thị info box cho các intent y tế (không hiển thị cho general_chat)
                    if routing_result["intent"] != "general_chat":
                        st.markdown(f"""
                        <div class="info-box {box_class}">
                            <strong>{icon} {intent_name}</strong><br>
                            {'✅ Sử dụng dữ liệu y tế' if routing_result['use_context'] else 'ℹ️ Trả lời tổng quát'}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Warning cho medicine inquiry
                    if routing_result["intent"] == "medicine_inquiry":
                        st.markdown("""
                        <div class="warning-box">
                            ⚠️ <strong>Lưu ý quan trọng:</strong> Thông tin thuốc chỉ mang tính tham khảo. 
                            Vui lòng tham khảo bác sĩ/dược sĩ trước khi sử dụng bất kỳ loại thuốc nào.
                        </div>
                        """, unsafe_allow_html=True)
                    
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", routing_result.get("system_prompt", "Bạn là trợ lý AI.")),
                        MessagesPlaceholder(variable_name="history"),
                        ("human", "{input}")
                    ])
                    
                    chain = prompt_template | st.session_state.llm
                    conversation = RunnableWithMessageHistory(
                        chain,
                        lambda session_id: st.session_state.chat_history,
                        input_messages_key="input",
                        history_messages_key="history"
                    )
                    
                    full_input = routing_result["prompt"]
                    
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    for chunk in conversation.stream(
                        {"input": full_input},
                        config={"configurable": {"session_id": "default"}}
                    ):
                        if hasattr(chunk, 'content'):
                            full_response += chunk.content
                            response_placeholder.markdown(full_response + "▌")
                    
                    response_placeholder.markdown(full_response)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "metadata": {
                            "intent": routing_result["intent"],
                            "use_context": routing_result["use_context"]
                        }
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Lỗi: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()

