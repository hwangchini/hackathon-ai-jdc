from enum import Enum
from typing import Dict, Any, Optional, Literal
from langgraph.graph import StateGraph, END
from src.models.llm import get_llm
from src.agents.medicine_agent import MedicineAgent
from src.agents.graph_state import GraphState
from src.tools.medical_tools import MedicalTools


class IntentType(Enum):
    """Các loại intent trong cuộc hội thoại"""
    MEDICAL_CONSULTATION = "medical_consultation"
    DOCTOR_RECOMMENDATION = "doctor_recommendation"
    MEDICINE_INQUIRY = "medicine_inquiry"
    GENERAL_CHAT = "general_chat"


class AgentRouterGraph:
    """Agent Router sử dụng LangGraph + Tools"""
    
    def __init__(self, vector_service=None):
        self.llm = get_llm(streaming=False)
        self.vector_service = vector_service
        self.medicine_agent = MedicineAgent(vector_service)
        
        # ✅ Initialize tools
        if vector_service:
            self.medical_tools = MedicalTools(vector_service)
            self.tools = self.medical_tools.get_all_tools()
        else:
            self.tools = []
        
        # Build graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Xây dựng LangGraph workflow"""
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("classify_intent", self.classify_intent_node)
        workflow.add_node("check_symptoms", self.check_symptoms_node)
        workflow.add_node("get_medical_context", self.get_medical_context_node)
        workflow.add_node("get_doctor_context", self.get_doctor_context_node)
        workflow.add_node("get_medicine_context", self.get_medicine_context_node)
        workflow.add_node("build_response", self.build_response_node)
        
        # Set entry point
        workflow.set_entry_point("classify_intent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "classify_intent",
            self.route_by_intent,
            {
                "medical_consultation": "get_medical_context",
                "doctor_recommendation": "check_symptoms",
                "medicine_inquiry": "check_symptoms",
                "general_chat": "build_response"
            }
        )
        
        workflow.add_conditional_edges(
            "check_symptoms",
            self.route_by_symptoms,
            {
                "has_symptoms_doctor": "get_doctor_context",
                "has_symptoms_medicine": "get_medicine_context",
                "no_symptoms": "build_response"
            }
        )
        
        workflow.add_edge("get_medical_context", "build_response")
        workflow.add_edge("get_doctor_context", "build_response")
        workflow.add_edge("get_medicine_context", "build_response")
        workflow.add_edge("build_response", END)
        
        return workflow.compile()
    
    # ==================== NODES ====================
    
    def classify_intent_node(self, state: GraphState) -> GraphState:
        """Node: Phân loại intent bằng LLM (không dùng keyword)"""
        user_message = state["user_message"]
        
        # Sử dụng LLM để phân loại với prompt chi tiết
        prompt = f"""Phân tích câu hỏi của người dùng và xác định intent (mục đích).

Câu hỏi: "{user_message}"

**Các loại intent:**

1. **MEDICAL_CONSULTATION** - Tư vấn y tế, phân tích triệu chứng
   - Người dùng MÔ TẢ triệu chứng đang gặp
   - Hỏi về nguyên nhân, chẩn đoán bệnh
   - VD: "tôi bị đau đầu", "con tôi sốt cao", "triệu chứng này là gì?"

2. **DOCTOR_RECOMMENDATION** - Gợi ý bác sĩ, chuyên khoa
   - Hỏi về bác sĩ, phòng khám, chuyên khoa
   - Muốn tìm bác sĩ để khám
   - VD: "bác sĩ nào giỏi?", "tôi nên đi khám ở đâu?", "gợi ý bác sĩ"

3. **MEDICINE_INQUIRY** - Hỏi về thuốc, liều dùng
   - Hỏi về thuốc điều trị
   - Liều lượng, cách dùng thuốc
   - VD: "tôi nên uống thuốc gì?", "liều dùng paracetamol?", "thuốc này có tác dụng phụ không?"

4. **GENERAL_CHAT** - Trò chuyện thông thường
   - Chào hỏi, cảm ơn, xin lỗi
   - Hỏi về AI, hệ thống
   - VD: "xin chào", "cảm ơn", "bạn là ai?"

**Hãy phân tích và CHỈ trả lời TÊN intent (một trong 4 loại trên):**

Intent:"""
        
        try:
            response = self.llm.invoke(prompt)
            intent_text = response.content.strip().upper()
            
            # Parse response
            if "MEDICINE_INQUIRY" in intent_text or "MEDICINE" in intent_text:
                intent = IntentType.MEDICINE_INQUIRY.value
            elif "DOCTOR_RECOMMENDATION" in intent_text or "DOCTOR" in intent_text:
                intent = IntentType.DOCTOR_RECOMMENDATION.value
            elif "MEDICAL_CONSULTATION" in intent_text or "MEDICAL" in intent_text:
                intent = IntentType.MEDICAL_CONSULTATION.value
            else:
                intent = IntentType.GENERAL_CHAT.value
            
            print(f"🎯 Intent: {intent} (LLM classified)")
        except Exception as e:
            print(f"⚠️ LLM classification error: {str(e)}")
            intent = IntentType.GENERAL_CHAT.value
        
        state["intent"] = intent
        return state
    
    def check_symptoms_node(self, state: GraphState) -> GraphState:
        """Node: Kiểm tra triệu chứng bằng LLM"""
        check_context = state.get("user_only_context", "") or state.get("conversation_context", "")
        
        if not check_context or len(check_context.strip()) < 5:
            state["has_symptoms"] = False
            return state
        
        prompt = f"""Phân tích: Người dùng đã MÔ TẢ triệu chứng bệnh lý hay chưa?

Lịch sử tin nhắn: "{check_context}"

QUY TẮC PHÂN BIỆT:

✅ CÓ triệu chứng (người dùng MÔ TẢ tình trạng sức khỏe):
- "tôi bị đau đầu"
- "con tôi sốt 39 độ"
- "tôi đang ho, khó thở"
- "bụng tôi đau quặn"
- "tôi cảm thấy chóng mặt"

❌ KHÔNG có triệu chứng (chỉ HỎI, CHƯA MÔ TẢ):
- "tôi nên uống thuốc gì?"
- "gợi ý bác sĩ cho tôi"
- "đau đầu là bệnh gì?"
- "bác sĩ nào giỏi?"
- "thuốc gì tốt?"

QUAN TRỌNG: 
- Người dùng phải MÔ TẢ rõ ràng họ ĐANG gặp triệu chứng gì
- Chỉ HỎI về thuốc/bác sĩ MÀ KHÔNG nói triệu chứng = CHƯA có

CHỈ trả lời: "CÓ" hoặc "KHÔNG"

Trả lời:"""
        
        try:
            response = self.llm.invoke(prompt)
            answer = response.content.strip().upper()
            has_symptoms = "CÓ" in answer or "CO" in answer
            
            print(f"🤖 LLM判断: '{answer}' → Has symptoms: {has_symptoms}")
            state["has_symptoms"] = has_symptoms
        except Exception as e:
            print(f"⚠️ LLM error: {str(e)}")
            state["has_symptoms"] = False
        
        return state
    
    def get_medical_context_node(self, state: GraphState) -> GraphState:
        """Node: Lấy context y tế"""
        if not self.vector_service or not self.vector_service.vector_store:
            state["medical_context"] = None
            return state
        
        try:
            docs = self.vector_service.similarity_search(state["user_message"], k=3)
            
            if docs:
                context_parts = []
                for i, doc in enumerate(docs, 1):
                    name = doc.metadata.get('symptom_name', doc.metadata.get('item_name', f'Doc {i}'))
                    context_parts.append(f"{'='*60}\n{name.upper()}\n{'='*60}\n{doc.page_content}")
                
                context = "\n\n".join(context_parts)
                state["medical_context"] = f"THÔNG TIN Y TẾ:\n\n{context}\n\n{'='*60}\n"
            else:
                state["medical_context"] = None
        except Exception as e:
            print(f"⚠️ Error: {str(e)}")
            state["medical_context"] = None
        
        return state
    
    def normalize_text(self, text: str) -> str:
        """Chuẩn hóa text để search tốt hơn"""
        import unicodedata
        text = unicodedata.normalize('NFKD', text)
        text = ''.join([c for c in text if not unicodedata.combining(c)])
        return text.lower().strip()
    
    def get_doctor_recommendations_logic(self, user_message: str, conversation_context: str = "") -> Optional[str]:
        """Logic tìm bác sĩ (di chuyển từ router.py)"""
        if not self.vector_service or not self.vector_service.vector_store:
            return None
        
        try:
            # Trích xuất triệu chứng
            symptoms_text = ""
            if conversation_context:
                extract_prompt = f"""Từ lịch sử, liệt kê triệu chứng:
{conversation_context}

Triệu chứng:"""
                response = self.llm.invoke(extract_prompt)
                symptoms_text = response.content.strip()
            
            # Map triệu chứng → chuyên khoa
            symptom_to_specialty = {
                'đau đầu': ['Nội khoa', 'Tim mạch', 'Nội tiết'],
                'đau bụng': ['Tiêu hóa', 'Nội khoa'],
                'ợ nóng': ['Tiêu hóa'],
                'tiêu chảy': ['Tiêu hóa'],
                'táo bón': ['Tiêu hóa'],
                'đau ngực': ['Tim mạch', 'Nội khoa'],
                'khó thở': ['Tim mạch', 'Hồi sức tích cực'],
                'ho': ['Tai-Mũi-Họng'],
                'sổ mũi': ['Tai-Mũi-Họng'],
                'đau họng': ['Tai-Mũi-Họng'],
                'mờ mắt': ['Mắt'],
                'ngứa': ['Da liễu'],
                'phát ban': ['Da liễu'],
            }
            
            possible_specialties = []
            symptoms_lower = symptoms_text.lower()
            for symptom, specialties in symptom_to_specialty.items():
                if symptom in symptoms_lower:
                    possible_specialties.extend(specialties)
            
            possible_specialties = list(set(possible_specialties))
            
            if not possible_specialties:
                specialty_prompt = f"""Triệu chứng: {symptoms_text}
Chọn khoa: Tim mạch, Tiêu hóa, Nội tiết, Tai-Mũi-Họng, Mắt, Da liễu
Chỉ trả về TÊN KHOA:"""
                response = self.llm.invoke(specialty_prompt)
                possible_specialties = [response.content.strip()]
            
            # Search với cosine similarity
            all_results_with_scores = []
            for specialty in possible_specialties:
                queries = [
                    specialty,
                    f"khoa {specialty}",
                    f"bác sĩ {specialty}",
                    self.normalize_text(specialty)
                ]
                
                for query in queries:
                    results = self.vector_service.similarity_search_with_scores(query, k=3)
                    all_results_with_scores.extend(results)
            
            # Lọc và rank
            dept_scores = {}
            for doc, cosine_score in all_results_with_scores:
                if doc.metadata.get('filename') == 'medical_personnel.json':
                    dept_name = doc.metadata.get('department_name')
                    if dept_name:
                        total_score = cosine_score
                        
                        # Bonus từ text matching
                        dept_lower = dept_name.lower()
                        specialty_lower = doc.metadata.get('specialty_name', '').lower()
                        
                        for spec in possible_specialties:
                            if spec.lower() in dept_lower:
                                total_score += 0.2
                            if spec.lower() in specialty_lower:
                                total_score += 0.1
                        
                        if dept_name not in dept_scores or dept_scores[dept_name]['score'] < total_score:
                            dept_scores[dept_name] = {
                                'doc': doc,
                                'score': total_score,
                                'cosine_score': cosine_score
                            }
            
            # Sort và format
            sorted_depts = sorted(dept_scores.items(), key=lambda x: x[1]['score'], reverse=True)
            doctor_docs = [item[1]['doc'] for item in sorted_depts[:3]]
            
            if doctor_docs:
                context_parts = []
                for doc in doctor_docs:
                    specialty_name = doc.metadata.get('specialty_name', 'N/A')
                    dept_name = doc.metadata.get('department_name', 'N/A')
                    context_parts.append(f"{'='*60}\n{dept_name.upper()} - {specialty_name}\n{'='*60}\n{doc.page_content}")
                
                context = "\n\n".join(context_parts)
                return f"THÔNG TIN BÁC SĨ:\n\n{context}\n\n{'='*60}\n"
            
            return None
            
        except Exception as e:
            print(f"⚠️ Lỗi: {str(e)}")
            return None
    
    def get_doctor_context_node(self, state: GraphState) -> GraphState:
        """Node: Lấy context bác sĩ - Hybrid approach"""
        
        # ✅ OPTION 1: Try using tool first (fast & accurate)
        if self.tools and state.get("user_message"):
            try:
                # Extract specialty from conversation
                extract_prompt = f"""Từ câu hỏi, xác định chuyên khoa:
{state['user_message']}

Lịch sử: {state['conversation_context']}

Chỉ trả về TÊN CHUYÊN KHOA (Tim mạch, Tiêu hóa, Tai-Mũi-Họng, Da liễu, Mắt, Nội tiết)
Nếu không rõ, trả về "Nội khoa"

Chuyên khoa:"""
                
                response = self.llm.invoke(extract_prompt)
                specialty = response.content.strip()
                
                print(f"🔧 Using tool: search_doctors_by_specialty('{specialty}')")
                
                # Use tool
                tool_result = self.medical_tools.search_doctors_by_specialty(specialty)
                
                if tool_result and "Không tìm thấy" not in tool_result:
                    doctor_context = f"THÔNG TIN BÁC SĨ:\n\n{tool_result}\n\n{'='*60}\n"
                    state["doctor_context"] = doctor_context
                    print(f"✅ Tool returned doctor context")
                    return state
                    
            except Exception as e:
                print(f"⚠️ Tool failed: {str(e)}, fallback to original method")
        
        # ✅ OPTION 2: Fallback to original vector search
        print(f"🔄 Fallback to original vector search")
        doctor_context = self.get_doctor_recommendations_logic(
            state["user_message"],
            state["conversation_context"]
        )
        
        state["doctor_context"] = doctor_context
        return state
    
    def get_medicine_context_node(self, state: GraphState) -> GraphState:
        """Node: Lấy context thuốc"""
        medicine_context = self.medicine_agent.search_medicine_by_symptoms(
            state["user_message"],
            state["conversation_context"]
        )
        
        # ✅ DEBUG
        if medicine_context:
            print(f"✅ Medicine context received: {len(medicine_context)} chars")
        else:
            print(f"❌ Medicine context is None or empty")
        
        state["medicine_context"] = medicine_context
        return state
    
    def build_response_node(self, state: GraphState) -> GraphState:
        """Node: Xây dựng response cuối cùng"""
        intent = state["intent"]
        
        if intent == "medical_consultation":
            if state.get("medical_context"):
                state["use_context"] = True
                state["system_prompt"] = """Bạn là trợ lý y tế AI chuyên nghiệp. 
Nhiệm vụ:
1. GHI NHỚ tất cả triệu chứng
2. Phân tích triệu chứng
3. Chẩn đoán khả năng bệnh lý
4. Đưa ra lời khuyên

LƯU Ý: Đây chỉ là thông tin tham khảo."""
                state["prompt"] = f"""{state['medical_context']}

Câu hỏi: {state['user_message']}

Phân tích và tư vấn:"""
            else:
                state["use_context"] = False
                state["system_prompt"] = "Bạn là trợ lý y tế AI."
                state["prompt"] = f"{state['user_message']}\n\nKHUYẾN NGHỊ gặp bác sĩ."
        
        elif intent == "doctor_recommendation":
            if not state.get("has_symptoms"):
                state["use_context"] = False
                state["system_prompt"] = "Bạn là trợ lý y tế AI. KHÔNG tự bịa triệu chứng."
                state["prompt"] = f"""Người dùng hỏi: {state['user_message']}

QUAN TRỌNG: Người dùng CHƯA cung cấp triệu chứng cụ thể.

Hãy trả lời:
"Để gợi ý bác sĩ phù hợp, tôi cần biết thêm thông tin về tình trạng sức khỏe của bạn.

Vui lòng cho tôi biết:
- Bạn đang gặp triệu chứng gì?
- Triệu chứng xuất hiện từ bao lâu?
- Mức độ nghiêm trọng như thế nào?"

KHÔNG tự bịa triệu chứng."""
            elif state.get("doctor_context"):
                state["use_context"] = True
                state["system_prompt"] = """Bạn là trợ lý tư vấn bác sĩ.
QUY TẮC:
- CHỈ dùng triệu chứng từ lịch sử
- KHÔNG tự bịa
- PHẢI đúng chuyên khoa"""
                state["prompt"] = f"""Lịch sử: {state['conversation_context']}

{state['doctor_context']}

Câu hỏi: {state['user_message']}

Format:
**Triệu chứng**: [Từ lịch sử]
**Chuyên khoa**: [Tên]
**Bác sĩ**:
1. [Họ tên] - [Học vị] - [Chức vụ] - [Khoa]"""
            else:
                state["use_context"] = False
                state["system_prompt"] = "Bạn là trợ lý y tế."
                state["prompt"] = f"""Không tìm thấy bác sĩ.

Hãy khuyên:
1. Mô tả rõ triệu chứng
2. Gọi 115 hoặc 19003115"""
        
        elif intent == "medicine_inquiry":
            if not state.get("has_symptoms"):
                state["use_context"] = False
                state["system_prompt"] = "Bạn là dược sĩ AI. KHÔNG tự bịa triệu chứng."
                state["prompt"] = f"""Người dùng hỏi: {state['user_message']}

QUAN TRỌNG: Người dùng CHƯA cung cấp triệu chứng cụ thể.

Hãy trả lời:
"Để gợi ý thuốc và liều lượng sử dụng, chế độ nghỉ ngơi phù hợp, tôi cần biết thêm thông tin về tình trạng sức khỏe của bạn.

Vui lòng cho tôi biết:
- Bạn đang gặp triệu chứng gì?
- Triệu chứng xuất hiện từ bao lâu?
- Mức độ nghiêm trọng như thế nào?"

KHÔNG tự bịa triệu chứng hoặc tư vấn thuốc."""
            elif state.get("medicine_context"):
                # ✅ DEBUG
                print(f"✅ Using medicine context in response")
                
                state["use_context"] = True
                state["system_prompt"] = """Bạn là dược sĩ AI.
QUY TẮC:
- CHỈ tư vấn OTC
- Cảnh báo tác dụng phụ
- Khuyên tham khảo bác sĩ"""
                state["prompt"] = f"""{state['medicine_context']}

Câu hỏi: {state['user_message']}

Tư vấn thuốc và khuyến nghị:"""
            else:
                # ✅ DEBUG
                print(f"❌ No medicine context, using fallback response")
                
                state["use_context"] = False
                state["system_prompt"] = "Bạn là dược sĩ AI thân thiện và lịch sự."
                state["prompt"] = f"""Người dùng hỏi: {state['user_message']}

QUAN TRỌNG: Không tìm thấy thông tin thuốc phù hợp trong cơ sở dữ liệu.

Hãy trả lời một cách lịch sự và hữu ích:
"Xin lỗi, hiện tại tôi chưa có thông tin chi tiết về thuốc phù hợp cho triệu chứng của bạn trong cơ sở dữ liệu của mình.

Để được tư vấn chính xác về thuốc và liều lượng phù hợp, tôi khuyên bạn:

1. **Đến phòng khám hoặc bệnh viện gần nhất** để được bác sĩ khám và kê đơn thuốc phù hợp
2. **Tham khảo dược sĩ tại nhà thuốc** để được tư vấn trực tiếp về thuốc không kê đơn
3. **Gọi tổng đài tư vấn y tế**: 
   - Tổng đài 115 (cấp cứu)
   - Hotline tư vấn dược: 19003190

**Lưu ý quan trọng:** Không tự ý mua và sử dụng thuốc mà chưa có chỉ định của bác sĩ hoặc dược sĩ, vì có thể gây ra tác dụng phụ không mong muốn."

Hãy thể hiện sự quan tâm và hỗ trợ tối đa có thể."""
        
        else:  # general_chat
            state["use_context"] = False
            state["system_prompt"] = "Bạn là trợ lý AI thân thiện."
            state["prompt"] = state["user_message"]
        
        return state
    
    # ==================== CONDITIONAL EDGES ====================
    
    def route_by_intent(self, state: GraphState) -> str:
        """Route dựa trên intent"""
        return state["intent"]
    
    def route_by_symptoms(self, state: GraphState) -> str:
        """Route dựa trên có triệu chứng hay không"""
        if not state.get("has_symptoms"):
            return "no_symptoms"
        
        if state["intent"] == "doctor_recommendation":
            return "has_symptoms_doctor"
        else:  # medicine_inquiry
            return "has_symptoms_medicine"
    
    # ==================== PUBLIC API ====================
    
    def route(self, user_message: str, conversation_context: str = "", user_only_context: str = "") -> Dict[str, Any]:
        """
        Main entry point - giống API cũ
        
        Returns:
            Dict với keys: intent, use_context, system_prompt, prompt
        """
        print(f"\n{'='*60}")
        print(f"🔍 LANGGRAPH ROUTER")
        print(f"{'='*60}")
        print(f"User: {user_message}")
        print(f"User context: '{user_only_context[:50]}...'")
        print(f"{'='*60}\n")
        
        # Prepare initial state
        initial_state: GraphState = {
            "user_message": user_message,
            "conversation_context": conversation_context,
            "user_only_context": user_only_context,
            "intent": "general_chat",
            "has_symptoms": None,
            "medical_context": None,
            "doctor_context": None,
            "medicine_context": None,
            "system_prompt": "",
            "prompt": "",
            "use_context": False
        }
        
        # Run graph
        final_state = self.graph.invoke(initial_state)
        
        # Return response
        return {
            "intent": final_state["intent"],
            "use_context": final_state["use_context"],
            "system_prompt": final_state["system_prompt"],
            "prompt": final_state["prompt"]
        }
