from enum import Enum
from typing import Dict, Any, Optional
from src.models.llm import get_llm
from src.agents.medicine_agent import MedicineAgent


class IntentType(Enum):
    """Các loại intent trong cuộc hội thoại"""
    MEDICAL_CONSULTATION = "medical_consultation"
    DOCTOR_RECOMMENDATION = "doctor_recommendation"
    MEDICINE_INQUIRY = "medicine_inquiry"
    GENERAL_CHAT = "general_chat"


class AgentRouter:
    """Agent Router để điều hướng cuộc hội thoại"""
    
    def __init__(self, vector_service=None):
        self.llm = get_llm(streaming=False)
        self.vector_service = vector_service
        self.medicine_agent = MedicineAgent(vector_service)
        
        # Keywords y tế
        self.medical_keywords = [
            'triệu chứng', 'bệnh', 'đau', 'sốt', 'ho', 'khó thở',
            'mệt mỏi', 'chóng mặt', 'buồn nôn', 'tiêu chảy', 'táo bón',
            'nhức đầu', 'đau bụng', 'ngứa', 'phát ban', 'sưng', 'viêm'
        ]
        
        # Keywords tìm bác sĩ
        self.doctor_keywords = [
            'bác sĩ', 'bác sỹ', 'doctor', 'khám', 'tư vấn', 'gặp ai',
            'nên đi khám', 'khoa nào', 'chuyên khoa', 'phòng khám'
        ]
        
        # Keywords tìm thuốc
        self.medicine_keywords = [
            'thuốc', 'uống thuốc gì', 'dùng thuốc', 'mua thuốc',
            'liều dùng', 'cách dùng', 'tác dụng phụ', 'chống chỉ định'
        ]
    
    def normalize_text(self, text: str) -> str:
        """Chuẩn hóa text để search tốt hơn"""
        import unicodedata
        text = unicodedata.normalize('NFKD', text)
        text = ''.join([c for c in text if not unicodedata.combining(c)])
        return text.lower().strip()
    
    def classify_intent(self, user_message: str) -> IntentType:
        """Phân loại intent của tin nhắn"""
        user_message_lower = user_message.lower()
        
        # Kiểm tra keywords thuốc
        if any(keyword in user_message_lower for keyword in self.medicine_keywords):
            return IntentType.MEDICINE_INQUIRY
        
        # Kiểm tra keywords tìm bác sĩ
        if any(keyword in user_message_lower for keyword in self.doctor_keywords):
            return IntentType.DOCTOR_RECOMMENDATION
        
        # Kiểm tra keywords y tế
        if any(keyword in user_message_lower for keyword in self.medical_keywords):
            return IntentType.MEDICAL_CONSULTATION
        
        # Sử dụng LLM để phân loại
        prompt = f"""Phân loại intent của câu hỏi người dùng.

Có 4 loại intent:
1. MEDICAL_CONSULTATION - Hỏi về triệu chứng bệnh lý, sức khỏe
2. DOCTOR_RECOMMENDATION - Tìm bác sĩ, hỏi nên đi khám ở đâu
3. MEDICINE_INQUIRY - Hỏi về thuốc, liều dùng, tác dụng phụ
4. GENERAL_CHAT - Trò chuyện bình thường

Chỉ trả về TÊN intent, không giải thích.

Câu hỏi: {user_message}
Intent:"""
        
        try:
            response = self.llm.invoke(prompt)
            intent_text = response.content.strip().upper()
            
            if "MEDICINE_INQUIRY" in intent_text:
                return IntentType.MEDICINE_INQUIRY
            elif "DOCTOR_RECOMMENDATION" in intent_text:
                return IntentType.DOCTOR_RECOMMENDATION
            elif "MEDICAL_CONSULTATION" in intent_text:
                return IntentType.MEDICAL_CONSULTATION
            else:
                return IntentType.GENERAL_CHAT
        except:
            return IntentType.GENERAL_CHAT
    
    def check_has_symptoms_with_llm(self, user_only_context: str) -> bool:
        """
        Sử dụng LLM để xác định xem người dùng đã cung cấp triệu chứng hay chưa
        
        Returns:
            True nếu đã có triệu chứng, False nếu chưa
        """
        if not user_only_context or len(user_only_context.strip()) < 5:
            return False
        
        prompt = f"""Phân tích xem người dùng đã cung cấp triệu chứng bệnh lý hay chưa.

Lịch sử tin nhắn của người dùng:
"{user_only_context}"

Hãy xác định:
- Có triệu chứng cụ thể không? (VD: đau đầu, sốt, ho, buồn nôn, đau bụng...)
- Chỉ tính triệu chứng THẬT, KHÔNG tính ví dụ hoặc từ trong câu hỏi

Ví dụ phân biệt:
✅ CÓ triệu chứng: "tôi bị đau đầu", "tôi đang ho", "con tôi sốt"
❌ KHÔNG có: "tôi nên uống thuốc gì?", "gợi ý bác sĩ cho tôi", "đau đầu là gì?"

CHỈ trả lời: "CÓ" hoặc "KHÔNG", không giải thích.

Trả lời:"""
        
        try:
            response = self.llm.invoke(prompt)
            answer = response.content.strip().upper()
            
            has_symptoms = "CÓ" in answer or "CO" in answer or "YES" in answer
            
            print(f"🤖 LLM判断: {answer} → {'CÓ triệu chứng' if has_symptoms else 'KHÔNG có triệu chứng'}")
            
            return has_symptoms
            
        except Exception as e:
            print(f"⚠️ Lỗi LLM check symptoms: {str(e)}")
            # Fallback: return False để an toàn
            return False
    
    def get_medical_context(self, user_message: str, k: int = 3) -> Optional[str]:
        """Truy vấn thông tin y tế từ vectorDB"""
        if not self.vector_service or not self.vector_service.vector_store:
            return None
        
        try:
            relevant_docs = self.vector_service.similarity_search(user_message, k=k)
            
            if relevant_docs:
                context_parts = []
                for i, doc in enumerate(relevant_docs, 1):
                    symptom_name = doc.metadata.get('symptom_name', doc.metadata.get('item_name', f'Tài liệu {i}'))
                    context_parts.append(f"{'='*60}\n{symptom_name.upper()}\n{'='*60}\n{doc.page_content}")
                
                context = "\n\n".join(context_parts)
                return f"THÔNG TIN Y TẾ:\n\n{context}\n\n{'='*60}\n"
            
            return None
        except Exception as e:
            print(f"⚠️ Lỗi truy vấn vectorDB: {str(e)}")
            return None
    
    def get_doctor_recommendations(self, user_message: str, conversation_context: str = "") -> Optional[str]:
        """Lấy gợi ý bác sĩ dựa trên triệu chứng"""
        if not self.vector_service or not self.vector_service.vector_store:
            return None
        
        try:
            # Bước 1: Trích xuất triệu chứng
            symptoms_text = ""
            if conversation_context:
                extract_prompt = f"""Từ lịch sử hội thoại, liệt kê TẤT CẢ triệu chứng.

Lịch sử:
{conversation_context}

Chỉ liệt kê triệu chứng, cách nhau bằng dấu phẩy.

Triệu chứng:"""
                
                response = self.llm.invoke(extract_prompt)
                symptoms_text = response.content.strip()
            
            # Bước 2: Map triệu chứng → chuyên khoa
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

Chọn khoa phù hợp từ danh sách:
Tim mạch, Tiêu hóa, Nội tiết, Tai-Mũi-Họng, Mắt, Da liễu, Nhi, Sản, Phẫu thuật Thần kinh

Chỉ trả về TÊN KHOA:"""
                
                response = self.llm.invoke(specialty_prompt)
                specialty = response.content.strip()
                possible_specialties = [specialty]
            
            print(f"🔍 Triệu chứng: {symptoms_text}")
            print(f"🏥 Chuyên khoa ứng viên: {possible_specialties}")
            
            # Bước 3: Search với cosine similarity scores
            all_results_with_scores = []
            for specialty in possible_specialties:
                queries = [
                    specialty,
                    f"khoa {specialty}",
                    f"bác sĩ {specialty}",
                    self.normalize_text(specialty),
                    self.normalize_text(f"khoa {specialty}")
                ]
                
                for query in queries:
                    results = self.vector_service.similarity_search_with_scores(query, k=3)
                    all_results_with_scores.extend(results)
            
            if symptoms_text:
                results = self.vector_service.similarity_search_with_scores(symptoms_text, k=3)
                all_results_with_scores.extend(results)
            
            # Bước 4: Lọc và combine scores
            dept_scores = {}
            
            for doc, cosine_score in all_results_with_scores:
                if doc.metadata.get('filename') == 'medical_personnel.json':
                    dept_name = doc.metadata.get('department_name')
                    specialty_name = doc.metadata.get('specialty_name', '')
                    
                    if dept_name:
                        # Combine: cosine similarity + text matching bonus
                        total_score = cosine_score
                        
                        # Cộng bonus từ text matching
                        dept_lower = dept_name.lower()
                        specialty_lower = specialty_name.lower()
                        
                        for spec in possible_specialties:
                            spec_lower = spec.lower()
                            if spec_lower in dept_lower:
                                total_score += 0.2
                            if spec_lower in specialty_lower:
                                total_score += 0.1
                        
                        if symptoms_text and symptoms_text.lower() in doc.page_content.lower():
                            total_score += 0.05
                        
                        if dept_name not in dept_scores or dept_scores[dept_name]['score'] < total_score:
                            dept_scores[dept_name] = {
                                'doc': doc,
                                'score': total_score,
                                'cosine_score': cosine_score
                            }
            
            # Sắp xếp theo điểm
            sorted_depts = sorted(dept_scores.items(), key=lambda x: x[1]['score'], reverse=True)
            doctor_docs = [item[1]['doc'] for item in sorted_depts[:3]]
            
            print(f"📊 Tìm thấy {len(doctor_docs)} khoa phù hợp")
            for dept_name, info in sorted_depts[:3]:
                print(f"  • {dept_name}: Cosine={info['cosine_score']:.3f}, Total={info['score']:.3f}")
            
            if doctor_docs:
                context_parts = []
                for i, doc in enumerate(doctor_docs, 1):
                    specialty_name = doc.metadata.get('specialty_name', 'N/A')
                    dept_name = doc.metadata.get('department_name', f'Khoa {i}')
                    
                    context_parts.append(f"{'='*60}\n{dept_name.upper()} - {specialty_name}\n{'='*60}\n{doc.page_content}")
                
                context = "\n\n".join(context_parts)
                return f"THÔNG TIN BÁC SĨ:\n\n{context}\n\n{'='*60}\n"
            
            print("❌ Không tìm thấy bác sĩ phù hợp")
            return None
            
        except Exception as e:
            print(f"⚠️ Lỗi truy vấn thông tin bác sĩ: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def route(self, user_message: str, conversation_context: str = "", user_only_context: str = "") -> Dict[str, Any]:
        """Điều hướng và tạo response phù hợp"""
        intent = self.classify_intent(user_message)
        
        # DEBUG
        print(f"\n{'='*60}")
        print(f"🔍 DEBUG ROUTER")
        print(f"{'='*60}")
        print(f"User message: {user_message}")
        print(f"User only context: '{user_only_context}'")
        print(f"Intent: {intent.value}")
        print(f"{'='*60}\n")
        
        if intent == IntentType.MEDICAL_CONSULTATION:
            medical_context = self.get_medical_context(user_message)
            
            if medical_context:
                return {
                    "intent": intent.value,
                    "use_context": True,
                    "system_prompt": """Bạn là trợ lý y tế AI chuyên nghiệp. 
Nhiệm vụ:
1. GHI NHỚ tất cả triệu chứng trong cuộc trò chuyện
2. Phân tích triệu chứng dựa trên thông tin y tế
3. Chẩn đoán khả năng bệnh lý
4. Đưa ra lời khuyên cụ thể

LƯU Ý: Đây chỉ là thông tin tham khảo, KHÔNG thay thế ý kiến bác sĩ.""",
                    "prompt": f"""{medical_context}

Câu hỏi: {user_message}

Phân tích và tư vấn:"""
                }
            else:
                return {
                    "intent": intent.value,
                    "use_context": False,
                    "system_prompt": "Bạn là trợ lý y tế AI. GHI NHỚ triệu chứng.",
                    "prompt": f"{user_message}\n\nKHUYẾN NGHỊ gặp bác sĩ."
                }
        
        # DOCTOR_RECOMMENDATION
        elif intent == IntentType.DOCTOR_RECOMMENDATION:
            check_context = user_only_context if user_only_context else conversation_context
            
            print(f"🔍 Checking symptoms with LLM...")
            
            # Sử dụng LLM thay vì keywords
            has_symptoms = self.check_has_symptoms_with_llm(check_context)
            
            if not has_symptoms:
                print("❌ LLM xác nhận: Chưa có triệu chứng → Yêu cầu user cung cấp")
                return {
                    "intent": intent.value,
                    "use_context": False,
                    "system_prompt": "Bạn là trợ lý y tế AI chuyên nghiệp. KHÔNG tự bịa triệu chứng.",
                    "prompt": f"""Người dùng hỏi: {user_message}

QUAN TRỌNG: Người dùng CHƯA cung cấp triệu chứng cụ thể.

Hãy trả lời:
"Để gợi ý bác sĩ phù hợp, tôi cần biết thêm thông tin về tình trạng sức khỏe của bạn.

Vui lòng cho tôi biết:
- Bạn đang gặp triệu chứng gì? (VD: đau đầu, đau bụng, ho, sốt...)
- Triệu chứng xuất hiện từ bao lâu?
- Mức độ nghiêm trọng như thế nào?

Sau khi có thông tin này, tôi sẽ gợi ý bác sĩ và chuyên khoa phù hợp cho bạn."

TUYỆT ĐỐI KHÔNG được tự bịa triệu chứng hoặc gợi ý bác sĩ."""
                }
            
            print("✅ LLM xác nhận: Có triệu chứng → Tìm bác sĩ")
            # Có triệu chứng, tìm bác sĩ phù hợp
            doctor_context = self.get_doctor_recommendations(user_message, conversation_context)
            medical_context = None
            
            if conversation_context:
                medical_context = self.get_medical_context(conversation_context)
            
            combined_context = ""
            if medical_context:
                combined_context += f"{medical_context}\n\n"
            if doctor_context:
                combined_context += doctor_context
            
            if combined_context:
                return {
                    "intent": intent.value,
                    "use_context": True,
                    "system_prompt": """Bạn là trợ lý tư vấn y tế chuyên gợi ý bác sĩ.
Nhiệm vụ:
1. Dựa vào LỊCH SỬ HỘI THOẠI để xác định chính xác triệu chứng
2. Phân tích và xác định chuyên khoa phù hợp
3. Đề xuất 2-3 bác sĩ phù hợp nhất
4. PHẢI ĐÚNG chuyên khoa với triệu chứng

QUY TẮC NGHIÊM NGẶT:
- CHỈ sử dụng triệu chứng có trong lịch sử hội thoại
- KHÔNG tự bịa triệu chứng
- KHÔNG gợi ý bác sĩ nếu không rõ triệu chứng""",
                    "prompt": f"""Lịch sử hội thoại:
{conversation_context}

{combined_context}

Câu hỏi: {user_message}

Hãy:
1. XÁC ĐỊNH CHÍNH XÁC triệu chứng từ lịch sử hội thoại (KHÔNG tự bịa)
2. Phân tích và chọn đúng chuyên khoa
3. Gợi ý 2-3 bác sĩ từ chuyên khoa đó
4. Giải thích tại sao

Format:
**Triệu chứng đã ghi nhận**: [CHỈ từ lịch sử hội thoại, KHÔNG bịa]
**Chuyên khoa phù hợp**: [Tên chuyên khoa]
**Bác sĩ gợi ý**:
1. [Họ tên] - [Học vị] - [Chức vụ] - [Khoa]

Trả lời:"""
                }
            else:
                return {
                    "intent": intent.value,
                    "use_context": False,
                    "system_prompt": "Bạn là trợ lý tư vấn y tế. KHÔNG tự bịa thông tin.",
                    "prompt": f"""Lịch sử: {conversation_context}

Câu hỏi: {user_message}

Hệ thống không tìm thấy thông tin bác sĩ phù hợp.

Hãy trả lời:
"Xin lỗi, để gợi ý bác sĩ chính xác, tôi cần biết thêm về triệu chứng của bạn.

Bạn có thể:
1. Mô tả rõ hơn về triệu chứng đang gặp phải
2. Cho biết thêm về tình trạng sức khỏe hiện tại
3. Hoặc gọi hotline y tế: 115 hoặc 19003115 để được tư vấn trực tiếp"

KHÔNG tự bịa triệu chứng hoặc gợi ý bác sĩ không phù hợp."""
                }
        
        # MEDICINE_INQUIRY
        elif intent == IntentType.MEDICINE_INQUIRY:
            check_context = user_only_context if user_only_context else conversation_context
            
            print(f"🔍 Checking symptoms with LLM...")
            
            # Sử dụng LLM thay vì keywords
            has_symptoms = self.check_has_symptoms_with_llm(check_context)
            
            if not has_symptoms:
                print("❌ LLM xác nhận: Chưa có triệu chứng → Yêu cầu user cung cấp")
                return {
                    "intent": intent.value,
                    "use_context": False,
                    "system_prompt": "Bạn là dược sĩ AI chuyên nghiệp. KHÔNG tự bịa triệu chứng.",
                    "prompt": f"""Người dùng hỏi: {user_message}

QUAN TRỌNG: Người dùng CHƯA cung cấp triệu chứng cụ thể.

Hãy trả lời:
"Để tư vấn thuốc phù hợp, tôi cần biết thêm thông tin về triệu chứng của bạn.

Vui lòng cho tôi biết:
- Bạn đang gặp triệu chứng gì? (VD: đau đầu, đau bụng, ho, sốt...)
- Triệu chứng xuất hiện từ bao lâu?
- Mức độ nghiêm trọng như thế nào?

Sau khi có thông tin này, tôi sẽ tư vấn thuốc phù hợp cho bạn."

TUYỆT ĐỐI KHÔNG được tự bịa triệu chứng hoặc tư vấn thuốc."""
                }
            
            print("✅ LLM xác nhận: Có triệu chứng → Tìm thuốc")
            # Có triệu chứng, tìm thuốc phù hợp
            medicine_context = self.medicine_agent.search_medicine_by_symptoms(
                user_message, 
                conversation_context
            )
            
            if medicine_context:
                return {
                    "intent": intent.value,
                    "use_context": True,
                    "system_prompt": """Bạn là dược sĩ AI chuyên nghiệp.

NHIỆM VỤ:
1. Tư vấn thuốc phù hợp với triệu chứng
2. Giải thích rõ cách dùng, liều lượng
3. Cảnh báo tác dụng phụ và chống chỉ định
4. NHẤN MẠNH: Đây chỉ là thông tin tham khảo

QUY TẮC NGHIÊM NGẶT:
- CHỈ tư vấn thuốc không kê đơn (OTC)
- BẮT BUỘC khuyên tham khảo bác sĩ
- KHÔNG bịa thông tin thuốc
- CHỈ dùng triệu chứng từ lịch sử hội thoại""",
                    "prompt": f"""{medicine_context}

Triệu chứng đã ghi nhận: {conversation_context}
Câu hỏi: {user_message}

Hãy:
1. Giới thiệu các thuốc phù hợp
2. Giải thích: Tên thương mại, liều dùng, cách dùng
3. Cảnh báo tác dụng phụ
4. Khuyến nghị tham khảo bác sĩ/dược sĩ

Trả lời:"""
                }
            else:
                return {
                    "intent": intent.value,
                    "use_context": False,
                    "system_prompt": "Bạn là dược sĩ AI. KHÔNG bịa thông tin.",
                    "prompt": f"""Câu hỏi: {user_message}

Không tìm thấy thông tin thuốc phù hợp.

Hãy trả lời:
"Xin lỗi, tôi chưa có thông tin chi tiết về thuốc cho triệu chứng này.

Tôi khuyên bạn:
1. Đến nhà thuốc/phòng khám để được tư vấn trực tiếp
2. Gặp bác sĩ để được kê đơn thuốc phù hợp
3. Gọi hotline: 19003190

Lưu ý: Không tự ý mua thuốc."""
                }
        
        else:  # GENERAL_CHAT
            return {
                "intent": intent.value,
                "use_context": False,
                "system_prompt": "Bạn là trợ lý AI thân thiện.",
                "prompt": user_message
            }
