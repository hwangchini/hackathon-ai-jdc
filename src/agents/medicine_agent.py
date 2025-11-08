from typing import Dict, Any, Optional, List
from src.models.llm import get_llm


class MedicineAgent:
    """Agent chuyên tư vấn về thuốc - Hybrid with tools"""
    
    def __init__(self, vector_service=None):
        self.llm = get_llm(streaming=False)
        self.vector_service = vector_service
        
        # ✅ Initialize tools if available
        if vector_service:
            from src.tools.medical_tools import MedicalTools
            self.medical_tools = MedicalTools(vector_service)
        else:
            self.medical_tools = None
    
    def _format_medicine_info(self, doc, score=None) -> str:
        """Helper method để format thông tin thuốc bao gồm nguồn"""
        medicine_name = doc.metadata.get('item_name', 'Thuốc')
        content = doc.page_content
        
        # ✅ Debug: In ra metadata để kiểm tra
        print(f"🔍 DEBUG - Metadata của {medicine_name}:")
        print(f"  - source: {doc.metadata.get('source', 'MISSING')}")
        print(f"  - reference_url: {doc.metadata.get('reference_url', 'MISSING')}")
        print(f"  - last_updated: {doc.metadata.get('last_updated', 'MISSING')}")
        
        # Thêm thông tin nguồn nếu có
        source = doc.metadata.get('source', '')
        reference_url = doc.metadata.get('reference_url', '')
        last_updated = doc.metadata.get('last_updated', '')
        
        result = f"{'='*60}\n{medicine_name.upper()}\n{'='*60}\n\n{content}"
        
        # ✅ Thêm section nguồn tham khảo với format nổi bật
        if source or reference_url or last_updated:
            result += f"\n\n{'='*60}\n📚 NGUỒN THAM KHẢO (BẮT BUỘC HIỂN THỊ)\n{'='*60}\n"
            if source:
                result += f"📖 Nguồn: {source}\n"
            if reference_url:
                result += f"🔗 Link tham khảo: {reference_url}\n"
            if last_updated:
                result += f"📅 Cập nhật: {last_updated}\n"
            result += "="*60
            print(f"✅ Đã thêm nguồn tham khảo cho {medicine_name}")
        else:
            print(f"⚠️ KHÔNG có thông tin nguồn cho {medicine_name}")
        
        return result
    
    def search_medicine_by_symptoms(self, symptoms: str, conversation_context: str = "") -> Optional[str]:
        """Tìm thuốc - Với LLM validation cải tiến"""
        if not self.vector_service or not self.vector_service.vector_store:
            return None
        
        try:
            # Trích xuất triệu chứng
            if conversation_context:
                extract_prompt = f"""Từ lịch sử, CHỈ liệt kê triệu chứng người dùng ĐÃ NÓI:
{conversation_context}

QUY TẮC:
- CHỈ ghi triệu chứng có trong tin nhắn người dùng
- KHÔNG thêm triệu chứng khác
- Format ngắn: "triệu chứng1, triệu chứng2"

Triệu chứng:"""
                
                response = self.llm.invoke(extract_prompt)
                extracted_symptoms = response.content.strip()
                
                # Validation
                context_lower = conversation_context.lower()
                symptom_keywords = [s.strip() for s in extracted_symptoms.split(',')]
                
                validated_keywords = []
                for keyword in symptom_keywords:
                    if len(keyword) < 2:
                        continue
                    if keyword.lower() in context_lower:
                        validated_keywords.append(keyword)
                    else:
                        print(f"⚠️ Loại bỏ triệu chứng không có trong lịch sử: '{keyword}'")
                
                extracted_symptoms = ", ".join(validated_keywords)
                print(f"✅ Triệu chứng sau validation: {extracted_symptoms}")
            else:
                extracted_symptoms = symptoms
            
            if not extracted_symptoms or len(extracted_symptoms) < 2:
                print("❌ Không có triệu chứng hợp lệ")
                return None
            
            print(f"💊 Tìm thuốc cho triệu chứng: {extracted_symptoms}")
            
            # ✅ Check if user asks for specific medicine by name
            if self.medical_tools:
                medicine_keywords = ["paracetamol", "ibuprofen", "omeprazole", "cetirizine", "loperamide"]
                query_lower = extracted_symptoms.lower()
                
                for med_name in medicine_keywords:
                    if med_name in query_lower:
                        print(f"🔧 Detected medicine name query: {med_name}")
                        tool_result = self.medical_tools.search_medicine_by_name(med_name)
                        
                        if tool_result and "Lỗi" not in tool_result:
                            return f"THÔNG TIN THUỐC:\n\n{tool_result}\n\n{'='*60}\n"
            
            # Original symptom-based search
            symptom_keywords = [s.strip() for s in extracted_symptoms.split(',')]
            medicine_scores = {}
            
            for keyword in symptom_keywords:
                if len(keyword) < 2:
                    continue
                
                queries = [keyword, f"thuốc {keyword}", f"điều trị {keyword}", f"giảm {keyword}"]
                
                for query in queries:
                    results = self.vector_service.similarity_search_with_filter_and_scores(
                        query=query,
                        k=5,
                        filter_dict={"filename": "medicines.json"}
                    )
                    
                    for doc, score in results:
                        medicine_name = doc.metadata.get('item_name')
                        if not medicine_name:
                            continue
                        
                        total_score = score
                        
                        indications_text = doc.metadata.get('indications_text', '').lower()
                        if keyword.lower() in indications_text:
                            total_score += 0.5
                        
                        if keyword.lower() in doc.page_content.lower():
                            total_score += 0.2
                        
                        if medicine_name not in medicine_scores or medicine_scores[medicine_name]['score'] < total_score:
                            medicine_scores[medicine_name] = {
                                'doc': doc,
                                'score': total_score,
                                'cosine_score': score
                            }
            
            sorted_medicines = sorted(medicine_scores.items(), key=lambda x: x[1]['score'], reverse=True)
            medicine_candidates = [(item[1]['doc'], item[1]['score']) for item in sorted_medicines[:5]]
            
            print(f"📊 Tìm thấy {len(medicine_candidates)} loại thuốc candidates")
            
            # ✅ LLM VALIDATION với prompt cải tiến
            if medicine_candidates:
                validated_medicines = []
                
                for doc, score in medicine_candidates:
                    medicine_name = doc.metadata.get('item_name', 'Unknown')
                    category = doc.metadata.get('category', '')
                    indications_text = doc.metadata.get('indications_text', '')
                    
                    # ✅ Prompt chi tiết hơn với strict rules
                    validation_prompt = f"""Bạn là dược sĩ chuyên nghiệp. Phân tích xem thuốc có TRỰC TIẾP điều trị triệu chứng không.

**TRIỆU CHỨNG CỦA BỆNH NHÂN:**
{extracted_symptoms}

**THUỐC ĐANG XÉT:**
- Tên: {medicine_name}
- Loại: {category}
- Chỉ định: {indications_text}

**QUY TẮC NGHIÊM NGẶT:**

✅ CHỈ TRẢ LỜI "PHÙ HỢP" KHI:
- Chỉ định của thuốc TRỰC TIẾP match với triệu chứng
- Ví dụ:
  + Triệu chứng "tiêu chảy" + Chỉ định "Tiêu chảy cấp, Tiêu chảy mạn tính" → PHÙ HỢP ✅
  + Triệu chứng "sốt" + Chỉ định "Hạ sốt, Giảm đau" → PHÙ HỢP ✅
  + Triệu chứng "đau đầu" + Chỉ định "Đau đầu, Giảm đau" → PHÙ HỢP ✅

❌ TRẢ LỜI "KHÔNG PHÙ HỢP" KHI:
- Chỉ định KHÔNG liên quan trực tiếp
- Ví dụ:
  + Triệu chứng "tiêu chảy" + Chỉ định "Loét dạ dày, Ợ nóng" → KHÔNG PHÙ HỢP ❌
  + Triệu chứng "sốt" + Chỉ định "Tiêu chảy cấp" → KHÔNG PHÙ HỢP ❌
  + Triệu chứng "đau đầu" + Chỉ định "Viêm mũi dị ứng" → KHÔNG PHÙ HỢP ❌

**CÂU HỎI:**
Với triệu chứng "{extracted_symptoms}", thuốc "{medicine_name}" (chỉ định: "{indications_text}") có PHÙ HỢP để điều trị TRỰC TIẾP không?

**CHỈ TRẢ LỜI MỘT TRONG HAI:**
- "PHÙ HỢP" (nếu chỉ định match trực tiếp)
- "KHÔNG PHÙ HỢP" (nếu không match)

Trả lời:"""
                    
                    try:
                        response = self.llm.invoke(validation_prompt)
                        decision = response.content.strip().upper()
                        
                        # Stricter parsing
                        if "PHÙ HỢP" in decision and "KHÔNG" not in decision:
                            validated_medicines.append((doc, score))
                            print(f"  ✅ {medicine_name} ({category}) - LLM: PHÙ HỢP")
                        else:
                            print(f"  ❌ {medicine_name} ({category}) - LLM: KHÔNG PHÙ HỢP")
                            print(f"       Lý do: Chỉ định '{indications_text}' không match '{extracted_symptoms}'")
                    
                    except Exception as e:
                        print(f"  ⚠️ {medicine_name} - LLM error: {str(e)}, skipping")
                        continue
                
                # Lấy top 3
                validated_medicines = validated_medicines[:3]
                
                print(f"✅ Sau LLM validation: {len(validated_medicines)} thuốc phù hợp")
                
                if validated_medicines:
                    context_parts = []
                    for i, (doc, score) in enumerate(validated_medicines, 1):
                        medicine_name = doc.metadata.get('item_name', f'Thuốc {i}')
                        cosine = medicine_scores[medicine_name]['cosine_score']
                        print(f"  {i}. {medicine_name} (Score: {score:.3f})")
                        
                        # ✅ Sử dụng helper method để format bao gồm nguồn
                        formatted_info = self._format_medicine_info(doc, score)
                        context_parts.append(formatted_info)
                    
                    context = "\n\n".join(context_parts)
                    result = f"THÔNG TIN THUỐC:\n\n{context}\n\n{'='*60}\n"
                    
                    print(f"✅ Returning {len(validated_medicines)} LLM-validated medicines")
                    
                    return result
            
            print("❌ Không tìm thấy thuốc phù hợp sau LLM validation")
            return None
            
        except Exception as e:
            print(f"⚠️ Lỗi tìm kiếm thuốc: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_medicine_details(self, medicine_name: str) -> Optional[str]:
        """Lấy thông tin chi tiết về một loại thuốc cụ thể"""
        if not self.vector_service or not self.vector_service.vector_store:
            return None
        
        try:
            results = self.vector_service.similarity_search(medicine_name, k=1)
            
            if results and results[0].metadata.get('filename') == 'medicines.json':
                doc = results[0]
                # ✅ Sử dụng helper method để format bao gồm nguồn
                return self._format_medicine_info(doc)
            
            return None
            
        except Exception as e:
            print(f"⚠️ Lỗi tra cứu thuốc: {str(e)}")
            return None
    
    def check_drug_interaction(self, drug1: str, drug2: str) -> Optional[str]:
        """Kiểm tra tương tác giữa hai loại thuốc"""
        if not self.vector_service or not self.vector_service.vector_store:
            return None
        
        try:
            query = f"{drug1} {drug2} tương tác"
            results = self.vector_service.similarity_search(query, k=3)
            
            interaction_docs = [doc for doc in results 
                              if doc.metadata.get('filename') == 'drug_interactions.json']
            
            if interaction_docs:
                context = "\n\n".join([doc.page_content for doc in interaction_docs])
                return f"CẢNH BÁO TƯƠNG TÁC THUỐC:\n\n{context}"
            
            return None
            
        except Exception as e:
            print(f"⚠️ Lỗi kiểm tra tương tác thuốc: {str(e)}")
            return None
    
    def get_health_tips(self, category: str = "") -> Optional[str]:
        """Lấy lời khuyên sức khỏe"""
        if not self.vector_service or not self.vector_service.vector_store:
            return None
        
        try:
            query = f"{category} sức khỏe lời khuyên" if category else "sức khỏe lời khuyên"
            results = self.vector_service.similarity_search(query, k=2)
            
            tip_docs = [doc for doc in results 
                       if doc.metadata.get('filename') == 'health_tips.json']
            
            if tip_docs:
                context = "\n\n".join([doc.page_content for doc in tip_docs])
                return f"💡 LỜI KHUYÊN SỨC KHỎE:\n\n{context}"
            
            return None
            
        except Exception as e:
            print(f"⚠️ Lỗi lấy health tips: {str(e)}")
            return None
