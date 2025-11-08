from langchain.tools import tool
from typing import Optional, List
from src.services.vector_store import VectorStoreService


class MedicalTools:
    """Collection of tools for medical assistant"""
    
    def __init__(self, vector_service: VectorStoreService):
        self.vector_service = vector_service
    
    @tool
    def search_doctors_by_specialty(self, specialty: str) -> str:
        """
        Tìm bác sĩ theo chuyên khoa.
        
        Args:
            specialty: Tên chuyên khoa (VD: "Tim mạch", "Tiêu hóa", "Da liễu")
            
        Returns:
            Danh sách bác sĩ
        """
        try:
            results = self.vector_service.similarity_search_with_filter_and_scores(
                query=specialty,
                k=5,
                filter_dict={"filename": "medical_personnel.json"}
            )
            
            if not results:
                return f"Không tìm thấy bác sĩ chuyên khoa {specialty}"
            
            # Format output
            doctors_info = []
            for doc, score in results[:3]:
                dept_name = doc.metadata.get('department_name', 'N/A')
                doctors_info.append(f"**{dept_name}**\n{doc.page_content}")
            
            return "\n\n".join(doctors_info)
            
        except Exception as e:
            return f"Lỗi tìm bác sĩ: {str(e)}"
    
    def _format_medicine_info(self, doc) -> str:
        """Helper method để format thông tin thuốc bao gồm nguồn"""
        medicine_name = doc.metadata.get('item_name', 'Thuốc')
        content = doc.page_content
        
        # ✅ Debug: In ra metadata để kiểm tra
        print(f"🔍 DEBUG [MedicalTools] - Metadata của {medicine_name}:")
        print(f"  - source: {doc.metadata.get('source', 'MISSING')}")
        print(f"  - reference_url: {doc.metadata.get('reference_url', 'MISSING')}")
        print(f"  - last_updated: {doc.metadata.get('last_updated', 'MISSING')}")
        
        # Thêm thông tin nguồn nếu có
        source = doc.metadata.get('source', '')
        reference_url = doc.metadata.get('reference_url', '')
        last_updated = doc.metadata.get('last_updated', '')
        
        result = f"{content}"
        
        # ✅ Thêm section nguồn tham khảo với format nổi bật và dễ parse
        if source or reference_url or last_updated:
            result += f"\n\n{'='*60}\n📚 NGUỒN THAM KHẢO (BẮT BUỘC HIỂN THỊ)\n{'='*60}\n"
            if source:
                result += f"📖 Nguồn: {source}\n"
            if reference_url:
                result += f"🔗 Link tham khảo: {reference_url}\n"
            if last_updated:
                result += f"📅 Cập nhật: {last_updated}\n"
            result += "="*60 + "\n"
            result += "⚠️ LƯU Ý: Phần nguồn tham khảo này BẮT BUỘC phải được bao gồm trong câu trả lời cuối cùng cho người dùng."
            print(f"✅ [MedicalTools] Đã thêm nguồn tham khảo cho {medicine_name}")
        else:
            print(f"⚠️ [MedicalTools] KHÔNG có thông tin nguồn cho {medicine_name}")
        
        return result
    
    @tool
    def search_medicine_by_name(self, medicine_name: str) -> str:
        """
        Tra cứu thông tin thuốc theo tên.
        
        Args:
            medicine_name: Tên thuốc cần tra (VD: "Paracetamol", "Ibuprofen")
            
        Returns:
            Thông tin chi tiết về thuốc
        """
        try:
            results = self.vector_service.similarity_search_with_filter_and_scores(
                query=medicine_name,
                k=3,
                filter_dict={"filename": "medicines.json"}
            )
            
            if not results:
                return f"Không tìm thấy thông tin về thuốc {medicine_name}"
            
            # Lấy kết quả có score cao nhất
            best_doc, best_score = results[0]
            found_name = best_doc.metadata.get('item_name', '')
            
            # Kiểm tra xem tên có match không (case-insensitive)
            if medicine_name.lower() in found_name.lower():
                # ✅ Sử dụng helper method để format bao gồm nguồn
                return self._format_medicine_info(best_doc)
            else:
                return f"❌ Không tìm thấy thông tin chính xác về thuốc '{medicine_name}'"
            
        except Exception as e:
            return f"⚠️ Lỗi khi tìm kiếm thuốc: {str(e)}"
    
    @tool
    def search_symptoms_info(self, symptom: str) -> str:
        """
        Tìm thông tin về triệu chứng và bệnh lý.
        
        Args:
            symptom: Triệu chứng (VD: "đau đầu", "sốt", "ho")
            
        Returns:
            Thông tin về triệu chứng
        """
        try:
            results = self.vector_service.similarity_search_with_filter_and_scores(
                query=symptom,
                k=3,
                filter_dict={"filename": "symptoms.json"}
            )
            
            if not results:
                return f"Không tìm thấy thông tin về triệu chứng {symptom}"
            
            # Format output
            info_parts = []
            for doc, score in results:
                info_parts.append(doc.page_content)
            
            return "\n\n".join(info_parts)
            
        except Exception as e:
            return f"Lỗi tìm triệu chứng: {str(e)}"
    
    def get_all_tools(self):
        """Lấy tất cả tools"""
        return [
            self.search_doctors_by_specialty,
            self.search_medicine_by_name,
            self.search_symptoms_info
        ]
