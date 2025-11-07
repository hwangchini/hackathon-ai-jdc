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
            
            # Get top result
            doc, score = results[0]
            return doc.page_content
            
        except Exception as e:
            return f"Lỗi tra cứu thuốc: {str(e)}"
    
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
