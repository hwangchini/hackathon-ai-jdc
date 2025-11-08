import openai
import json
import os
from typing import Dict
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMProcessor:
    """Process scraped data using LLM for translation and normalization"""
    
    def __init__(self):
        self.api_key = os.getenv('AZURE_OPENAI_API_KEY')
        self.api_base = os.getenv('AZURE_OPENAI_ENDPOINT')
        self.api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
        self.deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4o-mini')
        self.api_type = os.getenv('AZURE_OPENAI_API_TYPE', 'azure')
        
        if not self.api_key:
            raise ValueError("AZURE_OPENAI_API_KEY not found in environment variables")
        if not self.api_base:
            raise ValueError("AZURE_OPENAI_ENDPOINT not found in environment variables")
            
        openai.api_type = self.api_type
        openai.api_key = self.api_key
        openai.api_base = self.api_base
        openai.api_version = self.api_version
    
    def process_medicine_data(self, raw_data: Dict, target_structure: Dict) -> Dict:
        """
        Xử lý dữ liệu thuốc: dịch sang tiếng Việt và chuẩn hóa theo cấu trúc mẫu
        
        Args:
            raw_data: Dữ liệu đã scrape (tiếng Anh)
            target_structure: Cấu trúc mẫu (1 medicine từ JSON)
        """
        try:
            prompt = self._create_processing_prompt(raw_data, target_structure)
            
            response = openai.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": """Bạn là chuyên gia dược học, nhiệm vụ của bạn là:
1. Dịch thông tin thuốc từ tiếng Anh sang tiếng Việt chuyên nghiệp
2. Chuẩn hóa dữ liệu theo cấu trúc JSON được cung cấp
3. Bổ sung thông tin còn thiếu nếu có thể dựa trên kiến thức y khoa
4. Đảm bảo tính chính xác và đầy đủ của thông tin
5. Trả về ĐÚNG định dạng JSON, không thêm text giải thích"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            logger.info(f"Successfully processed {raw_data.get('medicine_name')}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing with LLM: {e}")
            return None
    
    def _create_processing_prompt(self, raw_data: Dict, target_structure: Dict) -> str:
        """Tạo prompt cho LLM"""
        return f"""
# Dữ liệu thuốc cần xử lý (tiếng Anh):
```json
{json.dumps(raw_data, indent=2, ensure_ascii=False)}
```

# Cấu trúc mẫu cần tuân theo:
```json
{json.dumps(target_structure, indent=2, ensure_ascii=False)}
```

# Yêu cầu:
1. Dịch toàn bộ thông tin sang tiếng Việt chuyên môn y khoa
2. Chuẩn hóa theo đúng cấu trúc mẫu với TẤT CẢ các trường
3. Với trường "category", phân loại thuốc theo tiếng Việt (VD: "Thuốc giảm đau - hạ sốt", "Thuốc kháng sinh", "Thuốc chống viêm")
4. Với trường "dosage", tách riêng "adult" và "children" nếu có thông tin
5. Bổ sung "last_updated" với ngày hiện tại (YYYY-MM-DD)
6. Nếu thiếu thông tin, dựa vào kiến thức y khoa để bổ sung hợp lý
7. Đảm bảo danh sách "indications", "contraindications", "side_effects" đầy đủ và chi tiết

Trả về JSON object hoàn chỉnh, không thêm markdown hay text khác.
"""
    
    def translate_and_enrich(self, medicine_name: str, existing_data: Dict = None) -> Dict:
        """
        Dịch và làm giàu thông tin thuốc dựa trên tên thuốc và dữ liệu hiện có
        """
        try:
            prompt = f"""
Thuốc: {medicine_name}

Dữ liệu hiện có (có thể thiếu):
```json
{json.dumps(existing_data, indent=2, ensure_ascii=False) if existing_data else "{}"}
```

Hãy cung cấp thông tin đầy đủ về thuốc này bằng tiếng Việt theo cấu trúc:
- medicine_name (tên thương mại)
- generic_name (hoạt chất)
- brand_names (các tên thương hiệu - mảng)
- category (phân loại thuốc bằng tiếng Việt)
- indications (chỉ định - mảng chi tiết)
- dosage (liều dùng - object có "adult" và "children")
- contraindications (chống chỉ định - mảng)
- side_effects (tác dụng phụ - mảng)
- warnings (cảnh báo - chuỗi)
- source (nguồn)
- reference_url (URL tham khảo uy tín)
- last_updated (ngày hiện tại YYYY-MM-DD)

Trả về JSON object hoàn chỉnh.
"""
            
            response = openai.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": "Bạn là chuyên gia dược học, cung cấp thông tin chính xác về thuốc bằng tiếng Việt."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            logger.error(f"Error in translate_and_enrich: {e}")
            return None
