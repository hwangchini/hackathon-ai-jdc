# 🏥 AI Medical Assistant - Chatbot Tư Vấn Y Tế

Ứng dụng chatbot y tế thông minh sử dụng **Azure OpenAI**, **LangChain**, **LangGraph** và **RAG** (Retrieval-Augmented Generation) để tư vấn y tế dựa trên dữ liệu chuyên môn.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0+-green.svg)](https://python.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.0.30+-purple.svg)](https://langchain-ai.github.io/langgraph/)
[![Azure OpenAI](https://img.shields.io/badge/Azure-OpenAI-orange.svg)](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

## 📋 Mục lục

- [Tính năng chính](#-tính-năng-chính)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
- [Cài đặt](#-cài-đặt)
- [Cấu hình](#️-cấu-hình)
- [Khởi chạy dự án](#-khởi-chạy-dự-án)
- [Sử dụng](#-sử-dụng)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [API và Components](#-api-và-components)
- [Khắc phục sự cố](#-khắc-phục-sự-cố)

## ✨ Tính năng chính

### 🏥 Tư vấn y tế thông minh
- ✅ Phân tích triệu chứng và chẩn đoán sơ bộ
- ✅ Gợi ý bác sĩ phù hợp theo chuyên khoa
- ✅ Tư vấn thuốc dựa trên triệu chứng
- ✅ Kiểm tra tương tác thuốc
- ✅ Cung cấp lời khuyên sức khỏe

### 🤖 AI-Powered với LangGraph
- **Intent Classification**: Tự động phân loại ý định người dùng
- **Symptom Validation**: LLM kiểm tra triệu chứng chính xác
- **Medicine Validation**: LLM phân tích thuốc có phù hợp không
- **Hybrid Search**: Kết hợp Vector Search + LLM Tools

### 🔍 RAG (Retrieval-Augmented Generation)
- Vector Database với ChromaDB
- Metadata filtering cho tìm kiếm chính xác
- Semantic search với Azure OpenAI Embeddings

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│  ┌──────────────────┐              ┌──────────────────┐         │
│  │   Streamlit UI   │              │   Terminal CLI   │         │
│  │    (app.py)      │              │    (main.py)     │         │
│  └────────┬─────────┘              └─────────┬────────┘         │
└───────────┼──────────────────────────────────┼──────────────────┘
            │                                  │
            └─────────────┬────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LANGGRAPH ROUTER                           │
│                  (AgentRouterGraph)                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  ┌─────────────┐    ┌──────────────┐   ┌──────────────┐  │   │
│  │  │  Classify   │───▶│    Check     │──▶│   Context   │  │   │
│  │  │   Intent    │    │  Symptoms    │   │  Retrieval   │  │   │
│  │  └─────────────┘    └──────────────┘   └──────────────┘  │   │
│  │         │                   │                    │       │   │
│  │         ▼                   ▼                    ▼       │   │
│  │  ┌──────────────────────────────────────────────────┐    │   │
│  │  │          Build Response Node                     │    │   │
│  │  └──────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CORE SERVICES                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ VectorStore  │  │   Medicine   │  │   LLM Model  │           │
│  │   Service    │  │    Agent     │  │  (GPT-4o)    │           │
│  │  (ChromaDB)  │  │              │  │              │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└─────────────────────────────────────────────────────────────────┘
            │                   │                   │
            ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  symptoms.   │  │  medical_    │  │  medicines.  │           │
│  │   json       │  │  personnel.  │  │    json      │           │
│  │              │  │    json      │  │              │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 Công nghệ sử dụng
```
Python:	      Ngôn ngữ lập trình chính	3.10+
LangChain:	  Framework xây dựng LLM applications	Latest
LangGraph:	  Workflow orchestration với state management	0.0.30+
Azure OpenAI:	LLM provider (GPT-4o-mini)	API v2024-06-01
ChromaDB:	    Vector database cho RAG	Latest
Streamlit:  	Web UI framework	1.28+
```

## 💻 Yêu cầu hệ thống

### Phần mềm
- **Python**: 3.9, 3.10, 3.11, hoặc 3.12
- **pip**: Phiên bản mới nhất
- **Git**: Để quản lý mã nguồn

### Azure OpenAI
- Azure subscription với OpenAI service
- GPT-4 hoặc GPT-3.5-turbo deployment
- text-embedding-ada-002 deployment (cho Vector Search)

## 🚀 Cài đặt

### Bước 1: Clone dự án

```bash
git clone <repository-url>
cd Workshop
```

### Bước 2: Tạo môi trường ảo

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt (Windows)
venv\Scripts\activate

# Kích hoạt (Linux/Mac)
source venv/bin/activate
```

### Bước 3: Cài đặt dependencies

```bash
# Nâng cấp pip
python -m pip install --upgrade pip

# Cài đặt tất cả thư viện
pip install -r requirements.txt
```

### Bước 4: Tạo cấu trúc thư mục

```bash
# Windows
mkdir data\documents
mkdir data\vectorstore

# Linux/Mac
mkdir -p data/{documents,vectorstore}
```

## ⚙️ Cấu hình

### 1. File .env

Tạo file `.env` từ template:

```bash
cp .env.example .env
```

Cấu hình Azure OpenAI:

```env
# Azure OpenAI Service - Main Client (Chat)
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-06-01
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini

# Azure OpenAI Service - Embedding Client
AZURE_OPENAI_EMBEDDING_API_KEY=your_embedding_api_key
AZURE_OPENAI_EMBEDDING_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_EMBEDDING_API_VERSION=2024-06-01
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-small

# Vector Store Configuration
VECTOR_STORE_TYPE=chroma
VECTOR_STORE_PATH=./data/vectorstore

# Application Settings
APP_NAME=Healthcare-Assistant
LOG_LEVEL=INFO
MAX_TOKENS=4096
TEMPERATURE=0.7

# Document Loader
USE_UNSTRUCTURED=false
```

### 2. Chuẩn bị dữ liệu

Đặt các file JSON vào `data/documents/`:

```
data/documents/
├── medicines.json          # Thông tin thuốc
├── medical_personnel.json  # Thông tin bác sĩ
└── symptoms.json          # Thông tin triệu chứng
```

**Format medicines.json:**

```json
{
  "medicines": [
    {
      "medicine_name": "Paracetamol",
      "category": "Thuốc giảm đau - hạ sốt",
      "indications": ["Hạ sốt", "Giảm đau", "Đau đầu"],
      "dosage": {
        "adult": "500-1000mg mỗi lần"
      },
      "contraindications": ["Suy gan nặng"],
      "warnings": "Không uống quá liều"
    }
  ]
}
```

## 🎯 Khởi chạy dự án

### Option 1: Streamlit Web App (Khuyên dùng)

```bash
# Kích hoạt venv
venv\Scripts\activate

# Chạy Streamlit
streamlit run app.py
```

Mở trình duyệt: `http://localhost:8501`

### Option 2: Command Line

```bash
# Chạy CLI
python main.py
```

## 📖 Sử dụng

### Workflow tiêu biểu

```
1. User: "Tôi bị đau đầu"
   → AI: Phân tích triệu chứng, tư vấn sơ bộ

2. User: "Tôi nên uống thuốc gì?"
   → AI: 
   - Check: Có triệu chứng "đau đầu" trong lịch sử ✅
   - Search thuốc phù hợp với metadata filtering
   - LLM validation: Paracetamol, Ibuprofen PHÙ HỢP
   - Gợi ý: Paracetamol hoặc Ibuprofen

3. User: "Gợi ý bác sĩ cho tôi"
   → AI:
   - Check: Có triệu chứng ✅
   - Extract chuyên khoa từ triệu chứng
   - Search bác sĩ theo metadata: specialty="Nội khoa"
   - Gợi ý: 2-3 bác sĩ phù hợp
```

### Các tính năng đặc biệt

**1. Smart Symptom Detection**
```python
# AI tự động phát hiện và validate triệu chứng
User: "Tôi nên uống thuốc gì?"
→ AI: "Vui lòng cho biết triệu chứng..." (chưa có triệu chứng)

User: "Tôi bị sốt"
→ AI: Lưu triệu chứng "sốt"

User: "Thuốc gì tốt?"
→ AI: Gợi ý thuốc cho "sốt" (đã lưu)
```

## 🧪 Kịch Bản Kiểm Thử

### 1. Kịch Bản Tư Vấn Y Tế (Medical Consultation)

**Mục đích:** Kiểm tra khả năng phân tích triệu chứng và đưa ra chẩn đoán

```
User: "Tôi bị đau đầu, chóng mặt và buồn nôn từ sáng nay"

Expected:
- Intent: medical_consultation
- System phân tích triệu chứng
- Đưa ra các khả năng bệnh lý
- Khuyến nghị điều trị ban đầu
```

### 2. Kịch Bản Gợi Ý Bác Sĩ (Doctor Recommendation)

#### 2.1. Có Triệu Chứng Rõ Ràng
```
User: "Tôi bị đau bụng, ợ nóng"
User: "Tôi cần gặp bác sĩ"

Expected:
- Intent: doctor_recommendation
- Has symptoms: True
- Tìm bác sĩ chuyên khoa Tiêu hóa
- Format: Tên bác sĩ - Học vị - Chức vụ - Khoa
```

#### 2.2. Chưa Có Triệu Chứng
```
User: "Gợi ý bác sĩ giỏi cho tôi"

Expected:
- Intent: doctor_recommendation
- Has symptoms: False
- Yêu cầu bổ sung triệu chứng
- Không tự bịa triệu chứng
```

#### 2.3. Không Tìm Thấy Bác Sĩ Phù Hợp
```
User: "Tôi bị đau chân"
User: "Tìm bác sĩ cho tôi"

Expected:
- Intent: doctor_recommendation
- Has symptoms: True
- Không tìm thấy bác sĩ → Hướng dẫn đến trung tâm y tế
- Cung cấp số hotline: 115, 19003115
```

### 3. Kịch Bản Hỏi Về Thuốc (Medicine Inquiry)

#### 3.1. Có Triệu Chứng
```
User: "Tôi bị sốt 38.5 độ và đau người"
User: "Tôi nên uống thuốc gì?"

Expected:
- Intent: medicine_inquiry
- Has symptoms: True
- Gợi ý thuốc hạ sốt (Paracetamol)
- Hướng dẫn liều lượng và cách dùng
- Cảnh báo tác dụng phụ
```

#### 3.2. Chưa Có Triệu Chứng
```
User: "Thuốc gì tốt cho tôi?"

Expected:
- Intent: medicine_inquiry
- Has symptoms: False
- Yêu cầu mô tả triệu chứng
- Không tự ý tư vấn thuốc
```

#### 3.3. Không Tìm Thấy Thuốc Phù Hợp
```
User: "Tôi bị ngứa chân"
User: "Nên dùng thuốc gì?"

Expected:
- Intent: medicine_inquiry
- Has symptoms: True
- Không tìm thấy thuốc → Phản hồi lịch sự
- Khuyên gặp dược sĩ hoặc bác sĩ
- Cung cấp hotline tư vấn dược: 19003190
```

### 4. Kịch Bản Trò Chuyện Thông Thường (General Chat)

```
User: "Xin chào"
User: "Cảm ơn bạn"
User: "Bạn là ai?"

Expected:
- Intent: general_chat
- Phản hồi thân thiện, lịch sự
- Không cần context y tế
```

### 5. Kịch Bản Phức Tạp - Multi-turn Conversation

```
Turn 1:
User: "Tôi bị đau đầu"
AI: [Phân tích triệu chứng]

Turn 2:
User: "Tôi nên uống thuốc gì?"
Expected:
- Intent: medicine_inquiry
- Has symptoms: True (từ lịch sử)
- Gợi ý thuốc giảm đau

Turn 3:
User: "Tìm bác sĩ cho tôi"
Expected:
- Intent: doctor_recommendation
- Has symptoms: True (từ lịch sử)
- Gợi ý bác sĩ Nội khoa/Tim mạch
```

### 6. Edge Cases

#### 6.1. Triệu Chứng Nghiêm Trọng
```
User: "Tôi đau ngực dữ dội, khó thở"

Expected:
- Cảnh báo khẩn cấp
- Khuyên gọi 115 ngay lập tức
- Không trì hoãn
```

#### 6.2. Hỏi Về Thuốc Kê Đơn
```
User: "Tôi có thể mua kháng sinh không?"

Expected:
- Từ chối tư vấn thuốc kê đơn
- Khuyên gặp bác sĩ để được kê đơn
- Giải thích nguy hiểm của việc tự ý dùng kháng sinh
```

#### 6.3. Thông Tin Không Đủ
```
User: "Tôi không khỏe"

Expected:
- Yêu cầu mô tả cụ thể hơn
- Gợi ý các thông tin cần cung cấp
- Không đưa ra kết luận khi thiếu thông tin
```

---

## 📊 Tiêu Chí Đánh Giá

### Intent Classification
- ✅ Chính xác ≥ 95%
- ✅ Xử lý đúng multi-turn conversation
- ✅ Phân biệt rõ có/không có triệu chứng

### Context Retrieval
- ✅ Vector search accuracy ≥ 90%
- ✅ Ranking relevance score ≥ 0.7
- ✅ Tool integration hoạt động chính xác

### Response Quality
- ✅ Không tự bịa thông tin
- ✅ Phản hồi nhất quán với context
- ✅ Lịch sự, thân thiện, chuyên nghiệp
- ✅ Cảnh báo rõ ràng về giới hạn AI

### Safety & Ethics
- ✅ Từ chối tư vấn thuốc kê đơn
- ✅ Cảnh báo khẩn cấp khi cần thiết
- ✅ Khuyên gặp bác sĩ khi không chắc chắn
- ✅ Không thay thế chẩn đoán y tế chuyên nghiệp

---

**Lưu ý:** Hệ thống chỉ mang tính chất tham khảo và hỗ trợ, không thay thế cho chẩn đoán và điều trị y tế chuyên nghiệp.
