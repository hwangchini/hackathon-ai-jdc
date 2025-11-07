import os
from dotenv import load_dotenv
from src.services.vector_store import VectorStoreService
from src.utils.document_loader import DocumentLoader

load_dotenv()

print("="*80)
print("🔍 DEBUG VECTORDB - KIỂM TRA THÔNG TIN BÁC SĨ")
print("="*80)

# 1. Load documents
print("\n📚 BƯỚC 1: Load documents từ thư mục...")
loader = DocumentLoader(use_unstructured=False)
documents = loader.load_documents_from_folder("./data/documents")

print(f"✅ Tổng số documents: {len(documents)}")

# 2. Kiểm tra medical_personnel.json
print("\n🏥 BƯỚC 2: Kiểm tra medical_personnel.json...")
medical_docs = [doc for doc in documents if doc.metadata.get('filename') == 'medical_personnel.json']
print(f"📋 Documents từ medical_personnel.json: {len(medical_docs)}")

if medical_docs:
    print("\n--- Danh sách khoa trong medical_personnel.json ---")
    for i, doc in enumerate(medical_docs, 1):
        dept = doc.metadata.get('department_name', 'N/A')
        specialty = doc.metadata.get('specialty_name', 'N/A')
        doctor_count = doc.metadata.get('doctor_count', 0)
        print(f"{i}. {dept} (Chuyên khoa: {specialty}) - {doctor_count} bác sĩ")
    
    # Hiển thị chi tiết 3 khoa đầu
    print("\n--- Chi tiết 3 khoa đầu tiên ---")
    for i, doc in enumerate(medical_docs[:3], 1):
        print(f"\n{'='*60}")
        print(f"Document {i}:")
        print(f"Department: {doc.metadata.get('department_name')}")
        print(f"Specialty: {doc.metadata.get('specialty_name')}")
        print(f"Content preview:\n{doc.page_content[:400]}...")
else:
    print("❌ KHÔNG tìm thấy medical_personnel.json!")

# 3. Tạo vector store
print("\n🔧 BƯỚC 3: Tạo vector store...")
vector_service = VectorStoreService()

# Phân loại documents
json_docs = [doc for doc in documents if doc.metadata.get('file_type') == 'json']
other_docs = [doc for doc in documents if doc.metadata.get('file_type') != 'json']
print(f"   JSON docs: {len(json_docs)} (không split)")
print(f"   Other docs: {len(other_docs)} (sẽ split)")

vector_service.create_vector_store(documents)
print("✅ Vector store đã được tạo")

# 4. Test các query khác nhau
print("\n🔍 BƯỚC 4: Test search queries...")
test_queries = [
    "bác sĩ đau đầu",
    "bác sĩ nội khoa",
    "bác sĩ nội thần kinh",
    "bác sĩ tim mạch",
    "bác sĩ tiêu hóa",
    "khoa tim mạch"
]

for query in test_queries:
    print(f"\n--- Query: '{query}' ---")
    results = vector_service.similarity_search(query, k=3)
    print(f"Số kết quả: {len(results)}")
    
    for i, doc in enumerate(results, 1):
        filename = doc.metadata.get('filename', 'N/A')
        dept = doc.metadata.get('department_name', doc.metadata.get('item_name', 'N/A'))
        specialty = doc.metadata.get('specialty_name', 'N/A')
        
        print(f"{i}. File: {filename}")
        print(f"   Dept: {dept} | Specialty: {specialty}")
        print(f"   Content: {doc.page_content[:150]}...")

# 5. Kiểm tra cụ thể cho "Nội khoa"
print("\n" + "="*80)
print("🎯 BƯỚC 5: Tìm kiếm chi tiết cho 'Nội khoa'...")
print("="*80)

internal_medicine_docs = [doc for doc in medical_docs 
                         if doc.metadata.get('specialty_name') == 'Nội khoa']
print(f"\n📊 Số khoa thuộc Nội khoa: {len(internal_medicine_docs)}")

if internal_medicine_docs:
    for doc in internal_medicine_docs:
        print(f"\n✓ {doc.metadata.get('department_name')}")
        print(f"  Content:\n{doc.page_content[:300]}")
        print("  ...")

# 6. Test search với metadata filtering
print("\n🔍 BƯỚC 6: Test search có filter metadata...")
results = vector_service.similarity_search("bác sĩ đau đầu nội khoa", k=10)
print(f"Tổng kết quả: {len(results)}")

medical_results = [doc for doc in results 
                  if doc.metadata.get('filename') == 'medical_personnel.json']
print(f"Kết quả từ medical_personnel.json: {len(medical_results)}")

if medical_results:
    print("\n--- Kết quả từ medical_personnel.json ---")
    for i, doc in enumerate(medical_results[:5], 1):
        print(f"\n{i}. {doc.metadata.get('department_name')} - {doc.metadata.get('specialty_name')}")
        print(f"   {doc.page_content[:200]}...")
else:
    print("❌ KHÔNG có kết quả từ medical_personnel.json!")

print("\n" + "="*80)
print("✅ HOÀN TẤT DEBUG")
print("="*80)
