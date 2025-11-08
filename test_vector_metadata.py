"""Script kiểm tra metadata trong VectorDB"""

from src.services.vector_store import VectorStoreService

def test_medicine_metadata():
    """Kiểm tra metadata của thuốc trong VectorDB"""
    print("🔍 Kiểm tra metadata trong VectorDB...\n")
    
    # Initialize service
    vector_service = VectorStoreService()
    
    # Test query với tên thuốc chính xác
    test_medicine = "Ibuprofen"
    print(f"📊 Tìm kiếm: {test_medicine}")
    
    results = vector_service.similarity_search_with_filter_and_scores(
        query=test_medicine,
        k=3,
        filter_dict={"filename": "medicines.json"}
    )
    
    if not results:
        print("❌ Không tìm thấy kết quả")
        return
    
    print(f"\n✅ Tìm thấy {len(results)} kết quả\n")
    
    for i, (doc, score) in enumerate(results, 1):
        item_name = doc.metadata.get('item_name', 'Unknown')
        print(f"{i}. {item_name} (Score: {score:.4f})")
        
        # Check if this is the medicine we're looking for
        if test_medicine.lower() in item_name.lower() or test_medicine.lower() in doc.page_content.lower():
            print(f"\n📋 METADATA CHI TIẾT:")
            print(f"   - filename: {doc.metadata.get('filename', 'MISSING')}")
            print(f"   - item_name: {doc.metadata.get('item_name', 'MISSING')}")
            print(f"   - category: {doc.metadata.get('category', 'MISSING')}")
            print(f"   - source: {doc.metadata.get('source', 'MISSING')}")
            print(f"   - reference_url: {doc.metadata.get('reference_url', 'MISSING')}")
            print(f"   - last_updated: {doc.metadata.get('last_updated', 'MISSING')}")
            
            print(f"\n📄 CONTENT (first 300 chars):")
            print(doc.page_content[:300])
            print("\n" + "="*60)
            
            # Final check
            source = doc.metadata.get('source', '')
            reference_url = doc.metadata.get('reference_url', '')
            
            if source and reference_url:
                print("\n✅ CẢ source VÀ reference_url ĐỀU CÓ trong VectorDB")
            elif source:
                print("\n⚠️ Chỉ có source, THIẾU reference_url")
            elif reference_url:
                print("\n⚠️ Chỉ có reference_url, THIẾU source")
            else:
                print("\n❌ THIẾU CẢ source VÀ reference_url - CẦN REBUILD!")
                print("\n💡 Hướng dẫn rebuild:")
                print("   1. Xóa thư mục: data/chroma_db/")
                print("   2. Chạy lại app hoặc: python src/services/vector_store.py")
            
            break
    else:
        print(f"\n⚠️ Không tìm thấy thuốc '{test_medicine}' chính xác trong kết quả")

if __name__ == "__main__":
    test_medicine_metadata()
