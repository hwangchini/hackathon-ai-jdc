import os
import sys

# ✅ Fix import path khi chạy trực tiếp file này
if __name__ == "__main__":
    # Thêm thư mục gốc vào sys.path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
from src.models.llm import get_embeddings
import json
import shutil

load_dotenv()


class VectorStoreService:
    """Service for managing vector store operations"""

    def __init__(self):
        try:
            self.embeddings = get_embeddings()
        except Exception as e:
            print(f"❌ Lỗi khởi tạo embeddings: {str(e)}")
            raise e
            
        chunk_size = int(os.getenv('CHUNK_SIZE', '1000'))
        chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '200'))
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.vector_store = None

    def create_vector_store(self, documents: List[Document]):
        """Create vector store from documents"""
        # Phân loại documents: JSON không split, còn lại thì split
        json_docs = [doc for doc in documents if doc.metadata.get('file_type') == 'json']
        other_docs = [doc for doc in documents if doc.metadata.get('file_type') != 'json']
        
        # Split các documents không phải JSON
        splits = self.text_splitter.split_documents(other_docs) if other_docs else []
        
        # Kết hợp: JSON documents giữ nguyên + các documents khác đã split
        all_documents = json_docs + splits
        
        vector_store_path = os.getenv('VECTOR_STORE_PATH', './data/vectorstore')
        
        print(f"📊 Tổng số documents: {len(all_documents)} (JSON: {len(json_docs)}, Splits: {len(splits)})")
        
        self.vector_store = Chroma.from_documents(
            documents=all_documents,
            embedding=self.embeddings,
            persist_directory=vector_store_path,
        )
        return self.vector_store

    def load_vector_store(self):
        """Load existing vector store"""
        vector_store_path = os.getenv('VECTOR_STORE_PATH', './data/vectorstore')
        
        if not os.path.exists(vector_store_path):
            raise FileNotFoundError("Vector store không tồn tại")
        
        try:
            self.vector_store = Chroma(
                persist_directory=vector_store_path,
                embedding_function=self.embeddings,
            )
            return self.vector_store
        except Exception as e:
            raise e

    def get_retriever(self, search_type: str = "similarity", k: int = 4):
        """
        Tạo retriever từ vector store
        
        Args:
            search_type: "similarity" hoặc "mmr" (Maximum Marginal Relevance)
            k: Số lượng documents trả về
            
        Returns:
            Retriever object
        """
        if not self.vector_store:
            self.load_vector_store()
        
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )

    def similarity_search(self, query: str, k: int = 4):
        """Perform similarity search (giữ lại để backward compatible)"""
        try:
            if not self.vector_store:
                self.load_vector_store()
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            if "key_model_access_denied" in str(e):
                print(f"❌ Lỗi model embedding: {os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME', 'text-embedding-ada-002')}")
            raise e

    def retrieve_with_score(self, query: str, k: int = 4):
        """Retrieve documents with relevance scores"""
        if not self.vector_store:
            self.load_vector_store()
        
        return self.vector_store.similarity_search_with_score(query, k=k)

    def need_update(self, documents: List[Document]) -> bool:
        """Kiểm tra xem vector store có cần cập nhật không"""
        if not self.vector_store:
            return True
        
        try:
            # So sánh số lượng documents
            current_count = len(self.vector_store.get()['ids'])
            new_count = len(documents)
            
            return current_count != new_count
        except:
            return True
    
    def update_vector_store(self, documents: List[Document]):
        """Cập nhật vector store (xóa cũ và tạo mới)"""
        vector_store_path = os.getenv('VECTOR_STORE_PATH', './data/vectorstore')
        
        # Xóa vector store cũ
        if os.path.exists(vector_store_path):
            shutil.rmtree(vector_store_path)
            print("🗑️ Đã xóa vector store cũ")
        
        # Tạo mới
        return self.create_vector_store(documents)

    def similarity_search_with_scores(self, query: str, k: int = 4):
        """
        Perform similarity search with cosine similarity scores
        
        Returns:
            List of (Document, score) tuples
        """
        try:
            if not self.vector_store:
                self.load_vector_store()
            
            # Lấy kết quả kèm scores
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            # results = [(doc1, 0.85), (doc2, 0.72), ...]
            return results
        except Exception as e:
            if "key_model_access_denied" in str(e):
                print(f"❌ Lỗi model embedding: {os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME', 'text-embedding-ada-002')}")
            raise e

    def similarity_search_with_filter(self, query: str, k: int = 4, filter_dict: dict = None):
        """
        Similarity search với metadata filtering
        
        Args:
            query: Query text
            k: Số kết quả
            filter_dict: Metadata filter, VD: {"filename": "medicines.json"}
            
        Returns:
            List of Documents
        """
        try:
            if not self.vector_store:
                self.load_vector_store()
            
            # ChromaDB filter syntax
            results = self.vector_store.similarity_search(
                query, 
                k=k,
                filter=filter_dict
            )
            
            return results
        except Exception as e:
            print(f"⚠️ Lỗi search with filter: {str(e)}")
            return []
    
    def similarity_search_with_filter_and_scores(self, query: str, k: int = 4, filter_dict: dict = None):
        """
        Similarity search với metadata filtering + scores
        
        Args:
            query: Query text
            k: Số kết quả
            filter_dict: Metadata filter
            
        Returns:
            List of (Document, score) tuples
        """
        try:
            if not self.vector_store:
                self.load_vector_store()
            
            results = self.vector_store.similarity_search_with_score(
                query,
                k=k,
                filter=filter_dict
            )
            
            return results
        except Exception as e:
            print(f"⚠️ Lỗi search with filter and scores: {str(e)}")
            return []
    
    def _process_medicines_json(self, file_path: str, filename: str) -> List[Document]:
        """Process medicines.json - Đảm bảo lưu đầy đủ metadata"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            medicines = data.get('medicines', [])
            
            print(f"\n🔍 Processing {len(medicines)} medicines from {filename}")
            print("="*60)
            
            for medicine in medicines:
                medicine_name = medicine.get('medicine_name', 'Unknown')
                generic_name = medicine.get('generic_name', '')
                category = medicine.get('category', '')
                brand_names = ', '.join(medicine.get('brand_names', []))
                
                # ✅ ĐỌC ĐÚNG TỪ JSON - QUAN TRỌNG!
                source = medicine.get('source', '')
                reference_url = medicine.get('reference_url', '')
                last_updated = medicine.get('last_updated', '')
                
                # ✅ Debug: In ra để kiểm tra
                print(f"\n📌 {medicine_name}:")
                print(f"   - source: '{source}' {'✅' if source else '❌ MISSING'}")
                print(f"   - reference_url: '{reference_url}' {'✅' if reference_url else '❌ MISSING'}")
                print(f"   - last_updated: '{last_updated}' {'✅' if last_updated else '❌ MISSING'}")
                
                # Indications
                indications = medicine.get('indications', [])
                indications_text = ', '.join(indications) if indications else ''
                indications_str = f"Chỉ định: {indications_text}" if indications_text else ""
                
                # Dosage
                dosage = medicine.get('dosage', {})
                dosage_str = "Liều dùng:\n"
                if dosage:
                    for key, value in dosage.items():
                        dosage_str += f"  - {key}: {value}\n"
            
                # Other fields
                contraindications = medicine.get('contraindications', [])
                contra_str = f"Chống chỉ định: {', '.join(contraindications)}" if contraindications else ""
                
                side_effects = medicine.get('side_effects', [])
                side_str = f"Tác dụng phụ: {', '.join(side_effects)}" if side_effects else ""
                
                warnings = medicine.get('warnings', '')
                warnings_str = f"Cảnh báo: {warnings}" if warnings else ""
                
                # Build content
                content_parts = [
                    f"Tên thuốc: {medicine_name}",
                    f"Tên generic: {generic_name}" if generic_name else "",
                    f"Tên thương mại: {brand_names}" if brand_names else "",
                    f"Loại: {category}" if category else "",
                    indications_str,
                    dosage_str,
                    contra_str,
                    side_str,
                    warnings_str
                ]
                
                content = "\n".join([part for part in content_parts if part])
                
                # ✅ METADATA - LƯU Ý: source, reference_url, last_updated từ JSON
                metadata = {
                    'filename': filename,
                    'item_name': medicine_name,
                    'category': category,
                    'indications_text': indications_text,
                    'source': source,  # ✅ Từ JSON
                    'reference_url': reference_url,  # ✅ Từ JSON
                    'last_updated': last_updated  # ✅ Từ JSON
                }
                
                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                documents.append(doc)
        
            print(f"\n{'='*60}")
            print(f"✅ Processed {len(documents)} medicines with metadata")
            print(f"{'='*60}\n")
        
            return documents
        
        except Exception as e:
            print(f"⚠️ Error processing medicines.json: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

if __name__ == "__main__":
    """Rebuild vector store từ documents"""
    print("\n" + "="*60)
    print("🔄 REBUILDING VECTOR STORE")
    print("="*60 + "\n")
    
    # Initialize service
    service = VectorStoreService()
    
    # Load documents
    documents_path = "./data/documents"
    
    if not os.path.exists(documents_path):
        print(f"❌ Thư mục documents không tồn tại: {documents_path}")
        sys.exit(1)
    
    all_documents = []
    
    # Process JSON files
    for filename in os.listdir(documents_path):
        if not filename.endswith('.json'):
            continue
        
        file_path = os.path.join(documents_path, filename)
        print(f"\n📄 Processing: {filename}")
        
        if filename == "medicines.json":
            docs = service._process_medicines_json(file_path, filename)
            all_documents.extend(docs)
        else:
            # Process other JSON files
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Generic JSON processing
                content = json.dumps(data, ensure_ascii=False, indent=2)
                doc = Document(
                    page_content=content,
                    metadata={
                        'filename': filename,
                        'file_type': 'json',
                        'source': file_path
                    }
                )
                all_documents.append(doc)
                print(f"✅ Processed {filename}")
            except Exception as e:
                print(f"⚠️ Error processing {filename}: {str(e)}")
    
    # Create vector store
    if all_documents:
        print(f"\n{'='*60}")
        print(f"📊 Tổng số documents: {len(all_documents)}")
        print(f"{'='*60}\n")
        
        print("🔨 Creating vector store...")
        service.create_vector_store(all_documents)
        
        vector_store_path = os.getenv('VECTOR_STORE_PATH', './data/vectorstore')
        print(f"\n✅ Vector store created successfully at: {vector_store_path}")
        print(f"{'='*60}\n")
    else:
        print("\n❌ No documents found to process")
        sys.exit(1)

