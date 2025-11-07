import os
import shutil
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
from src.models.llm import get_embeddings

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

