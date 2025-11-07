import os
import sys
from typing import List, Optional
from pathlib import Path
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)
import json
import csv
from bs4 import BeautifulSoup


class DocumentLoader:
    
    def __init__(self, use_unstructured: bool = False):
        """
        Initialize DocumentLoader
        
        Args:
            use_unstructured: Sử dụng UnstructuredFileLoader cho auto-detection
        """
        self.use_unstructured = use_unstructured
        self.supported_extensions = {
            '.pdf': self._load_pdf,
            '.txt': self._load_text,
            '.docx': self._load_docx,
            '.doc': self._load_docx,
            '.html': self._load_html,
            '.htm': self._load_html,
            '.json': self._load_json,
            '.csv': self._load_csv,
            '.md': self._load_markdown
        }
        
        # Check if unstructured is available
        if self.use_unstructured:
            try:
                from langchain_community.document_loaders import UnstructuredFileLoader
                self.unstructured_available = True
            except ImportError:
                print("⚠️ UnstructuredFileLoader not available. Install: pip install unstructured python-magic-bin")
                self.unstructured_available = False
                self.use_unstructured = False
    
    def load_documents_from_folder(self, folder_path: str) -> List[Document]:
        """
        Load all supported documents from a folder
        
        Args:
            folder_path: Path to the folder containing documents
            
        Returns:
            List of Document objects
        """
        documents = []
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            return documents
        
        for file_path in folder_path.rglob("*"):
            if file_path.is_file():
                try:
                    docs = self.load_document(str(file_path))
                    if docs:
                        documents.extend(docs)
                except Exception as e:
                    if 'streamlit' not in sys.modules:
                        print(f"❌ Lỗi tải {file_path.name}: {str(e)}")
        
        return documents
    
    def load_document(self, file_path: str) -> Optional[List[Document]]:
        """
        Load a single document - Hybrid approach
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects or None if unsupported
        """
        file_path_obj = Path(file_path)
        extension = file_path_obj.suffix.lower()
        
        # Try UnstructuredFileLoader first if enabled
        if self.use_unstructured and self.unstructured_available:
            try:
                from langchain_community.document_loaders import UnstructuredFileLoader
                
                loader = UnstructuredFileLoader(str(file_path_obj))
                docs = loader.load()
                
                # Add custom metadata
                for doc in docs:
                    doc.metadata.update({
                        'source': str(file_path_obj),
                        'file_type': extension[1:] if extension else 'unknown',
                        'filename': file_path_obj.name,
                        'loader_type': 'unstructured'
                    })
                
                return docs
                
            except Exception as e:
                if 'streamlit' not in sys.modules:
                    print(f"⚠️ Unstructured failed for {file_path_obj.name}, using custom loader: {str(e)}")
                # Fallback to custom loader
                pass
        
        # Custom loader (existing logic)
        if extension not in self.supported_extensions:
            return None
        
        try:
            docs = self.supported_extensions[extension](file_path_obj)
            
            # Add loader type to metadata
            if docs:
                for doc in docs:
                    doc.metadata['loader_type'] = 'custom'
            
            return docs
        except Exception:
            return None
    
    def _load_pdf(self, file_path: Path) -> List[Document]:
        """Load PDF document"""
        loader = PyPDFLoader(str(file_path))
        documents = loader.load()
        
        for doc in documents:
            doc.metadata.update({
                'source': str(file_path),
                'file_type': 'pdf',
                'filename': file_path.name
            })
        
        return documents
    
    def _load_text(self, file_path: Path) -> List[Document]:
        """Load text document"""
        try:
            loader = TextLoader(str(file_path), encoding='utf-8')
            documents = loader.load()
        except UnicodeDecodeError:
            loader = TextLoader(str(file_path), encoding='latin-1')
            documents = loader.load()
        
        for doc in documents:
            doc.metadata.update({
                'source': str(file_path),
                'file_type': 'text',
                'filename': file_path.name
            })
        
        return documents
    
    def _load_docx(self, file_path: Path) -> List[Document]:
        """Load Word document"""
        loader = Docx2txtLoader(str(file_path))
        documents = loader.load()
        
        for doc in documents:
            doc.metadata.update({
                'source': str(file_path),
                'file_type': 'docx',
                'filename': file_path.name
            })
        
        return documents
    
    def _load_html(self, file_path: Path) -> List[Document]:
        """Load HTML document"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
        
        soup = BeautifulSoup(content, 'html.parser')
        
        for script in soup(["script", "style"]):
            script.decompose()
        
        text_content = soup.get_text()
        lines = (line.strip() for line in text_content.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        title = soup.title.string if soup.title and soup.title.string else file_path.stem
        
        document = Document(
            page_content=text,
            metadata={
                'source': str(file_path),
                'file_type': 'html',
                'filename': file_path.name,
                'title': title
            }
        )
        
        return [document]
    
    def _load_json(self, file_path: Path) -> List[Document]:
        """Load JSON document"""
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        documents = []
        
        # Tự động phát hiện JSON array
        detected_array = self._detect_json_array(data)
        
        if detected_array:
            array_name, array_data, item_name_key = detected_array
            
            for idx, item in enumerate(array_data):
                if isinstance(item, dict):
                    item_name = item.get(item_name_key, f"{array_name} #{idx+1}") if item_name_key else f"{array_name} #{idx+1}"
                    
                    text_content = self._format_json_item(item, item_name)
                    
                    metadata = {
                        'source': str(file_path),
                        'file_type': 'json',
                        'filename': file_path.name,
                        'array_name': array_name,
                        'item_name': item_name,
                        'index': idx
                    }
                    
                    # ✅ Thêm metadata đặc biệt cho medicines.json
                    if file_path.name == 'medicines.json':
                        metadata['medicine_name'] = item.get('medicine_name')
                        metadata['category'] = item.get('category')
                        
                        # ✅ QUAN TRỌNG: Lưu indications dạng text CHÍNH XÁC
                        if 'indications' in item and isinstance(item['indications'], list):
                            # Giữ nguyên case, chỉ join
                            indications_original = ', '.join(item['indications'])
                            metadata['indications_text'] = indications_original
                    
                    # Metadata cho medical_personnel.json
                    if file_path.name == 'medical_personnel.json':
                        metadata['department_name'] = item.get('department_name')
                        metadata['specialty_name'] = item.get('specialty')
                    
                    document = Document(
                        page_content=text_content,
                        metadata=metadata
                    )
                    documents.append(document)
            
            return documents
        
        # JSON thông thường
        else:
            if isinstance(data, dict):
                text_content = self._dict_to_text(data)
            elif isinstance(data, list):
                text_content = "\n\n".join([
                    self._dict_to_text(item) if isinstance(item, dict) else str(item)
                    for item in data
                ])
            else:
                text_content = str(data)
            
            document = Document(
                page_content=text_content,
                metadata={
                    'source': str(file_path),
                    'file_type': 'json',
                    'filename': file_path.name
                }
            )
            
            return [document]
    
    def _detect_json_array(self, data: dict) -> Optional[tuple]:
        """
        Tự động phát hiện array lớn trong JSON và key name phù hợp
        
        Returns:
            tuple(array_name, array_data, item_name_key) hoặc None
        """
        # Nếu data là list trực tiếp và đủ lớn
        if isinstance(data, list) and len(data) > 3:
            # Tìm key name phổ biến trong các items
            item_name_key = self._find_name_key(data[0] if data else {})
            return ("items", data, item_name_key)
        
        # Nếu data là dict, tìm key chứa array lớn
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 3:
                    # Tìm key name phù hợp cho items trong array
                    item_name_key = self._find_name_key(value[0] if value else {})
                    return (key, value, item_name_key)
        
        return None
    
    def _find_name_key(self, item: dict) -> Optional[str]:
        """
        Tìm key phù hợp để làm tên cho item (name, title, symptom_name, ...)
        """
        if not isinstance(item, dict):
            return None
        
        # Danh sách các key thường dùng làm tên (theo thứ tự ưu tiên)
        name_candidates = [
            'name', 'title', 'symptom_name', 'disease_name', 'product_name',
            'label', 'heading', 'subject', 'topic', 'category', 'id'
        ]
        
        # Tìm key khớp chính xác
        for key in name_candidates:
            if key in item:
                return key
        
        # Tìm key chứa 'name' hoặc 'title'
        for key in item.keys():
            if 'name' in key.lower() or 'title' in key.lower():
                return key
        
        # Lấy key đầu tiên có giá trị string
        for key, value in item.items():
            if isinstance(value, str) and len(value) < 100:
                return key
        
        return None
    
    def _format_json_item(self, item: dict, item_name: str) -> str:
        """
        Format JSON item thành text có cấu trúc rõ ràng
        """
        lines = [f"{'='*60}", f"{item_name.upper()}", f"{'='*60}", ""]
        
        for key, value in item.items():
            # Bỏ qua key name đã dùng làm title
            if str(value) == item_name:
                continue
            
            # Format key thành tiêu đề đẹp
            formatted_key = key.replace('_', ' ').title()
            
            if isinstance(value, list):
                lines.append(f"{formatted_key}:")
                lines.append(self._format_list(value))
            elif isinstance(value, dict):
                lines.append(f"{formatted_key}:")
                lines.append(self._dict_to_text(value, indent=1))
            else:
                lines.append(f"{formatted_key}:")
                lines.append(str(value))
            
            lines.append("")  # Dòng trống giữa các sections
        
        return "\n".join(lines)
    
    def _format_list(self, items: list) -> str:
        """Format list items với dấu bullet"""
        if not items:
            return "Không có thông tin"
        return "\n".join([f"• {item}" for item in items])
    
    def _load_csv(self, file_path: Path) -> List[Document]:
        """Load CSV document"""
        documents = []
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            sample = file.read(1024)
            file.seek(0)
            
            try:
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
            except csv.Error:
                delimiter = ','
            
            reader = csv.DictReader(file, delimiter=delimiter)
            
            for i, row in enumerate(reader):
                text_content = "\n".join([f"{key}: {value}" for key, value in row.items() if value])
                
                document = Document(
                    page_content=text_content,
                    metadata={
                        'source': str(file_path),
                        'file_type': 'csv',
                        'filename': file_path.name,
                        'row_number': i + 1
                    }
                )
                documents.append(document)
        
        return documents
    
    def _load_markdown(self, file_path: Path) -> List[Document]:
        """Load Markdown document"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
        
        document = Document(
            page_content=content,
            metadata={
                'source': str(file_path),
                'file_type': 'markdown',
                'filename': file_path.name
            }
        )
        
        return [document]
    
    def _dict_to_text(self, data: dict, indent: int = 0) -> str:
        """Convert dictionary to readable text format"""
        lines = []
        prefix = "  " * indent
        
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(self._dict_to_text(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}:")
                for item in value:
                    if isinstance(item, dict):
                        lines.append(self._dict_to_text(item, indent + 1))
                    else:
                        lines.append(f"{prefix}  - {item}")
            else:
                lines.append(f"{prefix}{key}: {value}")
        
        return "\n".join(lines)
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        return list(self.supported_extensions.keys())
    
    def is_supported_format(self, file_path: str) -> bool:
        """Check if file format is supported"""
        extension = Path(file_path).suffix.lower()
        return extension in self.supported_extensions