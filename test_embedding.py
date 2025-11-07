# test_embedding.py
import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from pydantic import SecretStr

load_dotenv()

try:
    print("🔧 Cấu hình embedding:")
    api_key = os.getenv('AZURE_OPENAI_EMBEDDING_API_KEY') or os.getenv('AZURE_OPENAI_API_KEY')
    endpoint = os.getenv('AZURE_OPENAI_EMBEDDING_ENDPOINT') or os.getenv('AZURE_OPENAI_ENDPOINT')
    deployment = os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME', 'text-embedding-3-small')
    api_version = os.getenv('AZURE_OPENAI_EMBEDDING_API_VERSION', '2024-02-15-preview')
    
    print(f"   API Key: {api_key[:10] if api_key else 'Not set'}...")
    print(f"   Endpoint: {endpoint}")
    print(f"   Deployment: {deployment}")
    print(f"   API Version: {api_version}")
    # Tạo embeddings với các parameter explicit
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=endpoint,
        api_key=SecretStr(api_key) if api_key else None,
        api_version=api_version,
        azure_deployment=deployment,  # Sử dụng azure_deployment thay vì deployment
        model=deployment  # Thêm model parameter
    )
    print("✅ Embeddings khởi tạo thành công")
    
    # Test embed một câu ngắn
    test_text = "Hello world"
    result = embeddings.embed_query(test_text)
    print(f"✅ Test embedding thành công, vector size: {len(result)}")
    
except Exception as e:
    print(f"❌ Lỗi embedding: {str(e)}")