import os
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

load_dotenv()


def get_llm(streaming: bool = True):
    """
    Initialize Azure OpenAI LLM
    
    Args:
        streaming: Enable streaming response (default: True)
    """
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT') 
    deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME','gpt-4o-mini')
    api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-06-01')
    temperature = float(os.getenv('TEMPERATURE', '0.7'))
    max_tokens = int(os.getenv('MAX_TOKENS', '4096'))
    
    return AzureChatOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        deployment_name=deployment,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming  # ← Enable streaming
    )


def get_embeddings():
    """Initialize Azure OpenAI Embeddings"""
    api_key = os.getenv('AZURE_OPENAI_EMBEDDING_API_KEY') or os.getenv('AZURE_OPENAI_API_KEY')
    endpoint = os.getenv('AZURE_OPENAI_EMBEDDING_ENDPOINT') or os.getenv('AZURE_OPENAI_ENDPOINT')
    deployment = os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME', 'text-embedding-3-small')
    api_version = os.getenv('AZURE_OPENAI_EMBEDDING_API_VERSION', '2024-02-15-preview')
    
    return AzureOpenAIEmbeddings(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        azure_deployment=deployment,  # Sử dụng azure_deployment
        model=deployment  # Thêm model parameter để force model name
    )

