import os
from dataclasses import dataclass
from dotenv import load_dotenv
import torch

@dataclass
class Config:
    # LLM Settings
    model_name: str = "llama2"
    temperature: float = 0
    
    # Embedding Settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # File Paths
    knowledge_base_path: str = "knowledge-base/Projects.txt"
    
    # FAISS Settings
    retriever_k: int = 5
    retriever_fetch_k: int = 8

    @classmethod
    def load(cls):
        load_dotenv()
        return cls()