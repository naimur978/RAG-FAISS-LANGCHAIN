from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import Config

class EmbeddingsManager:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        config = Config.load()
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': config.device}
        )
        self.knowledge_base = None

    def init_knowledge_base(self, text_sections):
        self.knowledge_base = FAISS.from_texts(text_sections, self.embeddings)
        return self.knowledge_base

    def get_retriever(self, k=5, fetch_k=8):
        if not self.knowledge_base:
            raise ValueError("Knowledge base not initialized")
        return self.knowledge_base.as_retriever(search_kwargs={"k": k, "fetch_k": fetch_k})