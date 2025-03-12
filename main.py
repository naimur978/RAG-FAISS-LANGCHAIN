#!/usr/bin/env python
from langchain_ollama import OllamaLLM
from database import DatabaseManager
from embeddings import EmbeddingsManager
from agents import QueryClassifier, SQLAgent, RAGAgent
from config.config import Config


def load_project_content(file_path: str) -> list[str]:
    with open(file_path, 'r') as file:
        content = file.read()
    return content.split('##########')


class RAGApplication:
    def __init__(self):
        self.config = Config.load()
        self.llm = OllamaLLM(
            model=self.config.model_name, 
            temperature=self.config.temperature
        )
        self.db_manager = DatabaseManager()
        self.embeddings_manager = EmbeddingsManager(
            model_name=self.config.embedding_model
        )
        
        # Initialize knowledge base
        sections = load_project_content(self.config.knowledge_base_path)
        self.knowledge_base = self.embeddings_manager.init_knowledge_base(sections)
        
        # Initialize agents
        self.query_classifier = QueryClassifier(self.llm)
        self.sql_agent = SQLAgent(self.llm, self.db_manager)
        self.rag_agent = RAGAgent(
            self.llm, 
            self.embeddings_manager.get_retriever(
                k=self.config.retriever_k,
                fetch_k=self.config.retriever_fetch_k
            )
        )

    def process_query(self, query: str) -> str:
        try:
            # Classify query
            classification_response = self.query_classifier.classify_query(query)
            classification = self.query_classifier.decode_classification(
                classification_response, 
                query
            )
            
            query_type = classification['query_type']
            reasoning = classification['reasoning']
            
            print(f"\nQuery type: {query_type}")
            print(f"Reasoning: {reasoning}\n")

            # Process based on query type
            if query_type == "1":
                print("Using General LLM Intelligence")
                return self.llm.invoke(query)
            
            elif query_type == "2":
                print("Searching Project Knowledge Base")
                return self.rag_agent.query(query)
            
            elif query_type == "3":
                print("Querying Employee Database")
                sql_response = self.sql_agent.generate_sql_query(query)
                return self.sql_agent.execute_query(sql_response)
            
            else:
                raise ValueError(f"Invalid query type: {query_type}")
                
        except Exception as e:
            return f"Error processing query: {e}"

    def cleanup(self):
        self.db_manager.close()


def main():
    app = RAGApplication()
    
    # Example questions to demonstrate functionality
    questions = [
        # "Why is global warming on the rise?",
        
        # "Have we undertaken any projects related to robotics?",
        
        "Who are the employees in the company with more than three years of experience?",
        
        # "How does binary search algorithm work?",
        
        # "What are some projects on renewable energy?",
        
        # "What designations exist within the company?"
    ]

    # Process questions
    for query in questions:
        print(f"\n{'='*80}\n")
        print(f"Query: {query}")
        answer = app.process_query(query)
        print(f"\nAnswer: {answer}")
        print(f"\n{'='*80}")

    # Cleanup
    app.cleanup()


if __name__ == "__main__":
    main()