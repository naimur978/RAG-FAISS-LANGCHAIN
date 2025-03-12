import json
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from config import Config


class QueryClassifier:
    def __init__(self, llm):
        self.llm = llm
        self.config = Config.load()

    def classify_query(self, query):
        prompt = f"""
        You are a decision-making assistant for Company XYZ. Classify this query into exactly ONE of these types:
        1. General Knowledge Query: Questions about general topics, concepts, or algorithms that don't require company data
        2. Project Query: Questions specifically about company projects, technologies used, or project outcomes
        3. Employee Query: Questions about employee data, including names, emails, designations, salary, etc.
        
        User Query: {query}
        
        Respond ONLY in this exact JSON format:
        {{
            "query_type": "1 or 2 or 3",
            "reasoning": "brief explanation"
        }}
        """
        return self.llm.invoke(prompt)

    def decode_classification(self, res, query):
        try:
            query_lower = query.lower()
            if any(term in query_lower for term in ["employee", "salary", "designation", "experience"]):
                return {"query_type": "3", "reasoning": "Query contains employee-related terms"}
            elif any(term in query_lower for term in ["we", "our", "company", "project", "projects"]):
                return {"query_type": "2", "reasoning": "Query contains company/project-related terms"}
            
            start_index = res.find("{")
            while start_index != -1:
                try:
                    brace_count = 1
                    pos = start_index + 1
                    while pos < len(res) and brace_count > 0:
                        if res[pos] == '{':
                            brace_count += 1
                        elif res[pos] == '}':
                            brace_count -= 1
                        pos += 1
                    
                    if brace_count == 0:
                        json_str = res[start_index:pos]
                        parsed = json.loads(json_str)
                        if "query_type" in parsed and "reasoning" in parsed:
                            parsed["query_type"] = parsed["query_type"].strip()
                            return parsed
                except:
                    pass
                start_index = res.find("{", pos)
            
            return {"query_type": "1", "reasoning": "General knowledge query"}
        except Exception as e:
            raise ValueError(f"Classification error: {e}")


class SQLAgent:
    def __init__(self, llm, db_manager):
        self.llm = llm
        self.db_manager = db_manager
        self.config = Config.load()

    def generate_sql_query(self, user_query):
        prompt = f"""
        You are an expert SQL assistant. Generate an SQLite3 query for the following request.
        Always include relevant fields that would be helpful to the user, not just what they explicitly ask for.
        
        Table Schema:
        Employee (
            employee_name TEXT NOT NULL,
            employee_email TEXT UNIQUE NOT NULL,
            org_name TEXT,
            designation TEXT NOT NULL,
            years_of_experience REAL NOT NULL,
            salary REAL,
            location TEXT,
            hire_date TEXT
        )
        
        Guidelines:
        1. For experience queries, include name, designation, and years_of_experience
        2. For salary queries, include name, designation, and salary
        3. For location queries, include name, designation, and location
        4. For general employee queries, include name, designation, and org_name
        5. Sort results appropriately (e.g., by experience, salary, or name)
        6. Use proper column aliases for readability
        
        User Query: {user_query}
        
        Return ONLY a JSON object with a "query" field containing the SQL:
        {{"query": "SELECT employee_name, designation, years_of_experience FROM Employee WHERE years_of_experience > 3 ORDER BY years_of_experience DESC;"}}
        """
        return self.llm.invoke(prompt)

    def execute_query(self, res):
        try:
            start_index = res.find("{")
            while start_index != -1:
                try:
                    json_str = res[start_index:res.find("}", start_index) + 1]
                    response_dict = json.loads(json_str)
                    if "query" in response_dict:
                        sql_query = response_dict["query"].strip()
                        if sql_query:
                            results = self.db_manager.execute_query(sql_query)
                            if isinstance(results, list) and results:
                                # Format results for better readability
                                return self._format_results(results)
                            return results
                except:
                    pass
                start_index = res.find("{", start_index + 1)
            
            raise ValueError("No valid SQL query found")
        except Exception as e:
            return f"Query execution error: {str(e)}"
    
    def _format_results(self, results):
        """Format SQL results for better readability."""
        if not results:
            return "No results found"
        
        # If results is already a string, return as is
        if isinstance(results, str):
            return results
            
        # Create a formatted string for the results
        output = []
        for row in results:
            row_items = []
            for key, value in row.items():
                if value is not None:  # Only include non-null values
                    row_items.append(f"{key}: {value}")
            output.append(", ".join(row_items))
        
        return "\n".join(output)


class RAGAgent:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        self.config = Config.load()

    def query(self, question):
        try:
            # Get relevant documents
            docs = self.retriever.get_relevant_documents(question)
            context = "\n".join([doc.page_content for doc in docs])
            
            # Create prompt with context and ensure proper response formatting
            prompt = f"""
            You are a helpful assistant for Company XYZ. Use the following context to answer the question.
            If the answer cannot be found in the context, say "I don't have enough information about that in the project documents."
            
            Format your response to include:
            1. Relevant project details found
            2. Technologies and methods used
            3. Implementation details if available
            4. Performance metrics if mentioned
            
            Context:
            {context}
            
            Question: {question}
            """
            
            # Get response from LLM with proper error handling
            try:
                response = self.llm.invoke(prompt)
                if not response or response.isspace():
                    return "No relevant information found in the project documents."
                return response.strip()
            except Exception as e:
                return f"Error processing response: {str(e)}"
            
        except Exception as e:
            return f"RAG query error: {str(e)}"