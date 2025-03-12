#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
import sqlite3
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
import json

# Load environment variables
load_dotenv()

# Create an in-memory database
connection = sqlite3.connect(":memory:")
cursor = connection.cursor()

# Create the Employee table
cursor.execute('''
CREATE TABLE Employee (
    employee_name TEXT NOT NULL,
    employee_email TEXT UNIQUE NOT NULL,
    org_name TEXT,
    designation TEXT NOT NULL,
    years_of_experience REAL NOT NULL,
    salary REAL,
    location TEXT,
    hire_date TEXT
);
''')

# Sample data to populate the Employee table
employees = [
    ("Alice Johnson", "alice.johnson@example.com", "TechCorp", "Software Engineer", 3.5, 70000, "New York", "2019-06-15"),
    ("Bob Smith", "bob.smith@example.com", "TechCorp", "Product Manager", 7.0, 95000, "San Francisco", "2016-02-10"),
    ("Charlie Brown", "charlie.brown@example.com", "InnovateLtd", "Data Scientist", 2.0, 75000, "Austin", "2020-03-01"),
    ("Diana Prince", "diana.prince@example.com", "InnovateLtd", "Senior Analyst", 5.5, 85000, "Los Angeles", "2017-08-20"),
    ("Eve Davis", "eve.davis@example.com", "CreativeSolutions", "UX Designer", 4.0, 72000, "Chicago", "2018-05-12"),
    ("Frank Miller", "frank.miller@example.com", "CreativeSolutions", "Project Manager", 6.2, 90000, "Seattle", "2015-09-10"),
    ("Grace Hopper", "grace.hopper@example.com", "TechCorp", "Lead Engineer", 10.0, 120000, "New York", "2010-11-01"),
    ("Hank Pym", "hank.pym@example.com", "InnovateLtd", "Research Scientist", 8.5, 95000, "San Francisco", "2014-02-15"),
    ("Irene Adler", "irene.adler@example.com", "CreativeSolutions", "Content Strategist", 3.8, 75000, "Chicago", "2016-06-17"),
    ("Jack Reacher", "jack.reacher@example.com", "TechCorp", "QA Engineer", 2.4, 67000, "Austin", "2020-07-22"),
    ("Karen Page", "karen.page@example.com", "InnovateLtd", "Business Analyst", 6.0, 80000, "Los Angeles", "2014-12-05"),
    ("Leo Messi", "leo.messi@example.com", "TechCorp", "DevOps Engineer", 4.5, 85000, "New York", "2017-01-25"),
    ("Mia Wallace", "mia.wallace@example.com", "CreativeSolutions", "Graphic Designer", 3.3, 65000, "Seattle", "2019-11-30"),
    ("Nick Fury", "nick.fury@example.com", "InnovateLtd", "Operations Head", 15.0, 150000, "San Francisco", "2005-03-22"),
    ("Oscar Wilde", "oscar.wilde@example.com", "TechCorp", "Technical Writer", 5.1, 78000, "Austin", "2016-08-15")
]

# Insert data into the Employee table
cursor.executemany('''
INSERT INTO Employee (employee_name, employee_email, org_name, designation, years_of_experience, salary, location, hire_date)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
''', employees)
connection.commit()

# Read Projects.txt
with open('knowledge-base/Projects.txt', 'r') as file:
    content = file.read()
sections = content.split('##########')

# Initialize embeddings and knowledge base
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
knowledge_base = FAISS.from_texts(sections, embeddings)

# Prompt for Decision Maker Agent
def classify_query(query):
    prompt = f"""
    You are a decision-making assistant for Company XYZ. Classify this query into exactly ONE of these types:

    1. General Knowledge Query: Questions about general topics, concepts, or algorithms that don't require company data
    2. Project Query: Questions specifically about company projects, technologies used, or project outcomes. Any query that:
       - Asks about "our" projects or uses phrases like "have we", "did we", "our company"
       - Requests information about specific or all company projects
       - Asks about technologies or methods used in company projects
    3. Employee Query: Questions about employee data, including:
       - Names, emails, designations
       - Salary information
       - Years of experience
       - Work locations
       - Organization structure
       - Any other employee-related information

    User Query: {query}

    Think step by step:
    1. Does this query ask about employee/HR data? If yes -> type 3
    2. Does this query contain "we", "our", or ask about company projects? If yes -> type 2
    3. Is this a general knowledge question? If yes -> type 1

    Respond ONLY in this exact JSON format:
    {{
        "query_type": "1 or 2 or 3",
        "reasoning": "brief explanation"
    }}
    """
    return prompt


import json

def decode_classification_response(res, query):
    """
    Decodes the JSON response from the classification function with improved error handling.

    Parameters:
        res (str): Response string containing query_type and reasoning.
        query (str): The original query string.

    Returns:
        dict: A dictionary with the decoded query_type and reasoning.
    """
    try:
        if not isinstance(res, str):
            raise ValueError("Response is not a string.")
        
        # Check for company-specific indicators in the original query
        query_lower = query.lower()
        company_indicators = ["we", "our", "company", "project", "projects"]
        employee_indicators = ["employee", "salary", "designation", "experience"]
        
        # Force classification based on indicators
        if any(indicator in query_lower for indicator in employee_indicators):
            return {"query_type": "3", "reasoning": "Query contains employee-related terms"}
        elif any(indicator in query_lower for indicator in company_indicators):
            return {"query_type": "2", "reasoning": "Query contains company/project-related terms"}
        
        # If no forced classification, proceed with normal JSON parsing
        start_index = res.find("{")
        while start_index != -1:
            # Find the matching closing brace
            brace_count = 1
            pos = start_index + 1
            while pos < len(res) and brace_count > 0:
                if res[pos] == '{':
                    brace_count += 1
                elif res[pos] == '}':
                    brace_count -= 1
                pos += 1
                
            if brace_count == 0:
                try:
                    json_str = res[start_index:pos]
                    parsed = json.loads(json_str)
                    if "query_type" in parsed and "reasoning" in parsed:
                        # Clean up the query_type by removing any whitespace
                        parsed["query_type"] = parsed["query_type"].strip()
                        return parsed
                except:
                    pass
            
            start_index = res.find("{", pos)
        
        # Fallback: Look for specific patterns
        if '"query_type": "1"' in res:
            return {"query_type": "1", "reasoning": "General knowledge query"}
        elif '"query_type": "2"' in res:
            return {"query_type": "2", "reasoning": "Technical/project query"}
        elif '"query_type": "3"' in res:
            return {"query_type": "3", "reasoning": "Employee details query"}
            
        raise ValueError("No valid JSON found in the response.")
    except Exception as e:
        raise ValueError(f"An error occurred: {e}")


# SQl Agent

def sql_agent_query(user_query):
    prompt = f"""
    You are an expert SQL assistant. Generate an SQLite3 query for the following request.
    
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

    User Query: {user_query}

    Instructions:
    1. Generate ONLY a simple, direct SQL query that answers the user's question
    2. Return ONLY a JSON object with a "query" field containing the SQL
    3. Do not include any explanations or comments
    4. Use only the columns that exist in the schema
    5. Ensure the query is valid SQLite3 syntax

    Output Format Example:
    {{"query": "SELECT employee_name FROM Employee WHERE salary > 50000;"}}

    Your Response (must be valid JSON):
    """
    return prompt

def sql_query_result(res, cursor):
    """
    Decodes the JSON response to extract a SQL query and executes it.
    """
    try:
        if not isinstance(res, str):
            raise ValueError("Response is not a string.")
        
        # Find the first valid JSON object in the response
        start_index = res.find("{")
        while start_index != -1:
            # Find the matching closing brace
            brace_count = 1
            pos = start_index + 1
            while pos < len(res) and brace_count > 0:
                if res[pos] == '{':
                    brace_count += 1
                elif res[pos] == '}':
                    brace_count -= 1
                pos += 1
                
            if brace_count == 0:
                try:
                    json_str = res[start_index:pos]
                    response_dict = json.loads(json_str)
                    if "query" in response_dict:
                        sql_query = response_dict["query"].strip()
                        if sql_query:
                            print(f"Executing SQL Query: {sql_query}")
                            cursor.execute(sql_query)
                            rows = cursor.fetchall()
                            
                            # Format the results nicely
                            if not rows:
                                return "No results found"
                                
                            # Get column names from cursor description
                            columns = [desc[0] for desc in cursor.description]
                            
                            # Format results as list of dictionaries
                            formatted_rows = []
                            for row in rows:
                                formatted_rows.append(dict(zip(columns, row)))
                            return formatted_rows
                            
                except json.JSONDecodeError:
                    pass
                    
            start_index = res.find("{", pos)
            
        raise ValueError("No valid SQL query found in the response")
        
    except Exception as e:
        raise ValueError(f"An error occurred: {e}")

from langchain.prompts import PromptTemplate

def RAG_agent(query, custom_llm):
    prompt_template = """
    You are a helpful assistant for Company XYZ with access to project documentation. Your goal is to provide comprehensive information about our projects.
    
    When answering about projects, ALWAYS include these details for EACH relevant project:
    1. Project name and overview
    2. Technologies and algorithms:
       - List all algorithms mentioned
       - Describe the tech stack
       - Explain why certain approaches were chosen
    3. Performance and metrics:
       - Quote specific numbers and percentages
       - List all performance metrics mentioned
    4. Implementation details:
       - Hardware/sensors used
       - System components
       - Technical challenges
    5. Future plans and next steps
    
    Context from project documents:
    {context}
    
    Please answer the following question based ONLY on the information provided in the context above.
    Structure your response to include ALL the technical details found in the context.
    If the information is not available in the context, say "I don't have enough information about that in the project documents."
    
    Question: {question}
    """
    
    # Create a prompt template instance
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )
    
    # Use the prompt in the RetrievalQA chain with more relevant documents
    qa_chain = RetrievalQA.from_chain_type(
        llm=custom_llm,
        retriever=knowledge_base.as_retriever(search_kwargs={"k": 5, "fetch_k": 8}),
        chain_type_kwargs={"prompt": prompt}
    )
    
    try:
        response = qa_chain.run(query)
        return response.strip()
    except Exception as e:
        return f"Error retrieving project information: {str(e)}"


# Initialize Ollama (updated to use OllamaLLM)
custom_llm = OllamaLLM(model="llama2", temperature=0)

# Test questions
questions = [
    "Why is global warming on the rise?",
    "Have we undertaken any projects related to robotics?",
    "Who are the employees in the company with more than three years of experience (list only their names)?",
    "How does binary search algorithm work?",
    "What are some projects on renewable energy, and what techniques are used in them?",
    "What designations exist within the company?"
]

# Process questions
for query in questions:
    print(f"\nQuery: {query}")
    classification_prompt = classify_query(query)
    res = custom_llm.invoke(classification_prompt)
    
    try:
        classification_result = decode_classification_response(res, query)
        query_type = classification_result['query_type'].strip()
        reasoning = classification_result['reasoning']
        print(f"Query type: {query_type}")
        print(f"Reasoning: {reasoning}")
        
        if query_type == "1":
            print("General LLM Intelligence used")
            answer = custom_llm.invoke(query)
        elif query_type == "2":
            print("Company's internal Project-base searched")
            answer = RAG_agent(query, custom_llm)
        elif query_type == "3":
            print("Searching in company's database")
            sql_prompt = sql_agent_query(query)
            sql_response = custom_llm.invoke(sql_prompt)
            answer = sql_query_result(sql_response, cursor)
        else:
            raise ValueError(f"Invalid query type: {query_type}")
        
        print(f"Answer: {answer}")
    except Exception as e:
        print(f"Error processing query: {e}")
        continue

# Close the connection
connection.close()

