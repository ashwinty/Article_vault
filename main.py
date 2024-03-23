import logging
import sys
import pandas as pd
import faiss
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from llama_index.core import SimpleDirectoryReader, load_index_from_storage, VectorStoreIndex, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore

# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Load data for vector searching
documents = SimpleDirectoryReader(input_files=["final_data-file.json"]).load_data()
os.environ['OPENAI_API_KEY'] = " "

# Load vector store
vector_store = FaissVectorStore.from_persist_dir(persist_dir="./storage")
storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir="./storage")
index = load_index_from_storage(storage_context=storage_context)
query_engine = index.as_query_engine()

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Function to calculate sentence embeddings
def calculate_sentence_embeddings(sentences):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def extract_keywords(query):
    keywords = query.lower().split()
    return keywords

def highlight_matching_words(query, text):
    highlighted_text = text.replace(query, f"<span style='background-color: yellow; color: black'>{query}</span>")
    return highlighted_text


def highlight_matching_sentence(query, response):
    query_embedding = calculate_sentence_embeddings([query])
    response_sentences = response.split('.')  # Split response into sentences
    max_similarity_score = 0
    highlighted_response = ""
    for sentence in response_sentences:
        sentence_embedding = calculate_sentence_embeddings([sentence])
        similarity_score = torch.matmul(query_embedding, sentence_embedding.T)
        if similarity_score > max_similarity_score:
            max_similarity_score = similarity_score
            highlighted_sentence = f"<span style='background-color: yellow; color: black'>{sentence}</span>"
        else:
            highlighted_sentence = sentence
        highlighted_response += highlighted_sentence + ". "
    
    return highlighted_response


def to_run(query):
    keywords = extract_keywords(query)
    query_string = ' '.join(keywords)
    response = query_engine.query(query_string)
    formatted_responses = [] 
    for node in response.__dict__["source_nodes"]:
        metadata = node.metadata
        text = node.text
        highlighted_response = highlight_matching_sentence(query, text)  # Highlight sentences
        formatted_response = ([metadata["Title"], metadata["Tag"], metadata["Author"], metadata["Date"], metadata["Article URL"], metadata["Description"], metadata["Main Image URL"]], highlighted_response)
        formatted_responses.append(formatted_response)
    return formatted_responses


# def to_run(query, min_results=3):
#     adjusted_query = adjust_query(query)  # Adjust the query to be more inclusive
#     keywords = extract_keywords(adjusted_query)
#     query_string = ' '.join(keywords)
#     response = query_engine.query(query_string)
#     print(f"Total number of source nodes: {len(response.__dict__['source_nodes'])}")
#     formatted_responses = [] 
#     for i, node in enumerate(response.__dict__["source_nodes"]):
#         if i >= min_results:  # Limit the number of results to min_results
#             break
#         metadata = node.metadata
#         text = node.text
#         highlighted_response = highlight_matching_sentence(query, text)  # Highlight sentences
#         formatted_response = ([metadata["Title"], metadata["Tag"], metadata["Author"], metadata["Date"], metadata["Article URL"], metadata["Description"], metadata["Main Image URL"]], highlighted_response)
#         formatted_responses.append(formatted_response)
#         print(f"Processing result {i+1}: {metadata['Title']}")
#     if len(formatted_responses) < min_results:
#         print("Insufficient results, retrying without min_results constraint...")
#         # If we didn't retrieve enough results, try again without the min_results constraint
#         remaining_results = min_results - len(formatted_responses)
#         additional_responses = []
#         for i, node in enumerate(response.__dict__["source_nodes"][len(formatted_responses):]):
#             if i >= remaining_results:
#                 break
#             metadata = node.metadata
#             text = node.text
#             highlighted_response = highlight_matching_sentence(query, text)  # Highlight sentences
#             formatted_response = ([metadata["Title"], metadata["Tag"], metadata["Author"], metadata["Date"], metadata["Article URL"], metadata["Description"], metadata["Main Image URL"]], highlighted_response)
#             additional_responses.append(formatted_response)
#             print(f"Processing additional result {i+1}: {metadata['Title']}")
#         print(f"Number of additional responses: {len(additional_responses)}")
#         formatted_responses.extend(additional_responses)
#         print(f"Total number of formatted responses: {len(formatted_responses)}")
#     return formatted_responses


# # Example usage:
# query = input("Please enter your query: ")  # Get query from user
# response = to_run(query)
# highlighted_response = highlight_matching_sentences(query, response)
# print(highlighted_response)









# import logging
# import sys
# import os
# import torch
# from transformers import AutoTokenizer, AutoModel
# from llama_index.core import SimpleDirectoryReader, load_index_from_storage, StorageContext
# from llama_index.vector_stores.faiss import FaissVectorStore
# from langchain_openai import ChatOpenAI
# from langchain.agents import tool
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
# from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
# from langchain.agents import AgentExecutor
# from typing import List

# # Set up logging
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load data for vector searching
# try:
#     documents = SimpleDirectoryReader(input_files=["final_data-file.json"]).load_data()
# except Exception as e:
#     logger.error(f"Error loading data for vector searching: {e}")
#     sys.exit(1)

# # Load API key from environment variables
# os.environ["OPENAI_API_KEY"] = " "

# # Load vector store
# try:
#     vector_store = FaissVectorStore.from_persist_dir(persist_dir="./storage")
#     storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir="./storage")
#     index = load_index_from_storage(storage_context=storage_context)
#     query_engine = index.as_query_engine()
# except Exception as e:
#     logger.error(f"Error loading vector store: {e}")
#     sys.exit(1)

# # Load model from HuggingFace Hub
# try:
#     tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
#     model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
# except Exception as e:
#     logger.error(f"Error loading model: {e}")
#     sys.exit(1)

# # Function to calculate sentence embeddings
# def calculate_sentence_embeddings(sentences):
#     encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
#     with torch.no_grad():
#         model_output = model(**encoded_input)
#     token_embeddings = model_output.last_hidden_state
#     input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
#     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# # @tool


# # @tool
# # def semantic_search(query: str) -> List[str]:
# #     """
# #     Perform semantic search using the provided query.

# #     Args:
# #         query (str): The query string for semantic search.

# #     Returns:
# #         List[str]: List of search results.
# #     """
# #     try:
# #         response = query_engine.query(query)
# #         print(response)
# #         if response is not None and hasattr(response, 'json'):
# #             search_results = response.json().get("data", {}).get("results", [])
# #             # print(search_results)
# #             if search_results:
# #                 return search_results
# #             else:
# #                 logger.warning("Semantic search returned an empty result.")
# #                 return []
# #         else:
# #             logger.error("Semantic search request failed or returned None.")
# #             return []
# #     except Exception as e:
# #         logger.error(f"Error performing semantic search: {e}")
# #         return []



# # Modify the agent prompt to include semantic search results
# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a powerful assistant with semantic search capabilities.",
#         ),
#         ("user", "{input}"),
#         MessagesPlaceholder(variable_name="semantic_search_results"),
#         MessagesPlaceholder(variable_name="agent_scratchpad"),
#     ]
# )

# # Bind semantic search tool to the language model
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# llm_with_tools = llm.bind_tools([semantic_search])

# # Update the agent to include semantic search results
# agent = (
#     {
#         "input": lambda x: x["input"],
#         "agent_scratchpad": lambda x: format_to_openai_tool_messages(
#             x["intermediate_steps"]
#         ),
#         "semantic_search_results": lambda x: x["semantic_search_results"],
#     }
#     | prompt
#     | llm_with_tools
#     | OpenAIToolsAgentOutputParser()
# )

# # Define how the agent handles responses
# def handle_response(user_input, search_results, agent_output):
#     if isinstance(search_results, list):
#         for i, result in enumerate(search_results, 1):
#             print(f"Search Result {i}: {result}")
#     else:
#         print("Search results are not in the expected format.")

#     return agent_output

# # Initialize AgentExecutor
# agent_executor = AgentExecutor(agent=agent, tools=[semantic_search], verbose=True)

# # If running as standalone script
# if __name__ == "__main__":
#     try:
#         user_input = "User query"
#         response = agent_executor.invoke({"input": user_input})
#         print(response)
#     except Exception as e:
#         logger.error(f"Error executing agent: {e}")













# import logging
# import sys
# import os
# import torch
# from transformers import AutoTokenizer, AutoModel
# from llama_index.core import SimpleDirectoryReader, load_index_from_storage, StorageContext
# from llama_index.vector_stores.faiss import FaissVectorStore
# from langchain_openai import ChatOpenAI
# from langchain.agents import tool
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
# from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
# from langchain.agents import AgentExecutor
# from typing import List

# # Set up logging
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# # Load data for vector searching
# documents = SimpleDirectoryReader(input_files=["final_data-file.json"]).load_data()

# # Load API key from environment variables
# os.environ["OPENAI_API_KEY"] = "  "

# # Load vector store
# try:
#     vector_store = FaissVectorStore.from_persist_dir(persist_dir="./storage")
#     storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir="./storage")
#     index = load_index_from_storage(storage_context=storage_context)
#     query_engine = index.as_query_engine()
# except Exception as e:
#     logging.error(f"Error loading vector store: {e}")
#     sys.exit(1)

# # Load model from HuggingFace Hub
# try:
#     tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
#     model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
# except Exception as e:
#     logging.error(f"Error loading model: {e}")
#     sys.exit(1)

# # Function to calculate sentence embeddings
# def calculate_sentence_embeddings(sentences):
#     encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
#     with torch.no_grad():
#         model_output = model(**encoded_input)
#     token_embeddings = model_output.last_hidden_state
#     input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
#     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# @tool
# def semantic_search(query: str) -> List[str]:
#     """
#     Perform semantic search using the provided query.

#     Args:
#         query (str): The query string for semantic search.

#     Returns:
#         List[str]: List of search results.
#     """
#     # Perform semantic search using your existing model
#     response = query_engine.query(query)
    
#     # Check if the response is successful and in JSON format
#     if response is not None and hasattr(response, 'json'):
#         try:
#             # Try to parse the response as JSON
#             search_results = response.json().get("data", {}).get("results", [])
#             return search_results
#         except Exception as e:
#             logging.error(f"Failed to parse response as JSON: {e}")
#             return []
#     else:
#         logging.error("Semantic search request failed or returned None.")
#         return []


    
#     # Extract search results from the response object
#     if response.status_code == 200:
#         search_results = response.json().get("data", {}).get("results", [])
#         return search_results
#     else:
#         logging.error(f"Semantic search request failed with status code: {response.status_code}")
#         return []

# # Modify the agent prompt to include semantic search results
# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a powerful assistant with semantic search capabilities.",
#         ),
#         ("user", "{input}"),
#         MessagesPlaceholder(variable_name="semantic_search_results"),
#         MessagesPlaceholder(variable_name="agent_scratchpad"),
#     ]
# )

# # Bind semantic search tool to the language model
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# llm_with_tools = llm.bind_tools([semantic_search])

# # Update the agent to include semantic search results
# agent = (
#     {
#         "input": lambda x: x["input"],
#         "agent_scratchpad": lambda x: format_to_openai_tool_messages(
#             x["intermediate_steps"]
#         ),
#         "semantic_search_results": lambda x: x["semantic_search_results"],
#     }
#     | prompt
#     | llm_with_tools
#     | OpenAIToolsAgentOutputParser()
# )

# # Define how the agent handles responses
# def handle_response(user_input, search_results, agent_output):
#     # Ensure search_results is iterable and contains the expected data
#     if isinstance(search_results, list):
#         # Iterate over search_results
#         for i, result in enumerate(search_results, 1):
#             # Your logic for handling each search result
#             print(f"Search Result {i}: {result}")
#     else:
#         # Handle the case where search_results is not a list
#         print("Search results are not in the expected format.")

#     # Your logic for generating responses based on user input and search results
#     return agent_output

# # Initialize AgentExecutor
# agent_executor = AgentExecutor(agent=agent, tools=[semantic_search], verbose=True)

# # If running as standalone script
# if __name__ == "__main__":
#     # Example usage
#     user_input = "User query"
#     response = agent_executor.invoke({"input": user_input})
#     print(response)