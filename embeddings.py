import logging
import sys
import faiss
import os
import json
from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore

os.environ['OPENAI_API_KEY'] = " "

# Function to create Llama index document from JSON entry
def create_llama_document(entry):
    metadata = {
        "Title": entry['Title'],
        "Tag": entry['Tag'],
        "Author": entry['Author'],
        "Date": entry['Date'],
        "Article URL": entry['Article URL'],
        "Description": entry['Description'],
        "Main Image URL": entry['Main Image URL']
    }
    document = Document(
        text=entry['Article Text'],
        metadata=metadata,
        # excluded_llm_metadata_keys=["Main Image URL"],  # Exclude 'Main Image URL' from LLM metadata
        metadata_separator="::",
        metadata_template="{key}: {value}",
        text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
    )
    return document

# Load JSON file
with open('final_data-file.json', 'r') as file:
    data = json.load(file)

# Create Llama index documents from JSON data
documents = [create_llama_document(entry) for entry in data]

# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Set Faiss index parameters
embedding_dim = 1536  # Update with your actual embedding dimension
faiss_index = faiss.IndexFlatL2(embedding_dim)

# Initialize Faiss vector store
vector_store = FaissVectorStore(faiss_index=faiss_index)

# Create storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create vector store index from documents
index = VectorStoreIndex.from_documents(
    documents, 
    storage_context=storage_context,
    show_progress=True,
)

# Persist index to disk
index.storage_context.persist()