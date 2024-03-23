import json
import os
from llama_index.core import Document
from llama_index.core.schema import MetadataMode

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
    return document.get_content(metadata_mode=MetadataMode.LLM)

# Read JSON file
with open('final_data-file.json', 'r') as file:
    data = json.load(file)

# Create a folder to save the documents
output_folder = 'llama_documents'
os.makedirs(output_folder, exist_ok=True)

# Iterate over each entry/document in JSON
for index, entry in enumerate(data):
    llama_document = create_llama_document(entry)
    # Name the file based on the document's index
    file_name = os.path.join(output_folder, f"llama_document_{index}.txt")
    # Write the Llama index document to the file
    with open(file_name, 'w') as output_file:
        output_file.write(llama_document)
