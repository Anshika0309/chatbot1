import os
import sys

import openai
from langchain_community.document_loaders import CSVLoader
#from langchain_community.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.chains import RetrievalQA

import constants
openai.api_key= constants.APIKEY

# Initialize CSVLoader
loader = CSVLoader('C:\\AK\\chatbot\\Nifty Historical - NSE-NIFTY.csv')

# Initialize VectorstoreIndexCreator
#index_creator = VectorstoreIndexCreator()
#openai_client = openai.API(os.getenv("OPENAI_API_KEY"))
# Create index from loaders
#index = index_creator.from_loaders([loader])

# Wrap index using VectorStoreIndexWrapper
#index_wrapper = VectorStoreIndexWrapper(loader)

# Initialize RetrievalQA with appropriate parameters
chain = RetrievalQA(
    llm=openai,  # This should be the OpenAI client instance, not the module
    retriever={"type": "single_document", "document": loader},  # Pass the retriever as a dictionary
    input_key="question", # Specify the input key
    combine_documents_chain = None
)
# Define query
query = "Do you have a column called High?"

# Get response from the chain
response = chain({"question": query})

# Print response
print(response)
