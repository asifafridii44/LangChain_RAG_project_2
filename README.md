# LangChain_RAG_project_2
LangChain RAG Implementation with Google GenAI and Pinecone This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline using LangChain, Pinecone, and Google Generative AI models. It includes embedding generation, vector storage, and a seamless integration to handle and retrieve contextual responses.

# LangChain RAG Implementation with Google GenAI and Pinecone

## Overview
This repository demonstrates a practical implementation of a Retrieval-Augmented Generation (RAG) pipeline using LangChain, Pinecone, and Google Generative AI models. It combines embedding generation, vector storage, and an LLM to retrieve and generate responses based on contextual information. 

The project is implemented in a Jupyter Notebook on Google Colab, making it easy to set up and run.

## Features
- **Embedding Model**: Uses Google Generative AI embeddings to encode data efficiently.
- **Vector Store**: Stores and retrieves vectorized data using Pinecone.
- **Retrieval-Augmented Generation**: Combines retrieval and generation capabilities to produce context-aware responses.
- **PDF Loading**: Demonstrates loading and processing PDF documents as context for the pipeline.
- **LLM Integration**: Uses Google's Gemini model to generate human-like responses.

## Installation
Ensure the following dependencies are installed:

```bash
pip install -qU langchain-pinecone langchain-google-genai langchain requests pypdf langchain-community docling-core python-dotenv
```

## Usage

### 1. Setting Up Environment Variables
Define the following environment variables in your system or `.env` file:
- `GOOGLE_API_KEY`: API key for Google Generative AI.
- `PINECONE_API_KEY`: API key for Pinecone.

### 2. Code Walkthrough

#### a. Setting up Embedding and Vector Store
```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

# Set up Google Generative AI embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Set up Pinecone vector store
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "rag-1"
index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)
```

#### b. Loading PDF Files as Context
```python
from langchain.document_loaders import PyPDFLoader
import requests

url = "https://raw.githubusercontent.com/[user]/[repo]/main/sample.pdf"
filename = "sample.pdf"

# Download PDF
response = requests.get(url)
with open(filename, "wb") as f:
    f.write(response.content)

# Load document
loader = PyPDFLoader(filename)
documents = loader.load()
```

#### c. Building the RAG Chain
```python
from langchain_core.documents import Document as LCDocument
from docling.document_converter import DocumentConverter
from langchain.chains import ChatGoogleGenerativeAI

retriever = vector_store.as_retriever()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

template = """Answer this user query: {question}
Here's some information that might be helpful: {context}"""
prompt_template = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

query = "Tell me about surname"
response = rag_chain.invoke(query)
print(response)
```

## Results
- The RAG pipeline retrieves relevant context from the vector store and uses the Gemini model to generate responses.
- Contextual queries are enhanced by combining document embeddings with generative AI capabilities.
