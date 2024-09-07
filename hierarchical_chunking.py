from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set the path for the PDF and the directory for ChromaDB
pdf_path = "python-basics-sample-chapters-9-20.pdf"
persist_directory = "chroma_db"

# Load the PDF and split it into optimized chunks
def load_and_split_pdf(pdf_path):
    print(f"Loading PDF from: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # First, split into larger sections
    initial_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    large_chunks = initial_splitter.split_documents(documents)
    
    # Now, apply more granular splitting for contextual relevance
    refined_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    refined_chunks = []
    for chunk in large_chunks:
        refined_chunks.extend(refined_splitter.split_documents([chunk]))

    print(f"Total number of chunks created: {len(refined_chunks)}")
    return refined_chunks

# Create the ChromaDB vector store
def create_chroma_db(persist_directory, documents):
    print(f"Creating ChromaDB in: {persist_directory}")
    api_key = os.getenv("GPT_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please check your .env file.")
    
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    
    # Create the ChromaDB vector store
    vectordb = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)
    vectordb.persist()  # Persist the database

    if os.path.exists(persist_directory):
        print(f"ChromaDB successfully created in: {persist_directory}")
    else:
        print("Failed to create ChromaDB directory.")

if __name__ == "__main__":
    documents = load_and_split_pdf(pdf_path)
    create_chroma_db(persist_directory, documents)
    print("Process completed.")
