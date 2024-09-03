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

# Load the PDF and split it into chunks
def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    documents = loader.load_and_split(text_splitter=text_splitter)
    return documents

# Create the ChromaDB vector store
def create_chroma_db(persist_directory, documents):
    api_key = os.getenv("GPT_API_KEY")
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    
    vectordb = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)
    vectordb.persist()  # Persist the database

if __name__ == "__main__":
    documents = load_and_split_pdf(pdf_path)
    create_chroma_db(persist_directory, documents)
    print("ChromaDB created and stored successfully.")
