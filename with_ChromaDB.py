import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import time

# Load environment variables from .env file
load_dotenv()

# Function to create a conversational chain with GPT
def get_conversational_chain():
    api_key = os.getenv("GPT_API_KEY")
    if not api_key:
        st.error("API key not found. Please check your .env file.")
        return None

    prompt_template = """
    You are a knowledgeable assistant who provides accurate and detailed answers based on the provided context. 
    If the answer is not explicitly available in the context, say, "The answer is not available in the context." 
    Do not generate misleading or incorrect information.

    Context:\n{context}\n
    Question:\n{question}\n
    
    Answer:
    """
    
    model = ChatOpenAI(api_key=api_key, model="gpt-4", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"], output_variable="answer")
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to clean response text
def clean_response(response_text):
    cleaned_text = response_text.strip()
    if cleaned_text.lower().startswith("the answer is not available in the context"):
        return cleaned_text
    return cleaned_text

# Safe request function with retry logic for rate limits
def safe_request(api_func, *args, **kwargs):
    while True:
        try:
            return api_func(*args, **kwargs)
        except Exception as e:
            if 'Rate limit exceeded' in str(e):
                print(f"Rate limit exceeded: {e}")
                time.sleep(60)  # Wait for 1 minute before retrying
            else:
                raise

# Function to handle user input and generate responses
def user_input(user_question):
    api_key = os.getenv("GPT_API_KEY")
    if not api_key:
        st.error("API key not found. Please check your .env file.")
        return

    embeddings = OpenAIEmbeddings(api_key=api_key)
    
    # Load the ChromaDB vector store from the specified directory
    persist_directory = "chroma_db"
    if not os.path.exists(persist_directory):
        st.error("ChromaDB directory not found. Please create the ChromaDB first.")
        return

    new_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
    # Limit the number of documents retrieved
    docs = new_db.similarity_search(user_question, k=5)

    chain = get_conversational_chain()

    if chain:
        response = safe_request(
            chain,
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )

        cleaned_response = clean_response(response["output_text"])
        st.write("Reply: ", cleaned_response)

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="CheemsHUB")
    st.header("Chat with PDF using GPT-4")

    st.write("The PDF uploading option has been removed. Please ensure that the vector store (chroma_db) is already created.")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
