import openai
import streamlit as st
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# Set up OpenAI API key for GPT-4
openai.api_key = 'sk-proj-wrUaZtPR4Re7jQhJj7foT3BlbkFJjb2akhjXkKdQRYYd9V6i'

# Load CLIP model and processor for image processing
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# GPT-4 Text Processing Function
def query_gpt4(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"


# CLIP Image Processing Function
def query_image(image_path, query_text):
    try:
        image = Image.open(image_path)
        inputs = clip_processor(text=[query_text], images=image, return_tensors="pt", padding=True)
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probabilities = logits_per_image.softmax(dim=1).cpu().detach().numpy()
        return probabilities
    except Exception as e:
        return f"Error: {str(e)}"


# Pandas Table Processing Function
def query_table(data, question):
    try:
        df = pd.DataFrame(data)
        result = df.query(question)
        return result
    except Exception as e:
        return f"Error: {str(e)}"


# Streamlit Application
def main():
    st.title("Multimodal Chatbot (GPT-4 Style)")

    # Section 1: Text Query
    st.header("Query Text")
    text_query = st.text_input("Enter your text query:")
    if text_query:
        response = query_gpt4(text_query)
        st.write("Response:", response)

    # Section 2: Image Query
    st.header("Query Image")
    image_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    image_query = st.text_input("Describe the image:")
    if image_file and image_query:
        st.image(image_file, caption="Uploaded Image", use_column_width=True)
        image_response = query_image(image_file, image_query)
        st.write("Image query response:", image_response)

    # Section 3: Table Query
    st.header("Query Table")
    table_data = {
        'name': ['John', 'Jane', 'Doe', 'Alice', 'Bob'],
        'age': [28, 34, 45, 29, 32],
        'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
    }
    st.write(pd.DataFrame(table_data))
    table_query = st.text_input("Enter table query (e.g., 'age > 30'):")
    if table_query:
        table_response = query_table(table_data, table_query)
        st.write("Table query response:", table_response)


if __name__ == "__main__":
    main()
