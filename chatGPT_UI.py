import streamlit as st

# Page configuration
st.set_page_config(page_title="ChatGPT Clone", page_icon="🤖", layout="centered")

# Custom CSS to style the app to resemble ChatGPT
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stTextInput > div > div > input {
        font-size: 18px;
    }
    .stTextArea textarea {
        font-size: 18px;
    }
    .stButton button {
        font-size: 18px;
        width: 100%;
        background-color: #008CBA;
        color: white;
    }
    .message-container {
        background-color: #fff;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
        border: 1px solid #ddd;
        max-width: 700px;
        width: 100%;
    }
    .user-message {
        text-align: right;
        color: #333;
    }
    .assistant-message {
        text-align: left;
        color: #444;
    }
    </style>
""", unsafe_allow_html=True)

# App Title
st.title("ChatGPT Clone 🤖")

# Initialize chat history
if 'history' not in st.session_state:
    st.session_state.history = []

# Display chat history
for message in st.session_state.history:
    if message['role'] == 'user':
        st.markdown(f'<div class="message-container user-message">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="message-container assistant-message">{message["content"]}</div>', unsafe_allow_html=True)

# User input
user_input = st.text_input("Type your message here...")

# Send button action
if st.button("Send"):
    if user_input:
        # Add user message to chat history
        st.session_state.history.append({"role": "user", "content": user_input})
        
        # Placeholder response for bot (functionality will be added later)
        st.session_state.history.append({"role": "assistant", "content": "This is a placeholder response."})
        
        # Rerun to update the chat interface
        st.experimental_rerun()

