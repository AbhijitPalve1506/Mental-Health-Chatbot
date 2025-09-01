import streamlit as st
import requests

# FastAPI backend URL
API_URL = "https://mental-health-chatbot-backend-hi3p.onrender.com/ask"

st.set_page_config(page_title="Mental Health Chatbot", page_icon="💬")

st.title("💬 Mental Health Chatbot")
st.write("This chatbot uses **You Become What You Think & Mental Health Care** context to provide supportive responses.")

# Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# User input
if user_input := st.chat_input("How are you feeling today?"):
    # Show user message
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Send to backend
    try:
        response = requests.post(API_URL, json={"query": user_input})
        answer = response.json().get("answer", "Sorry, I couldn't process that.")
    except Exception as e:
        answer = f"⚠️ Error: {e}"

    # Show assistant message
    st.chat_message("assistant").write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})