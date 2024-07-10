import streamlit as st
import requests
import time

st.title("Phi-3 Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

if len(st.session_state.messages) > 4:
    st.session_state.messages = st.session_state.messages[-4:]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def response_generator(response_stream):
    buffer = ""
    for chunk in response_stream.iter_content(chunk_size=512):
        if chunk:
            buffer += chunk.decode("utf-8", errors="ignore")
            while " " in buffer:
                word, buffer = buffer.split(" ", 1)
                yield word + " "
    if buffer:
        yield buffer

if prompt := st.chat_input("Type here..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    payload = {
        "messages": st.session_state.messages,
        "model": "cuda/cuda-int4-rtn-block-32",
        "max_length": 4096,
        "do_sample": False,
    }

    response = requests.post("http://localhost:8000/generate/", json=payload, stream=True)
    
    if response.status_code == 200:
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_text = ""
            for word in response_generator(response):
                response_text += word
                response_placeholder.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
    else:
        st.error(f"Error: {response.text}")

# streamlit run phi3_chatbot.py
