import streamlit as st
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.messages import AIMessage
import os
import time


os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = st.secrets.get("LANGSMITH_TRACING_V2", "true")
os.environ["LANGCHAIN_PROJECT"] = st.secrets.get("LANGSMITH_PROJECT", "default")

st.title("ChatGPT Clone with Groq + LangChain 👌")

# Initialize the Groq LLM
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.0,
    max_retries=2,
    api_key=st.secrets["GROQ_API_KEY"],
)

# Set up session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input from user
if prompt := st.chat_input("What's up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Convert message format for LangChain
    lc_messages = [
        SystemMessage(content="You are a helpful assistant."),
    ] + [
        (
            HumanMessage(content=m["content"])
            if m["role"] == "user"
            else AIMessage(content=m["content"])
        )
        for m in st.session_state.messages
    ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Stream response
            full_response = ""

            def response_generator():
                for chunk in llm.stream(lc_messages):
                    text = chunk.content if hasattr(chunk, "content") else chunk.text()
                    time.sleep(0.02)
                    yield text

            full_response = st.write_stream(response_generator())

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
