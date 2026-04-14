import streamlit as st
import requests

BASE_URL = "http://127.0.0.1:8000"

st.title("📚 RAG Assistant")


st.header("Ask a question")

question = st.text_input("Type your question")

if st.button("Ask"):
    if question:
        response = requests.post(
            f"{BASE_URL}/ask",
            json={"query": question}
        )

        data = response.json()

        st.subheader("Answer")
        st.write(data["answer"])

        st.subheader("Retrieved chunks")
        for i, chunk in enumerate(data["retrieved_chunks"]):
            st.markdown(f"**Chunk {i+1}:**")
            st.write(chunk["text"])
