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


st.header("Upload document")

uploaded_file = st.file_uploader("Upload PDF")

if uploaded_file is not None and st.button('Upload'):
    files = {"file": uploaded_file.getvalue()}
    response = requests.post(
        f"{BASE_URL}/upload",
        files={"file": uploaded_file}
    )
    st.success(response.json()["message"])

st.header("Rebuild index")

if st.button('Rebuild'):
    response = requests.post(f"{BASE_URL}/rebuild")

    if response.status_code == 200:
        data = response.json()
        st.success(f"{data['status']} — {data['num_chunks']} chunks created")
    else:
        st.error(response.text)
