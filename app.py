import streamlit as st
from utils.ingest import load_file
from utils.embed import embed_and_store, load_vectorstore
from utils.query_engine import generate_notes, generate_quiz

st.set_page_config(page_title="UTA AI", layout="wide")
st.title("UTA AI Assistant for Faculty")

uploaded_file = st.file_uploader("Upload your study material (PDF, DOCX, PPTX)", type=["pdf", "docx", "pptx"])

if uploaded_file:
    with st.spinner("Extracting content..."):
        text_chunks = load_file(uploaded_file)

    with st.spinner("Embedding and indexing..."):
        embed_and_store(text_chunks)

    st.success("File processed successfully. Choose an action below:")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Generate Notes"):
            with st.spinner("Generating notes..."):
                notes = generate_notes()
                st.subheader("Generated Notes")
                st.write(notes)

    with col2:
        if st.button("Create Quiz"):
            with st.spinner("Generating quiz..."):
                quiz = generate_quiz()
                st.subheader("Quiz Questions")
                st.write(quiz)
