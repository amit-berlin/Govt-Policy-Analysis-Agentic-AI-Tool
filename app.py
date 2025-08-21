import streamlit as st
import pandas as pd
import numpy as np
import torch
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import CharacterTextSplitter
from fpdf import FPDF
import os

# -------------------------------
# 1. Streamlit UI
# -------------------------------
st.title("Government Policy Analysis AI Agent")
st.write("Minimal MVP: AI/GenAI + RAG + PDF Report")

uploaded_file = st.file_uploader("Upload a Government Policy PDF", type="pdf")

# -------------------------------
# 2. PDF Handling
# -------------------------------
def extract_text_from_pdf(file):
    from PyPDF2 import PdfReader
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + " "
    return text

# -------------------------------
# 3. RAG-like Retrieval
# -------------------------------
def build_vector_index(text):
    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    docs = splitter.split_text(text)
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(docs)
    return docs, embeddings, vectorizer

def retrieve_answer(query, docs, embeddings, vectorizer):
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, embeddings).flatten()
    idx = sims.argmax()
    return docs[idx]

# -------------------------------
# 4. Simple AI Models (Torch + TF)
# -------------------------------
def simple_pytorch_model(x):
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32)).numpy()

def simple_tf_model(x):
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(2,))])
    return model.predict(x, verbose=0)

# -------------------------------
# 5. PDF Report Generation
# -------------------------------
def generate_report(query, answer):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "Government Policy Analysis Report", ln=True, align="C")
    pdf.multi_cell(0, 10, f"Query: {query}\n\nAnswer:\n{answer}")
    report_path = "policy_report.pdf"
    pdf.output(report_path)
    return report_path

# -------------------------------
# 6. Main Workflow
# -------------------------------
if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)
    docs, embeddings, vectorizer = build_vector_index(text)

    st.success("PDF processed. Ask your policy-related query.")
    query = st.text_input("Enter your question:")

    if st.button("Analyze"):
        answer = retrieve_answer(query, docs, embeddings, vectorizer)

        # Demonstrate ML frameworks
        torch_out = simple_pytorch_model([[1, 2]])
        tf_out = simple_tf_model(np.array([[1.0, 2.0]]))

        st.write("### AI Answer")
        st.write(answer)
        st.write("### Torch Model Output:", torch_out)
        st.write("### TF Model Output:", tf_out)

        report_path = generate_report(query, answer)
        with open(report_path, "rb") as f:
            st.download_button("Download PDF Report", f, file_name="policy_report.pdf")

        st.success("MVP Showcase Complete ✅")
