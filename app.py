import streamlit as st
import pandas as pd
import numpy as np
import torch
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import CharacterTextSplitter
from fpdf import FPDF
from PyPDF2 import PdfReader

# -------------------------------
# 1. Streamlit UI
# -------------------------------
st.title("AI Policy Intelligence Agent")
st.write("Unified MVP: Policy Analysis • Citizen Feedback • Monitoring • Assistant • Budget Optimization")

uploaded_file = st.file_uploader("Upload Government Policy PDF", type="pdf")

# -------------------------------
# 2. Extract PDF Text
# -------------------------------
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])

# -------------------------------
# 3. RAG-style Policy QA
# -------------------------------
def build_index(text):
    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    docs = splitter.split_text(text)
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(docs)
    return docs, embeddings, vectorizer

def retrieve_answer(query, docs, embeddings, vectorizer):
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, embeddings).flatten()
    return docs[np.argmax(sims)]

# -------------------------------
# 4. Simulated Citizen Feedback Analysis
# -------------------------------
def citizen_feedback_demo():
    feedback = pd.DataFrame({
        "feedback": ["Hospitals lack doctors", "Schools need internet", "Roads improved", "Subsidy delayed"],
        "sentiment": [ -0.8, -0.6, 0.7, -0.5]
    })
    avg_sent = feedback["sentiment"].mean()
    return feedback, avg_sent

# -------------------------------
# 5. Simple ML Models (Torch + TF)
# -------------------------------
def torch_demo(x):
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32)).numpy()

def tf_demo(x):
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(2,))])
    return model.predict(x, verbose=0)

# -------------------------------
# 6. PDF Report Generation
# -------------------------------
def generate_report(query, answer, avg_sent):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "AI Policy Intelligence Report", ln=True, align="C")
    pdf.multi_cell(0, 10, f"Query: {query}\n\nAnswer: {answer}\n\nCitizen Sentiment Score: {avg_sent:.2f}")
    path = "policy_report.pdf"
    pdf.output(path)
    return path

# -------------------------------
# 7. Main Workflow
# -------------------------------
if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)
    docs, embeddings, vectorizer = build_index(text)

    st.success("Policy Document Processed ✅")
    query = st.text_input("Ask a policy-related question:")

    if st.button("Run AI Analysis"):
        answer = retrieve_answer(query, docs, embeddings, vectorizer)
        feedback, avg_sent = citizen_feedback_demo()
        torch_out = torch_demo([[1,2]])
        tf_out = tf_demo(np.array([[1.0,2.0]]))

        # Show results
        st.subheader("AI Policy Answer")
        st.write(answer)
        st.subheader("Citizen Feedback Analysis")
        st.write(feedback)
        st.write("Avg Sentiment:", avg_sent)
        st.subheader("Torch Model Output")
        st.write(torch_out)
        st.subheader("TF Model Output")
        st.write(tf_out)

        # Report
        report = generate_report(query, answer, avg_sent)
        with open(report, "rb") as f:
            st.download_button("Download PDF Report", f, file_name="policy_report.pdf")

        st.success("End-to-End AI Governance MVP Complete 🎉")
