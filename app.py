import streamlit as st
import numpy as np
import pandas as pd
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import CharacterTextSplitter

st.set_page_config(page_title="Policy RAG – Zero Hallucination", layout="wide")

# ---------------------------
# Utilities
# ---------------------------
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    texts = []
    for p in reader.pages:
        t = p.extract_text() or ""
        texts.append(t.replace("\x00", " "))
    return "\n".join(texts).strip()

def build_corpus_chunks(raw_text, chunk_size=800, overlap=100):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_text(raw_text)
    # Keep an id for each chunk for citations
    corpus = [c.strip().replace("\n", " ") for c in chunks if c and c.strip()]
    return corpus

def fit_vectorizer(corpus):
    # Light-weight TF-IDF over chunks
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, strip_accents="unicode")
    X = vec.fit_transform(corpus)
    return vec, X

def agent1_retrieve(query, vec, X, corpus, top_k=3):
    qv = vec.transform([query])
    sims = cosine_similarity(qv, X).flatten()
    order = np.argsort(-sims)[:top_k]
    results = [{"chunk_id": int(i), "score": float(sims[i]), "text": corpus[i]} for i in order]
    return results, qv

def split_sentences(text):
    # Tiny sentence splitter (no heavy libs)
    import re
    sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sents if s and s.strip()]

def agent2_evidence_selection(query_vec, vec, retrieved):
    # Select sentence-level evidence strictly from retrieved chunks
    evidences = []
    for r in retrieved:
        sents = split_sentences(r["text"])
        if not sents: 
            continue
        sent_vecs = vec.transform(sents)
        sims = cosine_similarity(query_vec, sent_vecs).flatten()
        best_idx = int(np.argmax(sims))
        evidences.append({
            "chunk_id": r["chunk_id"],
            "sentence": sents[best_idx],
            "similarity": float(sims[best_idx])
        })
    return evidences

def agent3_answer_writer(evidences, min_sentences=1, citation_style="[]"):
    # Compose answer STRICTLY by copying sentences from evidence (extractive -> zero hallucination)
    # Sort by similarity and keep unique sentences
    evidences = sorted(evidences, key=lambda e: -e["similarity"])
    seen = set()
    picked = []
    for e in evidences:
        s = e["sentence"]
        if s not in seen and len(s) > 0:
            seen.add(s)
            picked.append(e)
    picked = picked[: max(min_sentences, len(picked))]

    # Build answer with inline citations like [c0], [c1]
    parts = []
    for i, e in enumerate(picked):
        cite = f"[c{e['chunk_id']}]"
        parts.append(f"{e['sentence']} {cite}")
    answer_text = " ".join(parts).strip()
    return answer_text, picked

def guardrail_confidence(query_vec, vec, picked, floor=0.25, avg_floor=0.30):
    if not picked:
        return 0.0, False
    sims = np.array([p["similarity"] for p in picked])
    conf = float(np.clip(0.5 * (sims.max() + sims.mean()), 0.0, 1.0))
    safe = (sims.max() >= floor) and (sims.mean() >= avg_floor)
    return conf, safe

def format_sources_table(corpus, rows):
    data = []
    for r in rows:
        data.append({
            "chunk_id": r["chunk_id"],
            "similarity": round(r.get("score", r.get("similarity", 0.0)), 3),
            "snippet": (corpus[r["chunk_id"]][:220] + "…") if len(corpus[r["chunk_id"]]) > 220 else corpus[r["chunk_id"]]
        })
    return pd.DataFrame(data)

# ---------------------------
# UI – Sidebar
# ---------------------------
st.sidebar.title("Zero-Hallucination RAG")
st.sidebar.caption("All answers are extractive from the uploaded policy PDF. If evidence is insufficient, the bot will say it doesn’t know.")
top_k = st.sidebar.slider("Retriever: top_k", 1, 5, 3)
floor = st.sidebar.slider("Guardrail: min best similarity", 0.0, 1.0, 0.25, 0.01)
avg_floor = st.sidebar.slider("Guardrail: min avg similarity", 0.0, 1.0, 0.30, 0.01)
chunk_size = st.sidebar.slider("Chunk size", 300, 1200, 800, 50)
overlap = st.sidebar.slider("Chunk overlap", 0, 300, 100, 10)

# ---------------------------
# App Header
# ---------------------------
st.title("AI Policy RAG Chatbot – Agentic • Zero Hallucination")
st.write("Upload a **Government Policy PDF**. Chat with it. The response passes through multiple agents: "
         "**Agent-1 Retriever → Agent-2 Evidence Checker → Agent-3 Answer Composer → Guardrail**.")
file = st.file_uploader("Upload Policy PDF", type=["pdf"])

# Session state for index & chat
if "corpus" not in st.session_state:
    st.session_state.corpus = None
    st.session_state.vectorizer = None
    st.session_state.X = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Build index when file uploaded
if file is not None:
    raw = extract_text_from_pdf(file)
    if not raw:
        st.error("Could not extract text from the PDF. Please try another document.")
    else:
        corpus = build_corpus_chunks(raw, chunk_size=chunk_size, overlap=overlap)
        if len(corpus) == 0:
            st.error("No text chunks available after splitting.")
        else:
            vec, X = fit_vectorizer(corpus)
            st.session_state.corpus = corpus
            st.session_state.vectorizer = vec
            st.session_state.X = X
            st.success(f"Indexed {len(corpus)} chunks. Ready to chat ✅")

# Chat UI
if st.session_state.corpus is not None:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_q = st.chat_input("Ask a policy-related question")
    if user_q:
        st.session_state.messages.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        # --- Agent 1: Retriever ---
        retrieved, qv = agent1_retrieve(user_q, st.session_state.vectorizer, st.session_state.X,
                                        st.session_state.corpus, top_k=top_k)
        with st.expander("🔎 Agent-1 Retriever (top context)"):
            st.dataframe(format_sources_table(st.session_state.corpus, retrieved), use_container_width=True)

        # --- Agent 2: Evidence selection (sentence level) ---
        evidences = agent2_evidence_selection(qv, st.session_state.vectorizer, retrieved)
        with st.expander("🧪 Agent-2 Evidence Checker (best sentences)"):
            st.dataframe(format_sources_table(st.session_state.corpus, [
                {"chunk_id": e["chunk_id"], "similarity": e["similarity"], "text": e["sentence"]}
                for e in evidences
            ]), use_container_width=True)

        # --- Agent 3: Compose strictly extractive answer + citations ---
        answer_text, picked = agent3_answer_writer(evidences)

        # --- Guardrail: Confidence & safety ---
        conf, safe = guardrail_confidence(qv, st.session_state.vectorizer, picked, floor=floor, avg_floor=avg_floor)

        # Respond
        with st.chat_message("assistant"):
            if safe and answer_text:
                st.markdown(f"**Answer (extractive, 0-hallucination):**\n\n{answer_text}\n\n"
                            f"**Confidence:** {conf:.2f}")
                # Show which chunks were cited
                cited_ids = sorted({p['chunk_id'] for p in picked})
                st.caption(f"Cited chunks: {', '.join(f'c{id}' for id in cited_ids)}")
            else:
                st.warning("I can’t answer from this document with enough evidence. "
                           "Please rephrase or upload a richer policy PDF.")
                st.caption(f"Confidence: {conf:.2f}")

        # Keep in history
        final_msg = answer_text if (safe and answer_text) else "Not enough evidence in document."
        st.session_state.messages.append({"role": "assistant", "content": final_msg})

else:
    st.info("Upload a PDF to begin.")

# Footer note
st.caption("This demo enforces **extractive answers only** from the uploaded PDF to avoid hallucinations. "
           "If similarity thresholds aren’t met, it will abstain.")
