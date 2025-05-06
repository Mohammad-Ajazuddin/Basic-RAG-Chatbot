import streamlit as st
import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai
from dotenv import load_dotenv

# Load OpenAI API key from environment
load_dotenv()
api_key = os.getenv("OPEN_AI_KEY")

# ---------- Cached Model Loader ----------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ---------- Utility Functions ----------

# Extract text from PDF
def extract_text(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Split text into overlapping chunks
def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

# Cached embedding function
@st.cache_resource
def get_embeddings(pdf_path):
    text = extract_text(pdf_path)
    docs = split_text(text)
    embeddings = model.encode(docs)
    return embeddings, docs

# Retrieve top relevant chunks
def search_index(embeddings, texts, query, top_k=5):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)
    top_results = np.argsort(similarities[0])[-top_k:]
    return [texts[i] for i in reversed(top_results)]

# Ask OpenAI with context
def ask_openai(context, question, api_key):
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions using ONLY the provided context. "
                           "If the answer is not in the context, say 'I don’t know based on the document.' Always include "
                           "any relevant restrictions(refer 'Note' in the document), such as character limits, etc., in your response. Don't give just a generic response."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ],
        max_tokens=500,
        temperature=0.3,
    )
    return response['choices'][0]['message']['content'].strip()

# ---------- Streamlit UI ----------

st.title("📄 Skynet Chatbot")

# Module to PDF mapping
module_pdfs = {
    "System Setup - Admin": "system-setup-admin.pdf",
    "Profile Builder - Admin": "profile-builder-admin.pdf",
    "Events - Admin": "events-admin.pdf"
    #...
}

# Dropdown to select module
selected_module = st.selectbox("Choose a Module", list(module_pdfs.keys()))
pdf_path = module_pdfs[selected_module]

st.markdown(f"""
Ask questions related to the **{selected_module} Module**.  
If the response isn't helpful, feel free to rephrase your question and try again.
""")

question = st.text_input("Your Question?")

if st.button("Get Answer"):
    if not question:
        st.warning("Please type a question.")
    elif not api_key:
        st.error("API key not found. Please set your OpenAI API key.")
    else:
        with st.spinner("Retrieving answer..."):
            embeddings, texts = get_embeddings(pdf_path)
            relevant_chunks = search_index(embeddings, texts, question)
            context = "\n".join(relevant_chunks)
            answer = ask_openai(context, question, api_key)

            st.success("Answer:")
            st.write(answer)


