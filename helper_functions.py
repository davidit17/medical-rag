import os
import re
import pickle
import faiss
import numpy as np
import torch
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple

# Constants
SENTENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DIALOG_MODEL = "google/flan-t5-small"
FAISS_DB_DIR = "./faiss_pdf_db"
DEFAULT_QUERY = "how to Set Assay in case of Calibrator and Control Assays with Extraction+PCR protocol"


@st.cache_resource
def load_models():

    with st.spinner("Loading models..."):
        # Sentence transformer
        sentence_model = SentenceTransformer(SENTENCE_MODEL)
        
        # Dialog model
        tokenizer = AutoTokenizer.from_pretrained(DIALOG_MODEL)
        dialog_model = AutoModelForSeq2SeqLM.from_pretrained(DIALOG_MODEL)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    return sentence_model, dialog_model, tokenizer

@st.cache_resource
def load_database():

    index = faiss.read_index(os.path.join(FAISS_DB_DIR, "faiss_index.bin"))
    
    with open(os.path.join(FAISS_DB_DIR, "metadata.pkl"), 'rb') as f:
        metadata = pickle.load(f)
    
    # Extract summary data (assuming it's the last item)
    summary_data = metadata.pop() if metadata else {}
    
    return index, metadata, summary_data


def search_documents(query: str, model: SentenceTransformer, index, metadata: List[Dict], top_k: int = 5) -> List[Dict]:

    query_embedding = model.encode([query]).astype(np.float32)
    faiss.normalize_L2(query_embedding)
    
    scores, indices = index.search(query_embedding, top_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            results.append({
                **metadata[idx],
                'similarity_score': float(score)
            })
    
    return results

def get_similar_words(query: str, text: str, model: SentenceTransformer, top_n: int = 10) -> List[str]:

    words = list(set(w.strip(".,!?;:()[]{}\"'") for w in text.lower().split() if len(w) > 2))
    
    query_emb = model.encode([query])[0]
    word_embs = model.encode(words)
    
    similarities = cosine_similarity([query_emb], word_embs)[0]
    ranked_words = sorted(zip(words, similarities), key=lambda x: x[1], reverse=True)
    
    return [word for word, _ in ranked_words[:top_n]]

def highlight_text(text: str, words: List[str]) -> str:

    for word in words:
        pattern = r'\b' + re.escape(word) + r'\b'
        text = re.sub(pattern, f'<mark style="background-color:yellow;">{word}</mark>', 
                     text, flags=re.IGNORECASE)
    return text


def generate_ai_answer(query: str, context: str, model, tokenizer, max_length: int = 200) -> str:

    try:
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        
        input_ids = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=512)
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=input_ids.shape[1] + max_length,
                temperature=0.7,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                attention_mask=torch.ones_like(input_ids),
                no_repeat_ngram_size=2
            )
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text
       
        
    except Exception as e:
        return f"Error generating answer: {str(e)}"
