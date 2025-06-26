import os
import tempfile
import git
import torch
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# ✅ Use GPT-Neo 125M
model_name = "EleutherAI/gpt-neo-125M"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# Embedding model (free to use)
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# In-memory document store
DOC_STORE = []
EMBEDDINGS = []
index = None  # FAISS index

# 🔹 Convert text into vector embedding
def embed_text(text):
    return embed_model.encode(text)

# 🔹 Add a document to store and FAISS index
def upsert_text(text):
    global index
    emb = embed_text(text)
    EMBEDDINGS.append(emb)
    DOC_STORE.append(text)
    if index is None:
        index = faiss.IndexFlatL2(len(emb))
    index.add(emb.reshape(1, -1))

# 🔹 Extract Markdown or text files from repo
def extract_markdown_files(repo_dir):
    texts = []
    for root, _, files in os.walk(repo_dir):
        for file in files:
            if file.endswith(".md") or file.endswith(".txt"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    texts.append(f.read())
    return texts

# 🔹 Handle Git repo input
def process_git_repo(repo_url, branch):
    with tempfile.TemporaryDirectory() as tmpdirname:
        git.Repo.clone_from(repo_url, tmpdirname, branch=branch)
        docs = extract_markdown_files(tmpdirname)
        for doc in docs:
            upsert_text(doc)
    return {"status": "success", "docs_upserted": len(docs)}

# 🔹 Handle uploaded files
def process_uploaded_files(file):
    text = file.file.read().decode("utf-8")
    upsert_text(text)
    return {"status": "success", "docs_upserted": 1}

# 🔹 Query documents using RAG
def query_rag(query):
    if index is None or not DOC_STORE:
        return {"error": "No documents in database."}
    
    q_emb = embed_text(query)
    D, I = index.search(q_emb.reshape(1, -1), 1)

    if not I[0].size or I[0][0] >= len(DOC_STORE):
        return {"error": "No relevant documents found."}

    context = DOC_STORE[I[0][0]]
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids

    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=100, do_sample=False)

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    answer = response.split("Answer:")[-1].strip()

    return {
        "query": query,
        "matched_doc": context,
        "llm_answer": answer
    }
