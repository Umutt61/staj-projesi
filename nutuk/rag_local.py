import os
import pickle
import numpy as np
import faiss
import google.generativeai as genai
os.environ["GOOGLE_API_KEY"] = "api key giriniz."

# === Ayarlar ===
CHUNK_PATH = "chunks.pkl"
EMB_PATH = "embeddings.npy"
INDEX_PATH = "faiss.index"

EMBED_MODEL = "models/text-embedding-004"
MODEL = "gemini-1.5-flash"

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# === Metinleri parçalara ayır ===
def chunk_text(text, max_tokens=500):
    words = text.split()
    chunks, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) >= max_tokens:
            chunks.append(" ".join(cur))
            cur = []
    if cur:
        chunks.append(" ".join(cur))
    return chunks

# === Embed et ===
def embed_texts(texts):
    embeddings = []
    for t in texts:
        resp = genai.embed_content(model=EMBED_MODEL, content=t)
        embeddings.append(resp["embedding"])
    return np.array(embeddings, dtype="float32")

# === Bellekten oku veya hesapla ===
def get_or_build_chunks_and_embeddings():
    if os.path.exists(CHUNK_PATH) and os.path.exists(EMB_PATH) and os.path.exists(INDEX_PATH):
        print(">> Kaydedilmiş index yükleniyor...")
        with open(CHUNK_PATH, "rb") as f:
            chunks = pickle.load(f)
        embs = np.load(EMB_PATH)
        index = faiss.read_index(INDEX_PATH)
    else:
        print(">> Embedding hesaplanıyor...")
        with open("nutuk.txt", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_text(text)
        embs = embed_texts(chunks)

        # FAISS index
        dim = embs.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embs)

        with open(CHUNK_PATH, "wb") as f:
            pickle.dump(chunks, f)
        np.save(EMB_PATH, embs)
        faiss.write_index(index, INDEX_PATH)

    return chunks, embs, index

# === Sorgu ===
def search(query, chunks, index, top_k=3):
    q_emb = embed_texts([query])
    D, I = index.search(q_emb, top_k)
    return [chunks[i] for i in I[0]]

# === Yanıtla ===
def answer(query, chunks, index):
    rel_chunks = search(query, chunks, index)
    context = "\n".join(rel_chunks)
    prompt = f"Soru: {query}\n\nCevap için bağlam:\n{context}\n\nCevap:"
    resp = genai.GenerativeModel(MODEL).generate_content(prompt)
    return resp.text

# === Çalıştır ===
def main():
    chunks, embs, index = get_or_build_chunks_and_embeddings()
    while True:
        q = input("Soru: ")
        if q.lower() in ["exit", "quit", "q"]:
            break
        print("Cevap:", answer(q, chunks, index))

if __name__ == "__main__":
    main()