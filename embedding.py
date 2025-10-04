import faiss
import numpy as np
import json
from openai import OpenAI
from typing import List, Dict
import os
import dotenv

dotenv.load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    依序轉換為 embeddings
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    vectors = [r.embedding for r in response.data]
    return np.array(vectors, dtype="float32")

def build_faiss_index(chunks: List[Dict], index_path="book_embedding.index", meta_path="book_metadata.json"):
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)

    # 建立 index
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    # 儲存到index檔案中，方便後續取用
    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"已建立 index，共 {len(chunks)} 筆")