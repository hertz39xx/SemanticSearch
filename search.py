import faiss
import json
from embedding import embed_texts
import tiktoken
import time
import numpy as np
from openai import OpenAI
import os
import dotenv

dotenv.load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 快取變數，避免重複載入
_index_cache = {}
_metadata_cache = {}

def search_faiss_with_cost(query, index_path="book.index", meta_path="book_metadata.json", top_k=3):
    start_total = time.time()
    
    # 載入 index & metadata (使用快取)
    index_load_start = time.time()
    if index_path not in _index_cache:
        _index_cache[index_path] = faiss.read_index(index_path)
    if meta_path not in _metadata_cache:
        with open(meta_path, "r", encoding="utf-8") as f:
            _metadata_cache[meta_path] = json.load(f)
    
    index_load_time = time.time() - index_load_start
    print(f"📁 Index 載入時間: {index_load_time:.3f}s")
    
    index = _index_cache[index_path]
    chunks = _metadata_cache[meta_path]

    # 查詢 embedding 與成本
    q_vec, embed_cost, embed_time = embed_query_with_cost(query)

    # 搜尋最近鄰
    search_start = time.time()
    D, I = index.search(np.array([q_vec]), top_k)
    search_time = time.time() - search_start
    print(f"🔍 FAISS 搜尋時間: {search_time:.3f}s")

    results = []
    for idx in I[0]:
        if idx < len(chunks):
            results.append(chunks[idx])

    total_time = time.time() - start_total
    
    print(f"🔍 搜尋完成，共找到 {len(results)} 筆相似內容")
    print(f"💵 本次查詢總成本：${embed_cost:.8f} (embedding cost only)")
    print(f"🕒 總耗時：{total_time:.2f}s")
    print(f"📊 時間分解：Index載入({index_load_time:.3f}s) + Embedding({embed_time:.3f}s) + 搜尋({search_time:.3f}s)")

    return {
        "results": results,
        "embedding_cost_usd": embed_cost,
        "embedding_time_sec": embed_time,
        "total_time_sec": total_time,
        "index_load_time_sec": index_load_time,
        "search_time_sec": search_time
    }

def count_tokens(text, model="text-embedding-3-small"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def embed_query_with_cost(query, model="text-embedding-3-small"):
    """
    將使用者查詢轉成 embedding，同時計算花費。
    """
    start = time.time()
    token_count = count_tokens(query, model=model)

    response = client.embeddings.create(
        model=model,
        input=query
    )
    end = time.time()

    embedding = np.array(response.data[0].embedding, dtype="float32")

    # 成本計算
    price_per_1k = 0.00002  # text-embedding-3-small 單價
    cost = (token_count / 1000) * price_per_1k

    print(f"🧠 Query Tokens: {token_count}")
    print(f"💰 Embedding Cost: ${cost:.8f} USD")
    print(f"⚡ Embedding 時間: {end - start:.3f}s")

    return embedding, cost, end - start

# 清除快取的函數（可選）
def clear_cache():
    global _index_cache, _metadata_cache
    _index_cache.clear()
    _metadata_cache.clear()
    print("🗑️ 快取已清除")