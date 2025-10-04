import re
import os
from typing import List, Dict
from embedding import build_faiss_index
from search import search_faiss_with_cost, embed_query_with_cost

def load_book_with_pages(filepath: str) -> List[Dict]:
    """
    讀取文章，依照標記格式切分，回傳 {page, text}
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 根據頁碼標記切分 => 依實際檔案修改format
    chunks = re.split(r"(\[P\.\d+\])", content)
    
    results = []
    current_page = None
    buffer = []
    for chunk in chunks:
        match = re.match(r"\[P\.(\d+)\]", chunk)
        if match:
            # 遇到新的頁碼，將前一頁儲存
            if current_page and buffer:
                results.append({
                    "page": current_page,
                    "text": "".join(buffer).strip()
                })
                buffer = []
            current_page = int(match.group(1))
        else:
            buffer.append(chunk)
    
    if current_page and buffer:
        results.append({
            "page": current_page,
            "text": "".join(buffer).strip()
        })
    
    return results

if __name__ == "__main__":
    user_input = input("請輸入查詢：")
    results = search_faiss_with_cost(user_input, "embedding_files/week1.txt.index", "embedding_files/week1.txt_meta.json", 1)
    print(results)