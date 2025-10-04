# 文字嵌入搜尋系統 (Text to Embedding Search System)

這是一個基於 OpenAI 嵌入模型和 FAISS 向量資料庫的文字搜尋系統，能夠將文字文件轉換為向量嵌入並進行語義搜尋。

## 功能特色

- 🔍 **語義搜尋**：使用 OpenAI 的 `text-embedding-3-small` 模型進行語義搜尋
- ⚡ **高效檢索**：基於 FAISS 向量資料庫，支援快速相似度搜尋
- 💰 **成本追蹤**：即時計算 API 使用成本
- ⏱️ **效能監控**：詳細的執行時間分析
- 📄 **多文件支援**：支援多個文字文件的索引和搜尋
- 🚀 **快取機制**：內建快取系統提升搜尋效能

## 專案結構

```
txt_to_embedding/
├── embedding.py          # 嵌入生成和索引建立
├── search.py             # 搜尋功能和成本計算
├── main.py               # 主程式入口
├── txt_files/            # 原始文字文件請以txt放置於此
├── embedding_files/      # 生成的index與data放這
└── README.md
```

## 安裝需求

### Python 套件

```bash
pip install openai faiss-cpu numpy tiktoken python-dotenv
```

### 環境設定

1. 建立 `.env` 檔案並設定 OpenAI API Key：

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## 使用方法

### 1. 建立文字嵌入索引

```python
from embedding import build_faiss_index
from main import load_book_with_pages

# 載入並處理文字文件
chunks = load_book_with_pages("txt_files/week1.txt")

# 建立 FAISS 索引
build_faiss_index(
    chunks, 
    index_path="embedding_files/week1.txt.index",
    meta_path="embedding_files/week1.txt_meta.json"
)
```

### 2. 執行搜尋

```python
from search import search_faiss_with_cost

# 搜尋相似內容
results = search_faiss_with_cost(
    query="太空旅行的準備工作",
    index_path="embedding_files/week1.txt.index",
    meta_path="embedding_files/week1.txt_meta.json",
    top_k=3
)

print(results)
```

### 3. 執行主程式

```bash
python main.py
```

程式會提示您輸入查詢內容，並顯示搜尋結果。

## 核心功能說明

### 文字處理 (`main.py`)

- **`load_book_with_pages()`**：根據頁碼標記 `[P.數字]` 切分文字內容
- 支援多頁文件的分段處理
- 回傳包含頁碼和文字內容的字典列表

### 嵌入生成 (`embedding.py`)

- **`embed_texts()`**：將文字列表轉換為向量嵌入
- **`build_faiss_index()`**：建立 FAISS 索引並儲存到檔案
- 使用 OpenAI 的 `text-embedding-3-small` 模型

### 搜尋功能 (`search.py`)

- **`search_faiss_with_cost()`**：執行語義搜尋並計算成本
- **`embed_query_with_cost()`**：將查詢轉換為嵌入並計算費用
- **`count_tokens()`**：計算文字 token 數量
- **快取機制**：避免重複載入索引和元數據

## 成本計算

系統會即時計算 OpenAI API 的使用成本：

- **模型**：`text-embedding-3-small`
- **價格**：$0.00002 USD per 1K tokens
- **顯示資訊**：Token 數量、嵌入成本、執行時間

## 效能優化

- **快取機制**：索引和元數據載入後會快取在記憶體中
- **時間分析**：詳細記錄各階段執行時間
- **批次處理**：支援多個文字的批次嵌入

## 範例輸出

```
Query Tokens: 15
Embedding Cost: $0.00000030 USD
Embedding Time: 0.234s
Index 載入時間: 0.012s
FAISS 搜尋時間: 0.001s
共找到 3 筆相似內容
查詢耗費：$0.00000030 (embedding cost only)
總耗時：0.25s
時間分解：Index載入(0.012s) + Embedding(0.234s) + 搜尋(0.001s)
```

## 注意事項

1. **API Key 安全**：請確保 `.env` 檔案不要提交到版本控制系統
2. **成本控制**：建議設定 API 使用限制以避免意外費用
3. **檔案格式**：文字文件需要包含頁碼標記 `[P.數字]` 才能正確分段
4. **記憶體使用**：大型文件可能需要較多記憶體來建立索引
