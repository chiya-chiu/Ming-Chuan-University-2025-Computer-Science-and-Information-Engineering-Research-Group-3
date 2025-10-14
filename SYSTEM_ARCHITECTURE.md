# 智能 RAG 學習助理系統 - 完整架構圖

## 系統概覽

```
┌─────────────────────────────────────────────────────────────────────┐
│                    智能 RAG 學習助理系統                              │
│                                                                       │
│  ┌────────────────────┐              ┌────────────────────┐         │
│  │  預處理階段 (ABC)   │  ────────▶  │   運行時階段       │         │
│  │  (一次性/手動觸發)   │              │   (持續服務)       │         │
│  └────────────────────┘              └────────────────────┘         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 一、預處理階段 (ABC 三階段)

### 執行腳本：`preprocess_pdfs.py`

```
┌───────────────────────────────────────────────────────────────────────┐
│                          預處理流程 (ABC)                              │
└───────────────────────────────────────────────────────────────────────┘

輸入: pdfFiles/*.pdf
  │
  ├─────────────────────────────────────────────────────────────────┐
  │                                                                   │
  ▼                                                                   ▼
┌──────────────────────┐                                    ┌──────────────┐
│ 階段 A: 圖表識別      │                                    │ 文字萃取      │
│ (Caption Extraction) │                                    │ (PyMuPDF)    │
└──────────────────────┘                                    └──────────────┘
  │                                                                   │
  │ 使用: PDFChartExtractor                                           │
  │ 方法: 正則表達式識別                                              │
  │ 模式: 圖1-1, 表2-1, Figure 1.1 等                                 │
  │                                                                   │
  ├─▶ 輸出: chart_images/chart_*.png                                 │
  │   輸出: chart_metadata.json (初始)                                │
  │                                                                   │
  ▼                                                                   │
┌──────────────────────┐                                              │
│ 階段 B: LLM 描述生成  │                                              │
│ (Description)        │                                              │
└──────────────────────┘                                              │
  │                                                                   │
  │ 使用: LLMManager (llm_providers.py)                               │
  │ 模型: LLMConfig.CHART_DESCRIPTION_MODEL                           │
  │       (預設: gpt-3.5-turbo)                                       │
  │                                                                   │
  │ Prompt 範例:                                                      │
  │ ┌────────────────────────────────────────────┐                  │
  │ │ 請為以下圖表生成詳細的學術描述:             │                  │
  │ │ 1. 圖表主題和目的                           │                  │
  │ │ 2. 關鍵數據和趨勢                           │                  │
  │ │ 3. 重要結論                                 │                  │
  │ └────────────────────────────────────────────┘                  │
  │                                                                   │
  ├─▶ 更新: chart_metadata.json (新增 description 欄位)              │
  │                                                                   │
  ▼                                                                   │
┌──────────────────────┐                                              │
│ 階段 C: 增強文檔生成  │ ◀────────────────────────────────────────────┘
│ (Enhanced Docs)      │
└──────────────────────┘
  │
  │ 合併策略: 原文 + 圖表資訊
  │
  │ 格式範例:
  │ ┌────────────────────────────────────────────────────────┐
  │ │ === 計算機概論.pdf - 頁面 15 ===                        │
  │ │                                                          │
  │ │ [原始文字內容]                                           │
  │ │ 作業系統是管理電腦硬體與軟體資源的系統軟體...          │
  │ │                                                          │
  │ │ === 圖表資訊 ===                                         │
  │ │ 【圖 1-1】作業系統架構圖                                 │
  │ │ LLM 描述: 此圖展示了作業系統的層次架構，包含...        │
  │ └────────────────────────────────────────────────────────┘
  │
  ▼
輸出: pdfFiles/enhanced_doc_0.txt ~ enhanced_doc_N.txt
```

### 檔案變化偵測機制

```
┌──────────────────────────────────────┐
│ FileChangeDetector                   │
│ (.preprocess_cache.json)             │
└──────────────────────────────────────┘
  │
  │ 每個 PDF 的 MD5 雜湊值
  │
  ├─▶ 若 PDF 未變化 → 跳過處理
  └─▶ 若 PDF 變化 → 重新執行 ABC
```

---

## 二、運行時階段

### 主要服務：`main_web.py` (FastAPI)

```
┌───────────────────────────────────────────────────────────────────────┐
│                          運行時架構                                    │
└───────────────────────────────────────────────────────────────────────┘

啟動流程:
  │
  ▼
┌─────────────────────────────────────┐
│ lifespan() 生命週期管理              │
└─────────────────────────────────────┘
  │
  ├─▶ 初始化資料庫連接池 (PostgreSQL)
  │   └─ ThreadedConnectionPool (5~20 連接)
  │
  ├─▶ 初始化 RAG 實例
  │   └─ MultiTurnRAGHelper(pdf_folder="pdfFiles")
  │
  ├─▶ 載入圖表 metadata
  │   └─ chart_metadata.json → charts_data
  │
  └─▶ 容錯機制: DB 失敗仍可運行 ABC 功能


RAG 初始化流程:
  │
  ▼
┌──────────────────────────────────────────────────────────────────┐
│ MultiTurnRAGHelper.load_and_prepare(use_enhanced_docs=True)      │
└──────────────────────────────────────────────────────────────────┘
  │
  ├─ 檢查: my_faiss_index/ 是否存在?
  │   │
  │   ├─ YES ──▶ 直接載入現有向量資料庫
  │   │          └─ FAISS.load_local()
  │   │
  │   └─ NO ───▶ 建立新的向量資料庫
  │              │
  │              ▼
  │         ┌────────────────────────────────────────┐
  │         │ 優先檢查 enhanced_doc_*.txt 是否存在?  │
  │         └────────────────────────────────────────┘
  │              │
  │              ├─ YES ──▶ 【優先路徑】
  │              │          │
  │              │          ├─ 使用 TextLoader 載入 enhanced_doc
  │              │          ├─ 使用 RecursiveCharacterTextSplitter 切割
  │              │          │   └─ chunk_size=300, overlap=50
  │              │          │
  │              │          └─ 跳過 ABC 階段 (已預處理完成)
  │              │
  │              └─ NO ───▶ 【回退路徑】
  │                         │
  │                         ├─ 載入原始 PDF
  │                         ├─ 使用 Split_Helper 智能切割
  │                         │   └─ target_len=500, tolerance=100
  │                         │
  │                         └─ 無圖表描述整合
  │
  ▼
向量化與索引建立:
  │
  ├─ Embedding 模型: LLMConfig.EMBEDDING_MODEL_NAME
  │                  (預設: text-embedding-3-large)
  │
  ├─ 使用 OpenAI API 將文字段落轉換為向量
  │
  └─ 建立 FAISS 索引
     └─ 儲存至 my_faiss_index/
```

---

## 三、問答流程

```
┌───────────────────────────────────────────────────────────────────────┐
│                          用戶問答流程                                  │
└───────────────────────────────────────────────────────────────────────┘

用戶問題
  │
  ▼
┌──────────────────────────────┐
│ POST /api/chat               │
│ {                            │
│   "question": "什麼是作業系統?",│
│   "conversation_id": "uuid"  │
│ }                            │
└──────────────────────────────┘
  │
  ▼
┌────────────────────────────────────────────────────────────────┐
│ MultiTurnRAGHelper.ask_with_memory()                           │
└────────────────────────────────────────────────────────────────┘
  │
  ├─ 步驟 1: 向量檢索
  │   │
  │   ├─ 將問題轉換為向量 (Embedding)
  │   │
  │   ├─ FAISS 相似度搜尋
  │   │   └─ 參數: k=5, similarity_threshold=0.45
  │   │
  │   └─ 返回最相關的文字段落
  │      ┌────────────────────────────────────────────┐
  │      │ 檢索結果範例:                               │
  │      │ [                                           │
  │      │   {                                         │
  │      │     "content": "作業系統是...",             │
  │      │     "metadata": {                           │
  │      │       "source": "enhanced_doc_15.txt",     │
  │      │       "page": 15,                           │
  │      │       "score": 0.87                         │
  │      │     }                                       │
  │      │   }                                         │
  │      │ ]                                           │
  │      └────────────────────────────────────────────┘
  │
  ├─ 步驟 2: 圖表匹配 (可選)
  │   │
  │   ├─ 從 charts_data 查找相關圖表
  │   │   └─ 使用 is_chart_relevant() 判斷
  │   │
  │   └─ 附加圖表資訊到 context
  │      ┌────────────────────────────────────────────┐
  │      │ 【圖 1-1】作業系統架構圖                    │
  │      │ 描述: 展示了作業系統的層次結構...          │
  │      └────────────────────────────────────────────┘
  │
  ├─ 步驟 3: 多輪對話記憶
  │   │
  │   ├─ ConversationBufferWindowMemory
  │   │   └─ 保存最近 N 輪對話 (預設: 10 輪)
  │   │
  │   └─ 將歷史對話加入 context
  │
  ├─ 步驟 4: LLM 生成回答
  │   │
  │   ├─ 模型: LLMConfig.QA_MODEL_NAME (預設: gpt-4o)
  │   │
  │   ├─ Prompt 組成:
  │   │   ┌────────────────────────────────────────┐
  │   │   │ System: 你是一個專業的教學助理...       │
  │   │   │                                        │
  │   │   │ Context: [檢索到的文字段落]            │
  │   │   │          [相關圖表描述]                │
  │   │   │                                        │
  │   │   │ History: [過去的對話記錄]              │
  │   │   │                                        │
  │   │   │ Question: 什麼是作業系統?              │
  │   │   └────────────────────────────────────────┘
  │   │
  │   └─ 生成結構化回答
  │
  └─ 步驟 5: 返回結果
      │
      └─ JSON 格式:
         ┌─────────────────────────────────────────┐
         │ {                                        │
         │   "answer": "作業系統是管理電腦...",     │
         │   "charts": [                            │
         │     {                                    │
         │       "chart_id": "chart_15_1",         │
         │       "image_path": "chart_15_1.png",   │
         │       "description": "..."               │
         │     }                                    │
         │   ]                                      │
         │ }                                        │
         └─────────────────────────────────────────┘
```

---

## 四、統一配置管理

### llm_config.py 架構

```
┌────────────────────────────────────────────────────────────────┐
│                      LLMConfig (統一配置中心)                   │
└────────────────────────────────────────────────────────────────┘
  │
  ├─ API 配置
  │   ├─ OPENAI_API_KEY
  │   └─ OPENAI_BASE_URL
  │
  ├─ 問答系統 LLM
  │   ├─ QA_MODEL_NAME = "gpt-4o"
  │   ├─ QA_TEMPERATURE = 0.1
  │   └─ QA_MAX_TOKENS = 2000
  │
  ├─ Embedding 模型
  │   └─ EMBEDDING_MODEL_NAME = "text-embedding-3-large"
  │
  ├─ 圖表描述 LLM
  │   ├─ CHART_DESCRIPTION_MODEL = "gpt-3.5-turbo"
  │   ├─ CHART_DESCRIPTION_TEMPERATURE = 0.3
  │   └─ CHART_DESCRIPTION_MAX_TOKENS = 300
  │
  ├─ RAG 檢索配置
  │   ├─ RAG_RETRIEVAL_K = 5
  │   └─ RAG_SIMILARITY_THRESHOLD = 0.45
  │
  └─ 文字切割配置
      ├─ CHUNK_SIZE = 300
      ├─ CHUNK_OVERLAP = 50
      ├─ PDF_TARGET_LEN = 500
      └─ PDF_TOLERANCE = 100

便捷函數:
  ├─ get_openai_client() → OpenAI 客戶端
  ├─ get_openai_embeddings() → OpenAIEmbeddings 實例
  └─ get_chat_llm() → ChatOpenAI 實例
```

---

## 五、資料流向圖

```
┌─────────────────────────────────────────────────────────────────────┐
│                          完整資料流向                                │
└─────────────────────────────────────────────────────────────────────┘

【預處理階段】
PDF 文件
  ↓
  ├─→ PyMuPDF 萃取文字 ──────────────────┐
  │                                       │
  └─→ PDFChartExtractor 識別圖表          │
        ↓                                 │
      圖表圖片 (chart_*.png)              │
        ↓                                 │
      LLM 生成描述                        │
        ↓                                 │
      chart_metadata.json                 │
        ↓                                 │
      合併 ←─────────────────────────────┘
        ↓
  enhanced_doc_*.txt (原文 + 圖表描述)


【運行時階段】
enhanced_doc_*.txt
  ↓
TextLoader 載入
  ↓
RecursiveCharacterTextSplitter 切割
  ↓
文字段落 (chunks)
  ↓
OpenAI Embeddings API
  ↓
向量 (embeddings)
  ↓
FAISS 索引建立
  ↓
my_faiss_index/ (持久化儲存)


【問答階段】
用戶問題
  ↓
OpenAI Embeddings API (問題向量化)
  ↓
FAISS 相似度搜尋
  ↓
檢索相關段落
  ↓
  ├─→ 文字段落
  │
  └─→ 匹配圖表 (從 chart_metadata.json)
        ↓
      組合 Context
        ↓
  OpenAI Chat API (gpt-4o)
        ↓
      生成回答
        ↓
      返回用戶
```

---

## 六、核心類別關係圖

```
┌────────────────────────────────────────────────────────────────┐
│                        類別繼承與依賴關係                       │
└────────────────────────────────────────────────────────────────┘

LLMConfig (llm_config.py)
  │
  ├─ 被使用於 ↓
  │
  ├─→ RAGHelper (RAG_Helper.py)
  │     │
  │     ├─ 屬性:
  │     │   ├─ vectorstore: FAISS
  │     │   ├─ splitter_instance: SplitHelper
  │     │   └─ text_splitter: RecursiveCharacterTextSplitter
  │     │
  │     ├─ 方法:
  │     │   ├─ load_and_prepare(use_enhanced_docs=True)
  │     │   ├─ retrieve_documents(query, k, threshold)
  │     │   ├─ setup_retrieval_chain()
  │     │   └─ ask(question)
  │     │
  │     └─ 快取機制:
  │         ├─ _is_cache_valid()
  │         ├─ _save_to_cache()
  │         └─ _load_from_cache()
  │
  └─→ MultiTurnRAGHelper (MultiTurnRAGHelper.py)
        │
        ├─ 繼承自: RAGHelper 的核心邏輯
        │
        ├─ 新增屬性:
        │   ├─ conversation_memory: Dict[str, ConversationBufferWindowMemory]
        │   ├─ memory_window: int (預設 10)
        │   └─ max_conversations: int (預設 100)
        │
        └─ 新增方法:
            ├─ ask_with_memory(question, conversation_id)
            ├─ get_or_create_memory(conversation_id)
            ├─ clear_conversation(conversation_id)
            └─ get_conversation_history(conversation_id)


SplitHelper (Split_Helper.py)
  │
  └─ 智能文字切割
      ├─ chunk_pdf_full_page()
      ├─ detect_headers_footers()
      └─ smart_merge_chunks()


PDFChartExtractor (caption_extractor.py)
  │
  └─ 圖表識別
      ├─ extract_charts_from_pdf()
      ├─ _extract_caption_pattern()
      └─ _save_chart_image()


LLMManager (llm_providers.py)
  │
  ├─ 支援多種 Provider:
  │   ├─ OpenAIProvider
  │   ├─ MockProvider (測試用)
  │   └─ LocalProvider (本地模型)
  │
  └─ 方法:
      └─ generate(request) → LLMResponse
```

---

## 七、檔案結構樹狀圖

```
專案根目錄/
│
├─ 配置檔案
│   ├─ llm_config.py                    # 統一 LLM 配置
│   ├─ .env                             # 環境變數 (API Keys)
│   └─ .preprocess_cache.json           # 預處理快取
│
├─ 核心 RAG 模組
│   ├─ RAG_Helper.py                    # 基礎 RAG 系統
│   ├─ MultiTurnRAGHelper.py            # 多輪對話 RAG
│   └─ Split_Helper.py                  # 智能文字切割
│
├─ 預處理模組
│   ├─ preprocess_pdfs.py               # ABC 三階段統一腳本
│   └─ modules/pdf_Cutting_TextReplaceImage/
│       ├─ caption_extractor.py         # 階段 A: 圖表識別
│       ├─ enhanced_version/backend/
│       │   ├─ llm_description_generator_v2.py  # 階段 B: LLM 描述
│       │   └─ llm_providers.py         # LLM Provider 抽象層
│       └─ ignore_file/
│           └─ DEVELOPMENT_LOG.md       # 開發日誌
│
├─ Web 服務
│   ├─ main_web.py                      # FastAPI 主服務
│   ├─ static/                          # 前端靜態檔案
│   │   ├─ index.html
│   │   ├─ styles.css
│   │   └─ script.js
│   └─ templates/                       # HTML 模板
│
├─ 資料檔案
│   ├─ pdfFiles/                        # PDF 來源檔案
│   │   ├─ 計算機概論.pdf
│   │   ├─ enhanced_doc_0.txt           # 增強文檔 (階段 C 輸出)
│   │   ├─ enhanced_doc_1.txt
│   │   ├─ ...
│   │   └─ chart_metadata.json          # 圖表 metadata
│   │
│   ├─ chart_images/                    # 圖表圖片 (階段 A 輸出)
│   │   ├─ chart_15_1.png
│   │   └─ ...
│   │
│   ├─ my_faiss_index/                  # FAISS 向量索引
│   │   ├─ index.faiss
│   │   └─ index.pkl
│   │
│   └─ rag_cache/                       # RAG 快取
│       ├─ rag_vectorstore/
│       └─ rag_metadata.json
│
├─ 日誌檔案
│   └─ preprocess_logs/                 # 預處理日誌
│       └─ preprocess_20251004_HHMMSS.log
│
├─ 測試檔案
│   ├─ test_rag_enhanced.py             # RAG 測試
│   └─ modules/pdf_Cutting_TextReplaceImage/enhanced_version/tests/
│       ├─ test_stage_a.py              # 階段 A 測試
│       └─ stage_ab_simple_test.py      # A+B 整合測試
│
└─ 文檔
    ├─ SYSTEM_ARCHITECTURE.md           # 本文件
    ├─ CLAUDE.md                        # 專案說明
    └─ README.md                        # 使用說明
```

---

## 八、關鍵設計決策

### 1. **單一索引架構 vs 雙軌索引**

```
【採用方案】單一索引架構
┌──────────────────────────────────────┐
│ FAISS 向量索引 (統一)                 │
├──────────────────────────────────────┤
│ 文字段落 1: "作業系統是..."           │
│ 文字段落 2: "【圖1-1】作業系統架構..."│ ← 包含圖表描述
│ 文字段落 3: "CPU 的功能..."           │
│ ...                                   │
└──────────────────────────────────────┘

優點:
✅ 架構簡單，易於維護
✅ 檢索時自動包含圖文混合內容
✅ 無需額外的索引合併邏輯


【未採用】雙軌索引架構
┌─────────────────────┐  ┌─────────────────────┐
│ 文字向量索引         │  │ 圖表向量索引         │
├─────────────────────┤  ├─────────────────────┤
│ 純文字段落           │  │ 圖表描述             │
└─────────────────────┘  └─────────────────────┘
          │                        │
          └────────┬───────────────┘
                   ▼
             需要額外的合併邏輯

缺點:
❌ 需要維護兩個獨立索引
❌ 檢索時需要智能合併結果
❌ 增加系統複雜度
```

### 2. **預處理 vs 即時處理**

```
【採用方案】預處理分離
┌─────────────────────────────────────┐
│ preprocess_pdfs.py (一次性)          │
│ ├─ 識別圖表                          │
│ ├─ LLM 描述生成                      │
│ └─ 生成 enhanced_doc                 │
└─────────────────────────────────────┘
          ↓
┌─────────────────────────────────────┐
│ main_web.py (持續服務)               │
│ └─ 載入 enhanced_doc → 向量化        │
└─────────────────────────────────────┘

優點:
✅ 節省 API 成本 (LLM 只調用一次)
✅ 加快啟動速度
✅ 支援批量處理
✅ 便於檢查和調整描述品質


【未採用】即時處理
每次啟動時:
├─ 讀取 PDF
├─ 識別圖表
├─ 調用 LLM 描述 ← 每次都要付費
└─ 向量化

缺點:
❌ 每次啟動都要重新處理
❌ LLM API 成本高
❌ 啟動時間長
```

### 3. **配置管理策略**

```
【採用方案】統一配置文件 (llm_config.py)
┌──────────────────────────────────────────┐
│ LLMConfig                                 │
│ ├─ QA_MODEL_NAME = "gpt-4o"              │
│ ├─ EMBEDDING_MODEL_NAME = "text-emb..."  │
│ └─ CHART_DESC_MODEL = "gpt-3.5-turbo"    │
└──────────────────────────────────────────┘
    ↓ 被引用於
├─ RAG_Helper.py
├─ MultiTurnRAGHelper.py
└─ preprocess_pdfs.py

優點:
✅ 單一真相來源 (Single Source of Truth)
✅ 易於切換模型
✅ 參數集中管理


【未採用】分散配置
RAG_Helper.py:     llm = ChatOpenAI(model="gpt-4o")
MultiTurn...:      llm = ChatOpenAI(model="gpt-5-nano")
preprocess...:     llm = ChatOpenAI(model="gpt-3.5-turbo")

缺點:
❌ 多處硬編碼
❌ 難以維護一致性
❌ 修改需要改多處
```

---

## 九、效能優化機制

### 1. **快取系統**

```
RAG Helper 快取機制:
┌──────────────────────────────────────┐
│ rag_cache/                            │
│ ├─ rag_vectorstore/ (FAISS 索引)     │
│ └─ rag_metadata.json (檔案雜湊)      │
└──────────────────────────────────────┘
  │
  ├─ 檢查: 檔案是否變化? (MD5)
  │   ├─ NO → 直接載入快取
  │   └─ YES → 重新建立索引
  │
  └─ 節省時間: 避免重複向量化
```

### 2. **連接池管理**

```
PostgreSQL 連接池 (main_web.py):
┌──────────────────────────────────────┐
│ ThreadedConnectionPool               │
│ ├─ min_conn = 5                      │
│ ├─ max_conn = 20                     │
│ └─ 自動重用連接                      │
└──────────────────────────────────────┘

優點:
✅ 減少連接建立開銷
✅ 支援高並發請求
✅ 自動連接管理
```

### 3. **批量處理**

```
preprocess_pdfs.py 批量處理:
┌──────────────────────────────────────┐
│ 處理多個 PDF:                         │
│ for pdf in pdf_files:                │
│     if is_changed(pdf):               │
│         ├─ 階段 A: 批量識別           │
│         ├─ 階段 B: 批量 LLM 調用      │
│         └─ 階段 C: 批量生成文檔       │
└──────────────────────────────────────┘

優點:
✅ 減少手動操作
✅ 統一處理流程
✅ 便於監控和除錯
```

---

## 十、未來擴展方向

```
1. 多模態支援
   ├─ 直接將圖片向量化 (CLIP/Vision Transformer)
   └─ 圖文聯合檢索

2. 進階圖表處理
   ├─ OCR 識別表格數據
   ├─ Chart2Data 提取數值
   └─ 自動生成統計洞察

3. 個性化學習
   ├─ 用戶學習歷程追蹤
   ├─ 知識圖譜建構
   └─ 智能推薦系統

4. 分散式部署
   ├─ 向量資料庫分片
   ├─ LLM 服務負載均衡
   └─ 快取層優化
```

---

## 附錄：關鍵技術棧

| 類別 | 技術 | 用途 |
|------|------|------|
| **Web 框架** | FastAPI | API 服務 |
| **資料庫** | PostgreSQL | 用戶資料、對話記錄 |
| **向量資料庫** | FAISS | 文字向量索引與檢索 |
| **LLM** | OpenAI GPT-4o / GPT-3.5 | 問答生成、圖表描述 |
| **Embedding** | text-embedding-3-large | 文字向量化 |
| **PDF 處理** | PyMuPDF (fitz) | PDF 文字/圖片萃取 |
| **文字切割** | LangChain RecursiveCharacterTextSplitter | 智能文字分段 |
| **記憶管理** | LangChain ConversationBufferWindowMemory | 多輪對話記憶 |
| **前端** | HTML/CSS/JavaScript | 用戶界面 |

