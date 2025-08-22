# 銘傳大學 2025 資訊工程學系 專研第 3 組
## 智能RAG學習助理 - 圖文混合問答系統

一個結合 **RAG (檢索增強生成)** 與 **圖表處理技術** 的智能學習助理專案，讓AI能夠理解和回答PDF教材中的圖表問題。

### 🎯 **核心創新**
- **傳統RAG限制**：只能處理純文字，無法回答「圖1-1是什麼？」
- **本專案突破**：將PDF圖表轉換成文字描述，實現圖文混合問答

### 🌟 **主要功能**
- 📚 **基礎RAG系統**：支援多格式文檔問答(.pdf, .txt, .docx, .md, .csv)
- 🖼️ **圖表智能識別**：自動識別PDF中的圖表標題(圖1-1、表2-1等)
- 🤖 **AI描述生成**：使用LLM將圖表轉換成詳細文字描述
- 🔗 **圖文混合檢索**：同時搜尋文字內容和圖表描述
- 💻 **Web界面**：提供完整的問答Web應用
- 📊 **評估系統**：整合TruLens進行RAG系統性能評估

## 🚀 快速安裝（推薦）

### 方法一：一鍵安裝腳本
```bash
# 1. Clone 專案（不需要 --recurse-submodules）
git clone https://github.com/your-username/Ming-Chuan-University-2025-Computer-Science-and-Information-Engineering-Research-Group-3.git
cd Ming-Chuan-University-2025-Computer-Science-and-Information-Engineering-Research-Group-3

# 2. 運行自動安裝腳本
# Linux/macOS
chmod +x setup.sh
./setup.sh

# Windows
setup.bat
```

腳本會自動處理：
- ✅ Git submodules 初始化
- ✅ Python 環境檢查
- ✅ 虛擬環境建立（可選）
- ✅ 所有套件安裝
- ✅ .env 設定檔模板建立

### 方法二：傳統安裝
如果你不想使用自動腳本，可以依照以下步驟：

```bash
# Clone 專案並初始化 submodules
git clone --recurse-submodules https://github.com/your-username/Ming-Chuan-University-2025-Computer-Science-and-Information-Engineering-Research-Group-3.git
```

## 📦 手動安裝必要套件
為避免套件因版本互相干擾，建議使用虛擬環境安裝
```bash
pip install langchain
pip install -qU "langchain[openai]"
pip install -qU langchain-openai
pip install -qU langchain-community
pip install faiss-cpu
pip install -qU pypdf
pip install unstructured
pip install python-dotenv
pip install fastapi uvicorn python-multipart
pip install PyJWT python-multipart email-validator pydantic[email]
```
# .env 檔案
由於 API 金鑰需保密請參考這一影片 : https://youtu.be/-yf2nkZeiDU?si=KKhdLM3zw1pAVXJr 申請一個 API 金鑰  

並在根目錄中新增 ".env" 檔案，在裡面填上： 
```python
OPENAI_API_KEY="API_KEY"
OPENAI_BASE_URL=https://api.chatanywhere.org/v1

# JWT 密鑰（請改成你自己的隨機字串）
SECRET_KEY=隨機字串

# 資料庫設定（SQLite 會自動建立檔案）
DATABASE_URL=sqlite:///./rag_users.db
```
其中 API_KEY 請改成申請到的金鑰  

SECRET_KEY 請加上隨機的字串，比如 `This-kid-aspires-to-be-homeless`

## 🏗️ **系統架構**

### **📁 專案結構**
```
📦 專案根目錄
├── 🌟 核心RAG系統
│   ├── Main.py              # 命令列問答入口
│   ├── RAG_Helper.py        # RAG核心邏輯
│   ├── main_web.py          # 基礎Web界面
│   └── pdfFiles/            # PDF文件資料夾
│
├── 🖼️ 圖表處理子系統
│   └── modules/pdf_Cutting_TextReplaceImage/
│       ├── 基礎模組/
│       │   ├── caption_extractor.py     # 階段A：圖表標題識別
│       │   ├── pdf_chart_extractor.py   # 圖片萃取
│       │   ├── dto.py                   # 資料結構定義
│       │   └── interfaces.py            # 介面抽象層
│       │
│       └── enhanced_version/           # 完整增強版實作
│           ├── backend/
│           │   ├── enhanced_rag_helper.py      # 增強RAG系統
│           │   ├── llm_providers.py            # LLM提供者架構
│           │   ├── llm_description_generator_v2.py  # 描述生成器
│           │   └── enhanced_main_web.py        # 增強Web界面
│           ├── frontend/               # Web前端界面
│           └── tests/                  # 完整測試套件
│
├── 📊 評估與測試
│   ├── rag_evaluation_experiment.py    # RAG評估實驗
│   └── trulens_test.py                 # TruLens評估
│
└── 🔧 設定檔案
    ├── setup.sh / setup.bat           # 自動安裝腳本  
    ├── .env                           # 環境變數設定
    └── requirements.txt               # 套件依賴
```

### **🔄 技術流程**
```
PDF文檔 → 分岔處理
    ├── 📝 文字流程: 文字萃取 → 切割 → 向量化 → 基礎FAISS
    └── 🖼️ 圖表流程: 階段A(識別) → 階段B(LLM描述) → 階段C(向量化) → 增強FAISS
                                                                ↓
用戶問題 → 🔍 混合檢索(文字+圖表) → 🤖 LLM整合回答 → 👤 用戶
```

## 🚀 **使用方式**

### **模式1：基礎RAG系統（純文字）**
```bash
# 命令列問答
python Main.py

# Web界面
python main_web.py
# 瀏覽器開啟 http://localhost:8000
```

### **模式2：增強RAG系統（圖文混合）**
```bash
# 啟動增強版Web系統
cd modules/pdf_Cutting_TextReplaceImage/enhanced_version/backend
python enhanced_main_web.py
# 瀏覽器開啟 http://localhost:8000
```

### **模式3：階段性測試**
```bash
# 測試圖表識別功能（階段A）
cd modules/pdf_Cutting_TextReplaceImage/enhanced_version/tests
python test_stage_a.py

# 測試A+B整合功能
python stage_ab_simple_test.py

# 快速功能驗證
cd modules/pdf_Cutting_TextReplaceImage
python test_stage_A_complete.py
```

## 📊 **核心技術模組**

### **階段A：圖表識別**
- **檔案**：`caption_extractor.py`
- **功能**：使用正則表達式識別PDF中的圖表標題
- **支援格式**：圖1-1、表2-1、Figure 1.1等
- **準確率**：50-60%（可持續優化）

### **階段B：AI描述生成**
- **檔案**：`llm_description_generator_v2.py` + `llm_providers.py`
- **功能**：將圖表標題轉換成詳細文字描述
- **LLM支援**：OpenAI GPT / Mock LLM / 本地模型
- **特色**：可切換LLM提供者架構

### **階段C：RAG整合**
- **檔案**：`enhanced_rag_helper.py`
- **功能**：圖文混合向量化，伴生索引策略
- **技術**：LangChain + FAISS + OpenAI Embeddings

### **階段D：Web展示**
- **檔案**：`enhanced_main_web.py` + `frontend/`
- **功能**：完整Web問答界面
- **特色**：用戶系統、圖表庫、統計面板

## 📈 **系統特色**

### **🔧 模組化設計**
- 每個階段獨立可測試
- 基礎RAG與圖表處理可選擇性使用
- 清晰的介面抽象和依賴注入

### **🤖 LLM架構**
- 支援OpenAI GPT、Mock LLM、本地模型
- 自動選擇可用的LLM提供者
- 統一的請求/回應介面

### **📊 評估體系**
- 整合TruLens RAG評估框架
- 支援Groundedness、Answer Relevance等指標
- 完整的測試套件覆蓋

### **🌐 Web體驗**
- 響應式前端設計
- 即時圖文混合問答
- 完整的用戶管理系統

## 📝 **檔案說明**

### **核心檔案**
- `RAG_Helper.py`：基礎RAG系統核心，負責文檔處理、向量化、問答鏈
- `Main.py`：命令列問答入口，適合快速測試和開發
- `main_web.py`：基礎Web後端，提供用戶系統和Web問答
- `enhanced_main_web.py`：增強Web後端，包含完整圖表處理功能

### **圖表處理模組**
- `caption_extractor.py`：圖表標題識別核心邏輯
- `llm_providers.py`：LLM提供者抽象架構  
- `enhanced_rag_helper.py`：圖文混合RAG系統

### **資料與設定**
- `pdfFiles/`：PDF文檔資料來源，目前包含計算機概論教材
- `static/`：基礎Web前端資源（HTML、CSS、JS）
- `enhanced_version/frontend/`：增強版Web前端界面
- `.env`：環境變數設定（API金鑰、資料庫連線等）

## 🔬 **測試與評估**

### **功能測試**
```bash
# 圖表識別測試
python modules/pdf_Cutting_TextReplaceImage/test_stage_a_complete.py

# 完整流程測試  
python modules/pdf_Cutting_TextReplaceImage/enhanced_version/tests/stage_ab_simple_test.py

# RAG評估測試
python rag_evaluation_experiment.py
```

### **性能指標**
- **圖表識別準確率**：50-60%（持續優化中）
- **系統回應時間**：<2秒
- **支援檔案格式**：Digital PDF、多種文檔格式
- **LLM整合**：支援多種模型切換

## 🤝 **貢獻與開發**

### **開發指南**
1. **Fork** 專案到您的GitHub帳號
2. **Clone** 到本地：`git clone [your-fork-url]`
3. **安裝依賴**：執行 `./setup.sh` 或 `setup.bat`
4. **設定環境**：配置 `.env` 檔案
5. **測試功能**：執行相關測試腳本
6. **提交PR**：歡迎貢獻改進！

### **已知限制**
- 目前僅支援Digital PDF（不支援掃描版）
- Caption識別準確率有待提升
- Mock LLM描述品質有限，建議使用真實API
- 需要穩定網路連接（使用外部API時）

### **未來規劃**
- [ ] 整合本地LLM（Ollama、llama.cpp）
- [ ] 支援掃描版PDF的OCR處理
- [ ] 提升Caption識別準確率到70%+
- [ ] 增加更多檔案格式支援
- [ ] 實作PDF回寫功能（將描述嵌入原PDF）
- [ ] 多語言支援
- [ ] 改進Web界面用戶體驗

## 📚 **參考資料**

### **技術文檔**
- [LangChain官方文檔](https://docs.langchain.com/)
- [FAISS向量資料庫](https://faiss.ai/)
- [TruLens RAG評估](https://www.trulens.org/)
- [OpenAI API文檔](https://platform.openai.com/docs)

### **學術參考**
- [RAG系統評估方法論](https://hackmd.io/@YungHuiHsu/H16Y5cdi6)
- [AWS自動化RAG評估](https://aws.amazon.com/cn/blogs/china/automated-rag-project-assessment-testing-using-trulens/)

### **專案相關**
- [計算機概論教材來源](https://hackmd.io/@110FJU-MIIA/Sy2xnSE8K)
- [API金鑰申請教學](https://youtu.be/-yf2nkZeiDU?si=KKhdLM3zw1pAVXJr)

## 📞 **聯絡方式**

**銘傳大學 2025 資訊工程學系 專研第 3 組**

如有問題或建議，歡迎透過以下方式聯繫：
- 📧 提交 [GitHub Issues](https://github.com/your-username/Ming-Chuan-University-2025-Computer-Science-and-Information-Engineering-Research-Group-3/issues)
- 💬 參與 [Discussions](https://github.com/your-username/Ming-Chuan-University-2025-Computer-Science-and-Information-Engineering-Research-Group-3/discussions)

---

⭐ **如果這個專案對您有幫助，請給我們一個星星！** ⭐
