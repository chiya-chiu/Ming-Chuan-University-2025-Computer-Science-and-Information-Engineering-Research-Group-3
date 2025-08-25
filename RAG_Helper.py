import asyncio
import glob
import os
import uuid
from pathlib import Path
from typing import List, Tuple, Optional

# langchain 相關套件
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 可以讀取不同的檔案格式
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
)

from Split_Helper import SplitHelper


class RAGHelper:
    def __init__(self, pdf_folder, chunk_size=300, chunk_overlap=50, 
                 pdf_target_len=500, pdf_tolerance=100):
        self.pdf_folder = pdf_folder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.pdf_target_len = pdf_target_len       
        self.pdf_tolerance = pdf_tolerance  
        
        # 核心組件
        self.vectorstore = None
        self.retrieval_chain = None
        self.documents_loaded = False  # 新增：追蹤文檔載入狀態
        self.chain_setup = False       # 新增：追蹤檢索鏈設置狀態

        # 分割器設置
        self.splitter_instance = SplitHelper()
        self.splitter_instance.CENTER_ONLY = False
        self.splitter_instance.NUMBERED_HEADERS_ONLY = False
        self.splitter_instance.SMART_CONSOLIDATE = True

    def get_status(self) -> dict:
        """獲取系統狀態"""
        return {
            "vectorstore_ready": self.vectorstore is not None,
            "documents_loaded": self.documents_loaded,
            "chain_setup": self.chain_setup,
            "retrieval_chain_ready": self.retrieval_chain is not None
        }

    def get_loader(self, path: str):
        ext = Path(path).suffix.lower()
        if ext == ".pdf":
            return PyPDFLoader(path)
        elif ext == ".txt":
            return TextLoader(path, encoding="utf-8")
        elif ext == ".docx":
            return UnstructuredWordDocumentLoader(path)
        elif ext == ".md":
            return UnstructuredMarkdownLoader(path)
        elif ext == ".csv":
            return CSVLoader(path)
        else:
            raise ValueError(f"不支援的檔案類型: {ext}")

    async def load_any_file_async(self, path: str):
        loader = self.get_loader(path)
        if hasattr(loader, "alazy_load"):
            pages = []
            async for page in loader.alazy_load():
                pages.append(page)
            return pages
        else:
            return loader.load()

    def _split_documents(self, documents):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
            length_function=len,
        )
        return splitter.split_documents(documents)

    def _build_vectorstore(self, documents):
        print(f"建立向量資料庫... 共 {len(documents)} 個段落")

        # 確保文檔格式正確
        from langchain.schema import Document as LangchainDocument
        formatted_docs = []

        for i, doc in enumerate(documents):
            metadata = doc.metadata.copy() if hasattr(doc, 'metadata') and doc.metadata else {}
            metadata['id'] = f"doc_{i}_{uuid.uuid4().hex[:8]}"

            formatted_doc = LangchainDocument(
                page_content=doc.page_content if hasattr(doc, 'page_content') else str(doc),
                metadata=metadata
            )
            formatted_docs.append(formatted_doc)

        try:
            # 測試 OpenAI API 連接
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            
            # 測試嵌入一個小樣本
            test_embedding = embeddings.embed_query("測試")
            if not test_embedding:
                raise ValueError("OpenAI 嵌入服務無法正常工作")
            
            # 建立向量資料庫
            self.vectorstore = FAISS.from_documents(formatted_docs, embeddings)
            print("✅ 向量資料庫建立成功")
            
        except Exception as e:
            print(f"❌ 建立向量資料庫失敗: {e}")
            raise e

    async def load_and_prepare(self, file_extensions=None):
        """載入並準備文件"""
        print("開始載入檔案...")
        
        # 重置狀態
        self.documents_loaded = False
        self.chain_setup = False
        self.retrieval_chain = None

        try:
            if os.path.exists("my_faiss_index"):
                print("已偵測到現有向量資料庫，直接載入...")
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                
                # 測試 embeddings 是否正常工作
                test_embedding = embeddings.embed_query("測試")
                if not test_embedding:
                    raise ValueError("OpenAI 嵌入服務無法正常工作，無法載入現有索引")
                
                self.vectorstore = FAISS.load_local(
                    "my_faiss_index",
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                
                # 驗證載入的向量資料庫
                if self.vectorstore is None:
                    raise ValueError("向量資料庫載入失敗")
                
                # 測試檢索功能
                test_results = self.vectorstore.similarity_search("測試", k=1)
                if not test_results:
                    print("⚠️ 警告：向量資料庫中沒有找到任何文檔")
                
                print("✅ 現有向量資料庫載入成功")
                self.documents_loaded = True

            else:
                print("正在建立新的向量資料庫...")
                
                if file_extensions is None:
                    file_extensions = ['.pdf']

                all_chunks = []

                for ext in file_extensions:
                    pattern = f"*{ext}"
                    file_paths = glob.glob(os.path.join(self.pdf_folder, pattern))
                    
                    if not file_paths:
                        print(f"警告：沒有找到 {ext} 格式的檔案")
                        continue

                    for path in file_paths:
                        try:
                            fname = os.path.basename(path)
                            print(f"讀取中: {fname}")

                            if Path(path).suffix.lower() == ".pdf":
                                docs = self.splitter_instance.chunk_pdf_full_page(
                                    pdf_path=path,
                                    target_len=self.pdf_target_len,
                                    tol=self.pdf_tolerance
                                )
                                all_chunks.extend(docs)
                                print(f"✅ {fname}（PDF 智慧切）完成，共 {len(docs)} 段")
                            else:
                                pages = await self.load_any_file_async(path)
                                chunks = self._split_documents(pages)
                                all_chunks.extend(chunks)
                                print(f"✅ {fname} 分割完成，共 {len(chunks)} 段")

                        except Exception as e:
                            print(f"❌ 載入 {fname} 時發生錯誤: {e}")
                            continue

                print(f"所有檔案段落總數：{len(all_chunks)}")

                if len(all_chunks) == 0:
                    raise ValueError("沒有成功載入任何文件，請檢查檔案是否存在且格式正確")

                self._build_vectorstore(all_chunks)
                self.vectorstore.save_local("my_faiss_index")
                print("✅ 向量資料庫已保存到本地")
                self.documents_loaded = True

        except Exception as e:
            print(f"❌ 載入和準備過程失敗: {e}")
            # 清理狀態
            self.vectorstore = None
            self.documents_loaded = False
            self.chain_setup = False
            raise e

    def setup_retrieval_chain(self):
        """設置檢索鏈"""
        print("正在設置檢索鏈...")
        
        # 重置檢索鏈狀態
        self.chain_setup = False
        self.retrieval_chain = None
        
        if not self.vectorstore:
            raise ValueError("請先執行 load_and_prepare() - 向量資料庫未準備好")
        
        if not self.documents_loaded:
            raise ValueError("請先執行 load_and_prepare() - 文檔未載入")

        try:
            # 測試 OpenAI API
            print("測試 OpenAI API 連接...")
            llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
            
            # 簡單測試 LLM 是否正常工作
            test_response = llm.invoke("測試")
            if not test_response:
                raise ValueError("OpenAI API 無法正常工作")
            
            # 創建檢索器
            print("創建檢索器...")
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": 5}
            )
            
            # 測試檢索器
            test_results = retriever.get_relevant_documents("測試查詢")
            if not test_results:
                print("⚠️ 警告：檢索器沒有返回任何結果")
            
            # 創建提示詞模板
            print("創建提示詞模板...")
            system_prompt = (
                "你是一個基於 RAG 系統的計算機概論家教。請參考以下提供的內容來回答問題。"
                "用詞上請多使用正向鼓勵的詞語，並基於現有問題延伸出更多相關的問題。"
                "請針對問題舉出簡單好懂的比喻或例子。"
                "如果不知道如何回答問題，請說出來。"
                "如果問題和計算機概論無關，請將主題拉回計算機概論。"
                "使用 LaTeX 時，請使用 $ 符號作為塊級公式。"
                "請用繁體中文回答。\n\n"
                "{context}"
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            
            # 創建文檔合併鏈
            print("創建文檔合併鏈...")
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            
            # 創建檢索鏈
            print("創建檢索鏈...")
            self.retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)
            
            if not self.retrieval_chain:
                raise ValueError("檢索鏈創建失敗")
            
            self.chain_setup = True
            print("✅ 檢索鏈設置完成")
            
        except Exception as e:
            print(f"❌ 設置檢索鏈失敗: {e}")
            self.retrieval_chain = None
            self.chain_setup = False
            raise ValueError(f"設置檢索鏈失敗：{str(e)}")

    def ask(self, query: str) -> Tuple[str, List]:
        """回答問題"""
        if not self.retrieval_chain:
            status = self.get_status()
            error_msg = f"請先執行 setup_retrieval_chain() - 當前狀態: {status}"
            raise ValueError(error_msg)
        
        if not self.chain_setup:
            raise ValueError("檢索鏈未正確設置，請重新執行 setup_retrieval_chain()")

        try:
            print(f"處理問題：{query}")
            result = self.retrieval_chain.invoke({"input": query})
            
            if not result:
                raise ValueError("檢索鏈返回空結果")
            
            answer = result.get("answer", "")
            context = result.get("context", [])
            
            if not answer:
                raise ValueError("檢索鏈沒有返回答案")
            
            return answer, context
            
        except Exception as e:
            if "max_tokens_per_request" in str(e):
                print("內容過長，嘗試使用較短的上下文...")
                try:
                    self.setup_retrieval_chain_with_shorter_context()
                    result = self.retrieval_chain.invoke({"input": query})
                    return result["answer"], result["context"]
                except Exception as e2:
                    raise ValueError(f"使用較短上下文後仍然失敗：{str(e2)}")
            else:
                raise ValueError(f"回答問題時發生錯誤：{str(e)}")

    def setup_retrieval_chain_with_shorter_context(self):
        """設置更短上下文的檢索鏈"""
        if not self.vectorstore:
            raise ValueError("請先執行 load_and_prepare()")

        llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
        
        system_prompt = (
            "你是一個問答助手。基於以下提供的內容來回答問題。"
            "如果內容中沒有相關資訊，請說「根據提供的資料無法回答這個問題」。"
            "請用繁體中文簡潔回答。\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        self.retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)
        self.chain_setup = True
```

### 2. 強化的初始化端點
```python
@app.post("/initialize")
async def initialize_system(current_user: str = Depends(get_current_user)):
    """初始化 RAG 系統"""
    global rag_instance

    # 預檢查
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="請在 .env 檔案中設定 OPENAI_API_KEY")

    if not os.path.exists("./pdfFiles"):
        raise HTTPException(status_code=500, detail="找不到 pdfFiles 資料夾")

    # 檢查 pdfFiles 資料夾中是否有檔案
    supported_extensions = ['.pdf', '.txt', '.docx', '.md', '.csv']
    found_files = []
    for ext in supported_extensions:
        files = glob.glob(f"./pdfFiles/*{ext}")
        found_files.extend(files)
    
    if not found_files:
        raise HTTPException(
            status_code=500, 
            detail=f"pdfFiles 資料夾中沒有找到支援的檔案 {supported_extensions}"
        )

    try:
        print("🔄 開始初始化 RAG 系統...")
        
        # 步驟 1: 創建 RAG 實例
        print("1️⃣ 創建 RAG 實例...")
        rag_instance = RAGHelper(
            pdf_folder="./pdfFiles", 
            chunk_size=300, 
            chunk_overlap=50
        )
        
        # 步驟 2: 載入和準備文檔
        print("2️⃣ 載入和準備文檔...")
        await rag_instance.load_and_prepare(['.pdf', '.txt', '.docx', '.md', '.csv'])
        
        # 驗證載入狀態
        status = rag_instance.get_status()
        if not status['documents_loaded'] or not status['vectorstore_ready']:
            raise ValueError(f"文檔載入失敗，狀態：{status}")
        
        # 步驟 3: 設置檢索鏈
        print("3️⃣ 設置檢索鏈...")
        rag_instance.setup_retrieval_chain()
        
        # 最終驗證
        final_status = rag_instance.get_status()
        if not all(final_status.values()):
            raise ValueError(f"系統初始化不完整，狀態：{final_status}")
        
        # 測試問答功能
        print("4️⃣ 測試問答功能...")
        test_answer, test_sources = rag_instance.ask("測試")
        if not test_answer:
            raise ValueError("測試問答功能失敗")

        print("✅ RAG 系統初始化完成")
        return StatusResponse(
            status="success",
            message=f"RAG 系統初始化完成，狀態：{final_status}"
        )

    except Exception as e:
        print(f"❌ 系統初始化失敗: {e}")
        # 清理失敗的實例
        rag_instance = None
        raise HTTPException(status_code=500, detail=f"系統初始化失敗：{str(e)}")
```

### 3. 新增診斷端點
```python
@app.get("/debug/rag-status")
async def debug_rag_status(current_user: str = Depends(get_current_user)):
    """診斷 RAG 系統狀態"""
    global rag_instance
    
    if not rag_instance:
        return {
            "status": "not_initialized",
            "message": "RAG 實例未創建",
            "details": None
        }
    
    try:
        status = rag_instance.get_status()
        
        # 額外檢查
        extra_info = {
            "openai_api_key_set": bool(os.getenv("OPENAI_API_KEY")),
            "pdf_folder_exists": os.path.exists("./pdfFiles"),
            "faiss_index_exists": os.path.exists("my_faiss_index"),
            "pdf_files_count": len(glob.glob("./pdfFiles/*.pdf")),
        }
        
        return {
            "status": "initialized" if all(status.values()) else "partial",
            "rag_status": status,
            "extra_info": extra_info,
            "message": "系統正常" if all(status.values()) else "系統部分初始化"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"檢查狀態時發生錯誤：{str(e)}",
            "error": str(e)
        }

@app.post("/debug/force-reinit")
async def force_reinitialize(current_user: str = Depends(get_current_user)):
    """強制重新初始化系統"""
    global rag_instance
    
    try:
        # 清理現有實例
        rag_instance = None
        
        # 重新初始化
        return await initialize_system(current_user)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"強制重新初始化失敗：{str(e)}")
