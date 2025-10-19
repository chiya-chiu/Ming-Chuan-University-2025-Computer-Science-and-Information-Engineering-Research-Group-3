import asyncio
import glob
import os
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

# langchain 相關套件
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain, ConversationChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain.schema import BaseRetriever, Document, AIMessage, HumanMessage
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from pydantic import PrivateAttr

# 可以讀取不同的檔案格式
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
)

from Split_Helper import SplitHelper
from llm_config import LLMConfig, get_openai_embeddings, get_chat_llm


class MultiTurnRAGHelper:
    def __init__(self, pdf_folder, chunk_size=300, chunk_overlap=50, pdf_target_len=500,
                 pdf_tolerance=100, memory_window=3, max_conversations=100):
        self.pdf_folder = pdf_folder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.pdf_target_len = pdf_target_len
        self.pdf_tolerance = pdf_tolerance
        self.vectorstore = None
        self.retrieval_chain = None
        self.conversation_chain = None

        # 多輪對話相關
        self.conversation_memory = {}  # 存儲每個用戶的對話記憶
        self.memory_window = memory_window  # 記憶窗口大小

        self.splitter_instance = SplitHelper()
        self.splitter_instance.CENTER_ONLY = False
        self.splitter_instance.NUMBERED_HEADERS_ONLY = False
        self.splitter_instance.SMART_CONSOLIDATE = True
        self.max_conversations = max_conversations

    def get_or_create_memory(self, user_id: str) -> ConversationBufferWindowMemory:
        """為每個用戶獲取或創建對話記憶"""
        # 檢查是否超過最大對話數
        if len(self.conversation_memory) >= self.max_conversations:
            # 清理最舊的對話
            oldest_key = min(self.conversation_memory.keys())
            del self.conversation_memory[oldest_key]

        if user_id not in self.conversation_memory:
            self.conversation_memory[user_id] = ConversationBufferWindowMemory(
                k=self.memory_window,  # 保留最近的 k 輪對話
                return_messages=True,
                memory_key="chat_history"
            )
        return self.conversation_memory[user_id]

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

        # 使用統一配置的 Embedding 模型
        embeddings = get_openai_embeddings()
        self.vectorstore = FAISS.from_documents(formatted_docs, embeddings)

    def retrieve_documents(self, query, k=5, similarity_threshold=0.45):
        """檢索相關文件，並根據相似度門檻過濾結果"""
        if not self.vectorstore:
            print("❌ 向量資料庫未初始化，請先執行 load_and_prepare()")
            return []

        try:
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
            filtered_docs = []

            for doc, score in docs_with_scores:
                similarity = 1 / (1 + score)
                if similarity >= similarity_threshold:
                    filtered_docs.append(doc)

            return filtered_docs

        except Exception as e:
            print(f"❌ 檢索過程中發生錯誤: {e}")
            return []

    async def load_and_prepare(self, file_extensions=None, use_enhanced_docs=True):
        """
        載入並準備文件

        Args:
            file_extensions: 要載入的檔案副檔名列表，例如 ['.pdf', '.txt', '.docx']
            use_enhanced_docs: 是否優先使用 enhanced_doc_*.txt（預處理後的增強文件）
        """
        print("開始載入檔案...")

        if os.path.exists("my_faiss_index"):
            print("已偵測到現有向量資料庫，直接載入...")
            self.vectorstore = FAISS.load_local(
                "my_faiss_index",
                get_openai_embeddings(),
                allow_dangerous_deserialization=True
            )
        else:
            print("正在建立和讀取向量資料庫")

            # ★ 優先檢查是否存在 enhanced_doc_*.txt（預處理後的增強文件）
            enhanced_docs = []
            if use_enhanced_docs:
                enhanced_pattern = os.path.join(self.pdf_folder, "enhanced_doc_*.txt")
                enhanced_docs = glob.glob(enhanced_pattern)

            all_chunks = []

            if enhanced_docs:
                # ★ 發現 enhanced_doc 檔案，優先使用它們（已包含原文+圖表描述）
                print(f"偵測到 {len(enhanced_docs)} 個 enhanced_doc 檔案，優先載入...")

                for path in enhanced_docs:
                    try:
                        fname = os.path.basename(path)
                        print(f"讀取增強文件: {fname}")

                        # 使用 TextLoader 讀取增強文件
                        loader = TextLoader(path, encoding='utf-8')
                        pages = loader.load()

                        # 使用標準文字切割
                        chunks = self._split_documents(pages)
                        all_chunks.extend(chunks)
                        print(f" {fname} 載入完成，共 {len(chunks)} 段")

                    except Exception as e:
                        print(f"載入 {fname} 時發生錯誤: {e}")
            else:
                # ★ 沒有 enhanced_doc，回退到原本的 PDF/其他檔案處理流程
                print("未發現 enhanced_doc 檔案，使用原始檔案處理流程...")

                if file_extensions is None:
                    file_extensions = ['.txt']

                for ext in file_extensions:
                    pattern = f"*{ext}"
                    file_paths = glob.glob(os.path.join(self.pdf_folder, pattern))

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
                                print(f" {fname}（PDF 智慧切）完成，共 {len(docs)} 段")
                            else:
                                pages = await self.load_any_file_async(path)
                                chunks = self._split_documents(pages)
                                all_chunks.extend(chunks)
                                print(f" {fname} 分割完成，共 {len(chunks)} 段")

                        except Exception as e:
                            print(f"載入 {os.path.basename(path)} 時發生錯誤: {e}")

            print(f"所有檔案段落總數：{len(all_chunks)}")

            if len(all_chunks) == 0:
                raise ValueError("沒有成功載入任何文件")

            self._build_vectorstore(all_chunks)
            self.vectorstore.save_local("my_faiss_index")

    def setup_retrieval_chain(self, k=5, similarity_threshold=None):
        """設置檢索鏈（支持多輪對話）"""
        if not self.vectorstore:
            raise ValueError("請先執行 load_and_prepare()")

        # 使用統一配置的 LLM
        llm = get_chat_llm()

        # 創建帶有相似度門檻的檢索器
        if similarity_threshold is not None:
            class ThresholdRetriever(BaseRetriever):
                _rag_helper: "MultiTurnRAGHelper" = PrivateAttr()
                _k: int = PrivateAttr()
                _threshold: float = PrivateAttr()

                def __init__(self, rag_helper: "MultiTurnRAGHelper", k: int, threshold: float):
                    super().__init__()
                    self._rag_helper = rag_helper
                    self._k = k
                    self._threshold = threshold

                def _get_relevant_documents(
                        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
                ) -> List[Document]:
                    return self._rag_helper.retrieve_documents(query, self._k, self._threshold)

                @property
                def _identifying_params(self):
                    return {"k": self._k, "threshold": self._threshold}

            retriever = ThresholdRetriever(self, k, similarity_threshold)
        else:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

        # 創建多輪對話的提示詞模板
        system_prompt = (
            "你是一個基於 RAG 系統的計算機概論家教。請參考提供的內容和之前的對話歷史來回答問題。\n"
            "使用如 b_k 下標 LaTeX 語法，數值前後要加上 '$'\n"
            "如果問題和計算機概論無關，不要回答問題。\n"
            "如果問題和計算機概論有關，用詞上多使用正向鼓勵詞語，並基於現有問題延伸出相關的問題，針對問題舉出簡單好懂的比喻或例子。\n"
            "使用 LaTeX 時，請使用 $$ 或 $ 符號作為塊級公式。\n"
            "請用繁體中文回答。\n"
            "相關文檔內容：\n{context}\n\n"
            "對話歷史：\n{chat_history}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        self.retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)

    def ask_with_memory(self, query: str, user_id: str):
        """帶有記憶的問答功能"""
        if not self.retrieval_chain:
            raise ValueError("請先執行 setup_retrieval_chain()")

        # 獲取用戶的對話記憶
        memory = self.get_or_create_memory(user_id)

        try:
            # 獲取對話歷史
            chat_history = memory.chat_memory.messages

            # 格式化對話歷史為字符串
            history_text = ""
            for message in chat_history[-self.memory_window * 2:]:  # 限制歷史長度
                if isinstance(message, HumanMessage):
                    history_text += f"使用者: {message.content}\n"
                elif isinstance(message, AIMessage):
                    history_text += f"LLM 回答: {message.content}\n"

            # 調用檢索鏈，傳入對話歷史
            result = self.retrieval_chain.invoke({
                "input": query,
                "chat_history": history_text
            })

            answer = result["answer"]
            sources = result["context"]

            # 將當前對話存入記憶
            memory.chat_memory.add_user_message(query)
            memory.chat_memory.add_ai_message(answer)

            return answer, sources

        except Exception as e:
            if "max_tokens_per_request" in str(e):
                print("內容過長，嘗試使用較短的上下文...")
                # 可以實現降級策略
                return self.ask_fallback(query, user_id)
            else:
                raise e

    def ask_fallback(self, query: str, user_id: str):
        """降級策略：使用較短上下文"""
        # 簡化的檢索和回答邏輯
        docs = self.retrieve_documents(query, k=3, similarity_threshold=0.7)

        llm = ChatOpenAI(model="gpt-5-nano")

        context = "\n\n".join([doc.page_content[:200] for doc in docs])

        simple_prompt = f"""
        基於以下內容回答問題，如果內容不相關請說明：

        內容：{context}

        問題：{query}

        請用繁體中文簡潔回答。
        """

        response = llm.invoke([{"role": "user", "content": simple_prompt}])
        return response.content, docs

    def clear_memory(self, user_id: str):
        """清除指定用戶的對話記憶"""
        if user_id in self.conversation_memory:
            self.conversation_memory[user_id].clear()
            return True
        return False

    def get_conversation_history(self, user_id: str) -> List[Dict[str, str]]:
        """獲取用戶的對話歷史"""
        if user_id not in self.conversation_memory:
            return []

        memory = self.conversation_memory[user_id]
        history = []

        for message in memory.chat_memory.messages:
            if isinstance(message, HumanMessage):
                history.append({"type": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                history.append({"type": "assistant", "content": message.content})

        return history

    # 原有的 ask 方法保留向後兼容性
    def ask(self, query):
        """原有的問答方法（無記憶功能）"""
        if not self.retrieval_chain:
            raise ValueError("請先執行 setup_retrieval_chain()")
        try:
            result = self.retrieval_chain.invoke({
                "input": query,
                "chat_history": ""  # 空的對話歷史
            })
            return result["answer"], result["context"]
        except Exception as e:
            if "max_tokens_per_request" in str(e):
                return self.ask_fallback(query, "default")
            else:
                raise e
