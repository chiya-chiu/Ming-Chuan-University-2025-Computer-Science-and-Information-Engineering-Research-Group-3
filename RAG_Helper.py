# import asyncio
import glob  # 用來找多個檔案
import os
import uuid
from pathlib import Path
# langchain 相關套件
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 切割文字
from langchain_community.vectorstores import FAISS  # FAISS : Facebook 開發的向量資料庫，用來做快速相似度搜尋。
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # embeddings 用來將文字轉換成向量
from langchain.chains import create_retrieval_chain  # 建立 RAG 架構中的「檢索＋問答」流程。
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pydantic import PrivateAttr
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from typing import List

# 可以讀取不同的檔案格式
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
)

from Split_Helper import SplitHelper


class RAGHelper:
    def __init__(self, pdf_folder, chunk_size=300, chunk_overlap=50, pdf_target_len=500,
                 pdf_tolerance=100):  # __init__ 是 python 的建構子
        self.pdf_folder = pdf_folder  # 儲存 PDF 檔案的 PATH
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.pdf_target_len = pdf_target_len
        self.pdf_tolerance = pdf_tolerance
        self.vectorstore = None
        self.retrieval_chain = None

        self.splitter_instance = SplitHelper()
        self.splitter_instance.CENTER_ONLY = False  # 不要求標題居中
        self.splitter_instance.NUMBERED_HEADERS_ONLY = False  # 不要求標題有編號
        self.splitter_instance.SMART_CONSOLIDATE = True  # 啟用智能合併

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
        # 有些 loader 是 async 的，有些不是
        if hasattr(loader, "alazy_load"):
            pages = []
            async for page in loader.alazy_load():
                pages.append(page)
            return pages
        else:
            return loader.load()  # 同步方式載入

    # 切割檔案
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
            # 添加唯一 ID 到元數據
            metadata = doc.metadata.copy() if hasattr(doc, 'metadata') and doc.metadata else {}
            metadata['id'] = f"doc_{i}_{uuid.uuid4().hex[:8]}"

            # 轉換為 Langchain Document
            formatted_doc = LangchainDocument(
                page_content=doc.page_content if hasattr(doc, 'page_content') else str(doc),
                metadata=metadata
            )
            formatted_docs.append(formatted_doc)

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = FAISS.from_documents(formatted_docs, embeddings)

    def retrieve_documents(self, query, k=5, similarity_threshold=0.7):
        """
        檢索相關文件，並根據相似度門檻過濾結果

        Args:
            query (str): 查詢字串
            k (int): 檢索數量
            similarity_threshold (float): 相似度門檻 (0.0-1.0)，低於此值的結果會被過濾

        Returns:
            list: 過濾後的文件列表
        """
        if not self.vectorstore:
            print("❌ 向量資料庫未初始化，請先執行 load_and_prepare()")
            return []

        try:
            # 使用 similarity_search_with_score 獲取帶有分數的結果
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                query, k=k
            )

            # 根據相似度門檻過濾
            filtered_docs = []
            #print(f"🔍 檢索到 {len(docs_with_scores)} 個結果，應用相似度門檻 {similarity_threshold}")

            for doc, score in docs_with_scores:
                # FAISS 使用歐幾里得距離，分數越低表示越相似
                # 轉換為相似度百分比（可選）
                # similarity = 1 / (1 + score)  # 轉換公式，讓分數越高表示越相似

                #print(f"📄 距離分數: {score:.4f}")
                #print(f"   文件預覽: {doc.page_content[:100]}...")

                # 直接使用距離分數進行比較（分數越低越相似）
                if score <= similarity_threshold:
                    filtered_docs.append(doc)
                    #print(f"   ✅ 通過門檻，保留此文件")
                #else:
                    #print(f"   ❌ 高於門檻 {similarity_threshold}，過濾此文件")

            #print(f"📊 過濾結果：{len(docs_with_scores)} -> {len(filtered_docs)} 個文件")
            return filtered_docs

        except Exception as e:
            print(f"❌ 檢索過程中發生錯誤: {e}")
            return []

    def get_retriever_with_threshold(self, k=5, similarity_threshold=0.7):
        """
        創建一個帶有相似度門檻的自定義檢索器

        Args:
            k (int): 檢索數量
            similarity_threshold (float): 相似度門檻

        Returns:
            callable: 檢索函數
        """

        def custom_retriever(query):
            return self.retrieve_documents(query, k, similarity_threshold)

        return custom_retriever

    async def load_and_prepare(self, file_extensions=None):
        print("開始載入檔案...")

        if os.path.exists("my_faiss_index"):  # 如果本地有向量資料庫，載入本地的向量資料庫
            print("已偵測到現有向量資料庫，直接載入...")
            self.vectorstore = FAISS.load_local(
                "my_faiss_index",
                OpenAIEmbeddings(model="text-embedding-3-small"),
                allow_dangerous_deserialization=True
            )

        else:

            """
            載入並準備文件
            file_extensions: 要載入的檔案副檔名列表，例如 ['.pdf', '.txt', '.docx']
            如果為 None，則只載入 PDF 檔案（保持原有行為）
            """
            print("正在建立和讀取向量資料庫")

            if file_extensions is None:
                file_extensions = ['.pdf']  # 預設只載入 PDF

            all_chunks = []

            # 根據指定的副檔名載入檔案
            for ext in file_extensions:
                pattern = f"*{ext}"
                file_paths = glob.glob(os.path.join(self.pdf_folder, pattern))

                for path in file_paths:
                    try:
                        fname = os.path.basename(path)
                        print(f"讀取中: {fname}")

                        if Path(path).suffix.lower() == ".pdf":
                            # ★ PDF 使用智慧切割（標題偵測 / 頁眉頁腳過濾 / 細切 + 智慧合併）
                            docs = self.splitter_instance.chunk_pdf_full_page(
                                pdf_path=path,
                                target_len=self.pdf_target_len,
                                tol=self.pdf_tolerance
                            )
                            all_chunks.extend(docs)
                            print(f" {fname}（PDF 智慧切）完成，共 {len(docs)} 段")
                        else:
                            # 其它副檔名維持原本流程
                            pages = await self.load_any_file_async(path)
                            chunks = self._split_documents(pages)
                            all_chunks.extend(chunks)
                            print(f" {fname} 分割完成，共 {len(chunks)} 段")

                    except Exception as e:
                        print(f"載入 {os.path.basename(path)} 時發生錯誤: {e}")

            print(f"所有檔案段落總數：{len(all_chunks)}")

            if len(all_chunks) == 0:
                raise ValueError("沒有成功載入任何文件")

            self._build_vectorstore(all_chunks)  # 將文字轉成向量，並建立向量資料庫
            self.vectorstore.save_local("my_faiss_index")  # 將向量資料庫存到本地

    def setup_retrieval_chain(self, k=5, similarity_threshold=None):
        """
        設置檢索鏈

        Args:
            k (int): 檢索數量
            similarity_threshold (float, optional): 相似度門檻，如果提供則使用過濾檢索器
        """
        if not self.vectorstore:
            raise ValueError("請先執行 load_and_prepare()")

        llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

        # 根據是否有相似度門檻選擇不同的檢索器
        if similarity_threshold is not None:
            # 使用帶有相似度門檻的檢索器
            from langchain.schema import BaseRetriever
            from langchain.callbacks.manager import CallbackManagerForRetrieverRun
            from typing import List
            from langchain.schema import Document

            class ThresholdRetriever(BaseRetriever):
                _rag_helper: "RAGHelper" = PrivateAttr()
                _k: int = PrivateAttr()
                _threshold: float = PrivateAttr()

                def __init__(self, rag_helper: "RAGHelper", k: int, threshold: float):
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
            #print(f"🎯 使用相似度門檻檢索器 (k={k}, threshold={similarity_threshold})")
        else:
            # 使用標準檢索器
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": k}
            )
            #print(f"📋 使用標準檢索器 (k={k})")

        # 創建提示詞模板
        system_prompt = (
            "你是一個基於 RAG 系統的計算機概論家教。請參考以下提供的內容來回答問題。"
            "用詞上請多使用正向鼓勵的詞語，並基於現有問題延伸出更多相關的問題。"
            "請針對問題舉出簡單好懂的比喻或例子。"
            "如果不知道如何回答問題，請說出來。"
            "如果問題和計算機概論無關，請做出提醒，並且不要回答問題"
            "使用 LaTeX 時，請使用 $ 符號作為塊級公式"
            "請用繁體中文回答。\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        # 創建文檔合併鏈
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        # 創建檢索鏈
        self.retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)

    def ask(self, query):
        if not self.retrieval_chain:
            raise ValueError("請先執行 setup_retrieval_chain()")
        try:
            result = self.retrieval_chain.invoke({"input": query})  # 將使用者的問題傳給問答鏈，鏈內部會檢索並將檢索到的段落和問題交給大語言模型
            return result["answer"], result["context"]  # result["answer"] 是 語言模型給的答案，result["context"]  是檢索到的原始段落
        except Exception as e:
            if "max_tokens_per_request" in str(e):
                print("內容過長，嘗試使用較短的上下文...")
                self.setup_retrieval_chain_with_shorter_context()
                result = self.retrieval_chain.invoke({"input": query})
                return result["answer"], result["context"]
            else:
                raise e

    def setup_retrieval_chain_with_shorter_context(self):
        """設置更短上下文的檢索鏈"""
        if not self.vectorstore:
            raise ValueError("請先執行 load_and_prepare()")

        llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
        # 更嚴格的檢索配置
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