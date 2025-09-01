import glob
import os
import uuid
from pathlib import Path
from typing import List

import nest_asyncio
nest_asyncio.apply()  # 避免 "event loop already running"

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from pydantic import PrivateAttr

from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
)

from Split_Helper import SplitHelper


class RAGHelper:
    def __init__(self, pdf_folder: str, chunk_size=300, chunk_overlap=50,
                 pdf_target_len=500, pdf_tolerance=100):
        self.pdf_folder = pdf_folder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.pdf_target_len = pdf_target_len
        self.pdf_tolerance = pdf_tolerance
        self.vectorstore: FAISS | None = None
        self.retrieval_chain = None

        self.splitter_instance = SplitHelper()
        self.splitter_instance.CENTER_ONLY = False
        self.splitter_instance.NUMBERED_HEADERS_ONLY = False
        self.splitter_instance.SMART_CONSOLIDATE = True

    # -------------------- 檔案載入與向量庫建立 -------------------- #
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
        from langchain.schema import Document as LangchainDocument

        formatted_docs = []
        for i, doc in enumerate(documents):
            metadata = getattr(doc, "metadata", {}) or {}
            metadata['id'] = f"doc_{i}_{uuid.uuid4().hex[:8]}"
            formatted_doc = LangchainDocument(
                page_content=getattr(doc, "page_content", str(doc)),
                metadata=metadata
            )
            formatted_docs.append(formatted_doc)

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = FAISS.from_documents(formatted_docs, embeddings)

    async def load_and_prepare(self, file_extensions=None):
        """非同步載入檔案並建立向量資料庫"""
        print("開始載入檔案...")

        if os.path.exists("my_faiss_index"):
            print("已偵測到現有向量資料庫，直接載入...")
            self.vectorstore = FAISS.load_local(
                "my_faiss_index",
                OpenAIEmbeddings(model="text-embedding-3-small"),
                allow_dangerous_deserialization=True
            )
            return

        if file_extensions is None:
            file_extensions = ['.pdf']

        all_chunks = []
        for ext in file_extensions:
            for path in glob.glob(os.path.join(self.pdf_folder, f"*{ext}")):
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
                    print(f"載入 {fname} 發生錯誤: {e}")

        if not all_chunks:
            raise ValueError("沒有成功載入任何文件")

        self._build_vectorstore(all_chunks)
        self.vectorstore.save_local("my_faiss_index")
        print(f"向量資料庫建立完成，共 {len(all_chunks)} 段")

    # -------------------- 檢索與問答 -------------------- #
    async def setup_retrieval_chain(self, k=5, similarity_threshold=None,
                                    auto_prepare=True, file_extensions=None):
        """非同步建立檢索鏈"""
        if auto_prepare and not self.vectorstore:
            await self.load_and_prepare(file_extensions)

        if not self.vectorstore:
            raise ValueError("向量資料庫建立失敗")

        llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

        if similarity_threshold is not None:
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
        else:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

        system_prompt = (
            "你是一個基於 RAG 系統的計算機概論家教。請參考提供的內容來回答問題。"
            "如果問題和計算機概論無關，不要回答問題，告訴使用者問計算機概論相關問題"
            "如果不知道如何回答問題或是問題沒意義，請提醒輸入更多資訊。"  
            "用詞上請多使用正向鼓勵的詞語，並基於現有問題延伸出更多相關的問題。"
            "請針對問題舉出簡單好懂的比喻或例子。"
            "使用 LaTeX 時，請使用 $ 符號作為塊級公式"
            "請用繁體中文回答。\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        self.retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)

    def retrieve_documents(self, query, k=5, similarity_threshold=0.7):
        """檢索相關文件並過濾"""
        if not self.vectorstore:
            print("❌ 向量資料庫未初始化")
            return []

        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
        filtered = [(doc, 1/(1+score)) for doc, score in docs_with_scores if 1/(1+score) >= similarity_threshold]
        for i, (doc, sim) in enumerate(filtered):
            doc.page_content = f"重要性第 {i+1}名 {doc.page_content}"
        return [doc for doc, _ in filtered]

    # -------------------- 非同步問答 -------------------- #
    async def ask(self, query, auto_setup=True):
        if auto_setup and not self.retrieval_chain:
            await self.setup_retrieval_chain()

        if not self.retrieval_chain:
            raise ValueError("檢索鏈未建立")

        result = self.retrieval_chain.invoke({"input": query})
        return result["answer"], result["context"]
