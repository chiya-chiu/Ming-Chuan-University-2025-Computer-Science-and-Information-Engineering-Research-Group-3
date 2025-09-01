import glob, os, uuid
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain, create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pydantic import PrivateAttr
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
)

from Split_Helper import SplitHelper


class RAGHelper:
    def __init__(self, pdf_folder, chunk_size=300, chunk_overlap=50, pdf_target_len=500,
                 pdf_tolerance=100):
        self.pdf_folder = pdf_folder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.pdf_target_len = pdf_target_len
        self.pdf_tolerance = pdf_tolerance
        self.vectorstore = None
        self.retrieval_chain = None

        self.splitter_instance = SplitHelper()
        self.splitter_instance.CENTER_ONLY = False
        self.splitter_instance.NUMBERED_HEADERS_ONLY = False
        self.splitter_instance.SMART_CONSOLIDATE = True

    async def ensure_vectorstore(self, file_extensions=None):
        if not self.vectorstore:
            await self.load_and_prepare(file_extensions)

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
            metadata = doc.metadata.copy() if hasattr(doc, 'metadata') and doc.metadata else {}
            metadata['id'] = f"doc_{i}_{uuid.uuid4().hex[:8]}"
            formatted_docs.append(LangchainDocument(
                page_content=doc.page_content if hasattr(doc, 'page_content') else str(doc),
                metadata=metadata
            ))

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = FAISS.from_documents(formatted_docs, embeddings)

    async def load_and_prepare(self, file_extensions=None):
        if os.path.exists("my_faiss_index"):
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
                if Path(path).suffix.lower() == ".pdf":
                    docs = self.splitter_instance.chunk_pdf_full_page(
                        pdf_path=path,
                        target_len=self.pdf_target_len,
                        tol=self.pdf_tolerance
                    )
                    all_chunks.extend(docs)
                else:
                    pages = await self.load_any_file_async(path)
                    chunks = self._split_documents(pages)
                    all_chunks.extend(chunks)

        self._build_vectorstore(all_chunks)
        self.vectorstore.save_local("my_faiss_index")

    async def setup_retrieval_chain(self, k=5, similarity_threshold=None, auto_prepare=True, file_extensions=None):
        if auto_prepare and not self.vectorstore:
            await self.ensure_vectorstore(file_extensions)

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

                def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
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
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        self.retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)

    async def ask(self, query, auto_setup=True):
        if auto_setup and not self.retrieval_chain:
            await self.setup_retrieval_chain()

        if not self.retrieval_chain:
            raise ValueError("檢索鏈未建立")

        result = self.retrieval_chain.invoke({"input": query})
        return result["answer"], result["context"]
