import glob, os, uuid
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
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

    def retrieve_documents(self, query: str, k: int, threshold: float) -> List[Document]:
        """添加這個方法來支持相似度門檻檢索"""
        if not self.vectorstore:
            return []
        
        # 使用相似度搜尋並過濾結果
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k*2)  # 取更多結果以便過濾
        filtered_docs = [doc for doc, score in docs_with_scores if score >= threshold]
        return filtered_docs[:k]  # 返回前k個結果

    async def load_and_prepare(self, file_extensions=None):
        if os.path.exists("my_faiss_index"):
            try:
                self.vectorstore = FAISS.load_local(
                    "my_faiss_index",
                    OpenAIEmbeddings(model="text-embedding-3-small"),
                    allow_dangerous_deserialization=True
                )
                return
            except Exception as e:
                print(f"載入現有索引失敗，將重新建立: {e}")

        if file_extensions is None:
            file_extensions = ['.pdf']

        all_chunks = []

        for ext in file_extensions:
            pattern = os.path.join(self.pdf_folder, f"*{ext}")
            for path in glob.glob(pattern):
                try:
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
                except Exception as e:
                    print(f"處理檔案 {path} 時發生錯誤: {e}")
                    continue

        if not all_chunks:
            raise ValueError(f"在 {self.pdf_folder} 中沒有找到有效的檔案")

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
                def __init__(self, rag_helper: "RAGHelper", k: int, threshold: float):
                    super().__init__()
                    self._rag_helper = rag_helper
                    self._k = k
                    self._threshold = threshold

                def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
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

        try:
            result = self.retrieval_chain.invoke({"input": query})
            return result["answer"], result["context"]
        except Exception as e:
            print(f"檢索時發生錯誤: {e}")
            raise


# 使用範例 - 正確的異步調用方式
async def example_usage():
    """示範如何正確使用 RAGHelper"""
    rag = RAGHelper(pdf_folder="./documents")
    
    # 方法 1: 分步驟調用
    await rag.ensure_vectorstore(['.pdf', '.txt'])
    await rag.setup_retrieval_chain(k=5)
    answer, context = await rag.ask("什麼是演算法？")
    print(f"回答: {answer}")
    
    # 方法 2: 一次性調用（推薦）
    answer, context = await rag.ask("什麼是資料結構？", auto_setup=True)
    print(f"回答: {answer}")


# 如果你需要在同步環境中使用，可以這樣包裝：
import asyncio

def sync_ask(rag_helper, query):
    """同步版本的詢問函數"""
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # 如果在 Jupyter notebook 或其他異步環境中
        import nest_asyncio
        nest_asyncio.apply()
    
    return asyncio.run(rag_helper.ask(query))


# 同步使用範例
def sync_example():
    rag = RAGHelper(pdf_folder="./documents")
    answer, context = sync_ask(rag, "什麼是資料庫？")
    print(f"回答: {answer}")
