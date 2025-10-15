#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統一管理所有LLM配置的「控制中心」       

LLM 統一配置管理模組

此模組統一管理專案中所有 LLM 相關配置，包括：
- 問答系統的 LLM 模型
- RAG 預處理的 Embedding 模型
- 圖表描述生成的 LLM 模型

所有 LLM 調用都應該透過此模組取得配置，確保一致性和可維護性。
"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from enum import Enum

# 載入環境變數
load_dotenv()


class LLMProvider(Enum):
    """LLM 提供者枚舉"""
    OPENAI = "openai"
    MOCK = "mock"
    LOCAL = "local"


class LLMConfig:
    """LLM 統一配置類別"""

    # ==================== API 配置 ====================
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

    # ==================== 問答系統 LLM 配置 ====================
    # 用於 main_web.py 的對話問答
    QA_MODEL_NAME = os.getenv("QA_MODEL_NAME", "gpt-5-nano")
    QA_TEMPERATURE = float(os.getenv("QA_TEMPERATURE", "0.1"))
    QA_MAX_TOKENS = int(os.getenv("QA_MAX_TOKENS", "2000"))

    # ==================== Embedding 模型配置 ====================
    # 用於向量化文字（RAG_Helper、MultiTurnRAGHelper）
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-large")

    # ==================== 圖表描述生成 LLM 配置 ====================
    # 用於 ABC 階段的圖表描述生成
    CHART_DESCRIPTION_MODEL = os.getenv("CHART_DESCRIPTION_MODEL", "gpt-3.5-turbo")
    CHART_DESCRIPTION_TEMPERATURE = float(os.getenv("CHART_DESCRIPTION_TEMPERATURE", "0.3"))
    CHART_DESCRIPTION_MAX_TOKENS = int(os.getenv("CHART_DESCRIPTION_MAX_TOKENS", "300"))

    # 圖表描述的提供者（openai/mock/auto）
    # ⚠️ 測試時請設定為 "mock"，正式使用時改為 "openai"
    CHART_DESCRIPTION_PROVIDER = os.getenv("CHART_DESCRIPTION_PROVIDER", "openai")  # 👈 使用真實 LLM

    # ==================== RAG 檢索配置 ====================
    # 檢索數量
    RAG_RETRIEVAL_K = int(os.getenv("RAG_RETRIEVAL_K", "5"))
    # 相似度門檻
    RAG_SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.45"))

    # ==================== 文字切割配置 ====================
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
    PDF_TARGET_LEN = int(os.getenv("PDF_TARGET_LEN", "500"))
    PDF_TOLERANCE = int(os.getenv("PDF_TOLERANCE", "100"))

    # ==================== 多輪對話配置 ====================
    CONVERSATION_MEMORY_WINDOW = int(os.getenv("CONVERSATION_MEMORY_WINDOW", "10"))
    MAX_CONVERSATIONS = int(os.getenv("MAX_CONVERSATIONS", "100"))

    @classmethod
    def get_qa_llm_params(cls) -> Dict[str, Any]:
        """
        取得問答系統的 LLM 參數

        Returns:
            dict: 包含 model, temperature, max_tokens 的字典
        """
        return {
            "model": cls.QA_MODEL_NAME,
            "temperature": cls.QA_TEMPERATURE,
            "max_tokens": cls.QA_MAX_TOKENS
        }

    @classmethod
    def get_embedding_params(cls) -> Dict[str, Any]:
        """
        取得 Embedding 模型參數

        Returns:
            dict: 包含 model 的字典
        """
        return {
            "model": cls.EMBEDDING_MODEL_NAME
        }

    @classmethod
    def get_chart_description_params(cls) -> Dict[str, Any]:
        """
        取得圖表描述生成的 LLM 參數

        Returns:
            dict: 包含 model, temperature, max_tokens 的字典
        """
        return {
            "model": cls.CHART_DESCRIPTION_MODEL,
            "temperature": cls.CHART_DESCRIPTION_TEMPERATURE,
            "max_tokens": cls.CHART_DESCRIPTION_MAX_TOKENS
        }

    @classmethod
    def get_rag_retrieval_params(cls) -> Dict[str, Any]:
        """
        取得 RAG 檢索參數

        Returns:
            dict: 包含 k, similarity_threshold 的字典
        """
        return {
            "k": cls.RAG_RETRIEVAL_K,
            "similarity_threshold": cls.RAG_SIMILARITY_THRESHOLD
        }

    @classmethod
    def get_chunk_params(cls) -> Dict[str, Any]:
        """
        取得文字切割參數

        Returns:
            dict: 包含 chunk_size, chunk_overlap, pdf_target_len, pdf_tolerance 的字典
        """
        return {
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
            "pdf_target_len": cls.PDF_TARGET_LEN,
            "pdf_tolerance": cls.PDF_TOLERANCE
        }

    @classmethod
    def get_conversation_params(cls) -> Dict[str, Any]:
        """
        取得多輪對話參數

        Returns:
            dict: 包含 memory_window, max_conversations 的字典
        """
        return {
            "memory_window": cls.CONVERSATION_MEMORY_WINDOW,
            "max_conversations": cls.MAX_CONVERSATIONS
        }

    @classmethod
    def validate_config(cls) -> Dict[str, bool]:
        """
        驗證配置是否完整

        Returns:
            dict: 各項配置的驗證結果
        """
        return {
            "openai_api_key_set": bool(cls.OPENAI_API_KEY),
            "openai_base_url_set": bool(cls.OPENAI_BASE_URL),
            "qa_model_configured": bool(cls.QA_MODEL_NAME),
            "embedding_model_configured": bool(cls.EMBEDDING_MODEL_NAME),
            "chart_model_configured": bool(cls.CHART_DESCRIPTION_MODEL),
        }

    @classmethod
    def print_current_config(cls):
        """印出當前配置（用於除錯）"""
        print("=" * 60)
        print("LLM 配置資訊")
        print("=" * 60)
        print(f"OpenAI API Key: {'已設定' if cls.OPENAI_API_KEY else '❌ 未設定'}")
        print(f"OpenAI Base URL: {cls.OPENAI_BASE_URL}")
        print()
        print("【問答系統】")
        print(f"  模型: {cls.QA_MODEL_NAME}")
        print(f"  溫度: {cls.QA_TEMPERATURE}")
        print(f"  最大 Token: {cls.QA_MAX_TOKENS}")
        print()
        print("【Embedding】")
        print(f"  模型: {cls.EMBEDDING_MODEL_NAME}")
        print()
        print("【圖表描述】")
        print(f"  模型: {cls.CHART_DESCRIPTION_MODEL}")
        print(f"  提供者: {cls.CHART_DESCRIPTION_PROVIDER}")
        print(f"  溫度: {cls.CHART_DESCRIPTION_TEMPERATURE}")
        print(f"  最大 Token: {cls.CHART_DESCRIPTION_MAX_TOKENS}")
        print()
        print("【RAG 檢索】")
        print(f"  檢索數量 (k): {cls.RAG_RETRIEVAL_K}")
        print(f"  相似度門檻: {cls.RAG_SIMILARITY_THRESHOLD}")
        print()
        print("【文字切割】")
        print(f"  區塊大小: {cls.CHUNK_SIZE}")
        print(f"  重疊大小: {cls.CHUNK_OVERLAP}")
        print(f"  PDF 目標長度: {cls.PDF_TARGET_LEN}")
        print(f"  PDF 容許誤差: {cls.PDF_TOLERANCE}")
        print("=" * 60)


# 便捷函數
def get_openai_client():
    """
    取得已配置的 OpenAI Client

    Returns:
        OpenAI: OpenAI 客戶端實例
    """
    from openai import OpenAI

    return OpenAI(
        api_key=LLMConfig.OPENAI_API_KEY,
        base_url=LLMConfig.OPENAI_BASE_URL
    )


def get_openai_embeddings():
    """
    取得已配置的 OpenAI Embeddings

    Returns:
        OpenAIEmbeddings: LangChain 的 OpenAI Embeddings 實例
    """
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=LLMConfig.EMBEDDING_MODEL_NAME,
        openai_api_key=LLMConfig.OPENAI_API_KEY,
        openai_api_base=LLMConfig.OPENAI_BASE_URL
    )


def get_chat_llm(model: Optional[str] = None, temperature: Optional[float] = None):
    """
    取得已配置的 ChatOpenAI 實例

    Args:
        model: 可選的模型名稱，若不指定則使用配置的 QA_MODEL_NAME
        temperature: 可選的溫度參數，若不指定則使用配置的 QA_TEMPERATURE

    Returns:
        ChatOpenAI: LangChain 的 ChatOpenAI 實例
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model or LLMConfig.QA_MODEL_NAME,
        temperature=temperature if temperature is not None else LLMConfig.QA_TEMPERATURE,
        openai_api_key=LLMConfig.OPENAI_API_KEY,
        openai_api_base=LLMConfig.OPENAI_BASE_URL
    )


# 測試用主程式
if __name__ == "__main__":
    print("測試 LLM 配置模組\n")

    # 印出配置
    LLMConfig.print_current_config()

    # 驗證配置
    print("\n配置驗證結果:")
    validation = LLMConfig.validate_config()
    for key, value in validation.items():
        status = "✅" if value else "❌"
        print(f"  {status} {key}")
