import os
import asyncio
import hashlib
import jwt
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
from RAG_Helper import RAGHelper
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
import uuid
from contextlib import asynccontextmanager
import glob
import re
import json
from pathlib import Path

# 載入 .env 檔案
load_dotenv()

# 安全設定
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# 資料庫連線設定
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("請在 .env 檔案中設定 DATABASE_URL")

def get_db_connection():
    """取得 PostgreSQL 資料庫連線"""
    try:
        if "render.com" in DATABASE_URL or "amazonaws.com" in DATABASE_URL:
            conn = psycopg2.connect(
                DATABASE_URL,
                cursor_factory=RealDictCursor,
                sslmode='require'
            )
        else:
            conn = psycopg2.connect(
                DATABASE_URL,
                cursor_factory=RealDictCursor,
                sslmode='disable'
            )
        return conn
    except Exception as e:
        print(f"資料庫連線失敗: {e}")
        raise e

def test_db_connection():
    """測試資料庫連線"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"✅ 資料庫連線成功: {version['version']}")
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ 資料庫連線失敗: {e}")
        return False

def init_database():
    """初始化 PostgreSQL 資料庫（移除 email 欄位）"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 簡化的使用者表結構
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(255) UNIQUE NOT NULL,
            username VARCHAR(255) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE,
            is_admin BOOLEAN DEFAULT FALSE
        )
    ''')

    # 問答紀錄表保持不變
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS questions_log (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            sources_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            response_time REAL,
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        )
    ''')

    conn.commit()
    cursor.close()
    conn.close()

# 應用程式生命週期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔧 初始化資料庫...")
    test_db_connection()
    init_database()
    print("✅ 資料庫初始化完成")

    global rag_instance
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY 尚未設定")
    elif not os.path.exists("./pdfFiles"):
        print("⚠️ pdfFiles 資料夾不存在")
    else:
        print("🚀 初始化 RAG 系統...")
        rag_instance = RAGHelper(pdf_folder="./pdfFiles", chunk_size=300, chunk_overlap=50)
        await rag_instance.load_and_prepare(['.pdf', '.txt', '.docx', '.md', '.csv'])
        rag_instance.setup_retrieval_chain(k=5, similarity_threshold=0.45)
        print("✅ RAG 系統已就緒")

    yield

# 全域 RAG 實例
rag_instance: Optional[RAGHelper] = None
security = HTTPBearer()

# 建立 FastAPI 應用
app = FastAPI(lifespan=lifespan)

# 掛載靜態資料夾
app.mount("/static", StaticFiles(directory="static"), name="static")

# ===== 資料模型 =====
class UserRegister(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user_info: dict

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: List[dict]

class StatusResponse(BaseModel):
    status: str
    message: str

class UserStats(BaseModel):
    total_questions: int
    questions_today: int
    avg_response_time: float

class ChatHistoryItem(BaseModel):
    question: str
    answer: str
    timestamp: str
    response_time: float

class ChatHistoryResponse(BaseModel):
    history: List[ChatHistoryItem]
    total_count: int

class ImageResponse(BaseModel):
    image_url: str
    image_name: str
    message: str

# ===== 工具函數 =====
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    return hash_password(password) == hashed

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")

def get_user_from_db(user_id: str = None, username: str = None):
    conn = get_db_connection()
    cursor = conn.cursor()
    if user_id:
        cursor.execute("SELECT * FROM users WHERE user_id = %s", (user_id,))
    elif username:
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
    else:
        cursor.close()
        conn.close()
        return None
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    return user

def log_question(user_id: str, question: str, answer: str, sources_count: int, response_time: float):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO questions_log (user_id, question, answer, sources_count, response_time)
        VALUES (%s, %s, %s, %s, %s)
    ''', (user_id, question, answer, sources_count, response_time))
    conn.commit()
    cursor.close()
    conn.close()

def verify_admin(user_id: str):
    user = get_user_from_db(user_id=user_id)
    if not user or not user['is_admin']:
        raise HTTPException(status_code=403, detail="您沒有管理員權限")

def is_chart_relevant(chart_info, source_content):
    description = chart_info.get('generated_description', '').lower()
    caption = chart_info.get('original_caption', '').lower()
    description_keywords = [word for word in description.split() if len(word) > 3][:10]
    caption_keywords = [word for word in caption.split() if len(word) >= 5]
    return (any(keyword in source_content for keyword in description_keywords) or
            any(keyword in source_content for keyword in caption_keywords))

# ===== 這裡開始是你的所有 API 路由 =====
# (保留你原本的 /register, /login, /ask, /stats, /admin/stats, /status, /chat/history, /chat/history 清除, /test/image, /test/image-list, /health, /debug/... 所有端點)

# 範例：
@app.get("/")
async def root():
    return {"message": "RAG 系統首頁，請使用前端介面"}

# 其餘路由保持原本的程式碼不變

# ===== 主程式 =====
if __name__ == "__main__":
    import uvicorn
    print("🚀 啟動 RAG 網站服務（簡化版）...")
    print("📱 網站網址：http://localhost:8080")
    print("📚 API 文件：http://localhost:8080/docs")
    print("📁 請確保 pdfFiles 資料夾中有要處理的檔案")
    print("🔑 請在 .env 檔案中設定 SECRET_KEY、OPENAI_API_KEY 和 DATABASE_URL")
    uvicorn.run(app, host="0.0.0.0", port=8080)
