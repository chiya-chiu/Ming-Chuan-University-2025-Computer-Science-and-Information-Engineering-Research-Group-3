import os
import asyncio
import hashlib
import jwt
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional
from RAG_Helper import RAGHelper
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import psycopg2
from psycopg2.extras import RealDictCursor
import uuid
from contextlib import asynccontextmanager
import glob
import re


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

# 資料庫初始化 - 修改為學習追蹤系統
def init_database():
    """初始化 PostgreSQL 資料庫 - 學習追蹤版本"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 學習者資料表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS learners (
            id SERIAL PRIMARY KEY,
            username VARCHAR(255) UNIQUE NOT NULL,     -- 學習者姓名
            user_id VARCHAR(255) UNIQUE NOT NULL,       -- 學習者唯一 ID
            password_hash VARCHAR(255) NOT NULL,        -- 密碼雜湊值
            last_visit_time TIMESTAMP,                  -- 最後點擊學習網站的時間
            today_visit_count INTEGER DEFAULT 0,        -- 當天點擊學習網站的次數
            last_count_reset_date DATE,                 -- 最後重置計數的日期
            total_questions INTEGER DEFAULT 0,          -- 總詢問問題數
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE
        )
    ''')

    # 學習問題記錄表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS learning_questions (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,              -- 學習者 ID
            question TEXT NOT NULL,                     -- 詢問的問題內容
            answer TEXT NOT NULL,                       -- 系統回答內容
            sources_count INTEGER DEFAULT 0,            -- 參考來源數量
            asked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- 詢問時間
            response_time REAL DEFAULT 0,               -- 回應時間
            FOREIGN KEY(user_id) REFERENCES learners(user_id)
        )
    ''')

    # 學習網站存取記錄表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS visit_logs (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,              -- 學習者 ID
            visit_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- 造訪時間
            action_type VARCHAR(50) DEFAULT 'visit',    -- 動作類型 (visit, question, etc.)
            FOREIGN KEY(user_id) REFERENCES learners(user_id)
        )
    ''')

    conn.commit()
    cursor.close()
    conn.close()

# 應用程式生命週期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔧 初始化學習追蹤資料庫...")
    test_db_connection()
    init_database()
    print("✅ 學習追蹤資料庫初始化完成")
    yield

app = FastAPI(
    title="學習追蹤 RAG 問答系統",
    description="追蹤學習者行為的智能問答系統",
    lifespan=lifespan
)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全域 RAG 實例
rag_instance: Optional[RAGHelper] = None

# 安全相關
security = HTTPBearer()

# 資料模型 - 修改為學習追蹤版本
class LearnerRegister(BaseModel):
    user_name: str      # 改為 user_name
    password: str

class LearnerLogin(BaseModel):
    user_name: str      # 改為 user_name
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

class LearnerStats(BaseModel):
    user_name: str
    total_questions: int
    questions_today: int
    today_visit_count: int
    last_visit_time: Optional[str]
    avg_response_time: float

class LearningHistoryItem(BaseModel):
    question: str
    answer: str
    timestamp: str
    response_time: float

class LearningHistoryResponse(BaseModel):
    history: List[LearningHistoryItem]
    total_count: int

class ImageResponse(BaseModel):
    image_url: str
    image_name: str
    message: str

# 工具函數
def hash_password(password: str) -> str:
    """密碼雜湊"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    """驗證密碼"""
    return hash_password(password) == hashed

def create_access_token(data: dict):
    """建立 JWT token"""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """取得目前使用者"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

def get_learner_from_db(user_id: str = None, user_name: str = None):
    """從資料庫取得學習者資料"""
    conn = get_db_connection()
    cursor = conn.cursor()

    if user_id:
        cursor.execute("SELECT * FROM learners WHERE user_id = %s", (user_id,))
    elif user_name:
        cursor.execute("SELECT * FROM learners WHERE user_name = %s", (user_name,))
    else:
        cursor.close()
        conn.close()
        return None

    learner = cursor.fetchone()
    cursor.close()
    conn.close()
    return learner

def update_visit_count(user_id: str):
    """更新造訪計數和最後造訪時間"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    now = datetime.now()
    today = now.date()
    
    # 獲取使用者目前資料
    cursor.execute("SELECT today_visit_count, last_count_reset_date FROM learners WHERE user_id = %s", (user_id,))
    result = cursor.fetchone()
    
    if result:
        current_count = result['today_visit_count'] or 0
        last_reset_date = result['last_count_reset_date']
        
        # 如果是新的一天，重置計數
        if not last_reset_date or last_reset_date != today:
            new_count = 1
            cursor.execute('''
                UPDATE learners 
                SET today_visit_count = %s, 
                    last_visit_time = %s, 
                    last_count_reset_date = %s 
                WHERE user_id = %s
            ''', (new_count, now, today, user_id))
        else:
            # 同一天，增加計數
            new_count = current_count + 1
            cursor.execute('''
                UPDATE learners 
                SET today_visit_count = %s, 
                    last_visit_time = %s 
                WHERE user_id = %s
            ''', (new_count, now, user_id))
    
    # 記錄造訪日誌
    cursor.execute('''
        INSERT INTO visit_logs (user_id, visit_time, action_type)
        VALUES (%s, %s, 'visit')
    ''', (user_id, now))
    
    conn.commit()
    cursor.close()
    conn.close()

def log_learning_question(user_id: str, question: str, answer: str, sources_count: int, response_time: float):
    """記錄學習問題到資料庫"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 記錄問題
    cursor.execute('''
        INSERT INTO learning_questions (user_id, question, answer, sources_count, response_time)
        VALUES (%s, %s, %s, %s, %s)
    ''', (user_id, question, answer, sources_count, response_time))
    
    # 更新總問題數
    cursor.execute('''
        UPDATE learners 
        SET total_questions = total_questions + 1 
        WHERE user_id = %s
    ''', (user_id,))
    
    # 記錄問題動作日誌
    cursor.execute('''
        INSERT INTO visit_logs (user_id, action_type)
        VALUES (%s, 'question')
    ''', (user_id,))
    
    conn.commit()
    cursor.close()
    conn.close()

# API 端點

@app.get("/", response_class=FileResponse)
async def serve_index():
    """顯示首頁"""
    return FileResponse("static/index.html")

@app.post("/register")
async def register_learner(learner: LearnerRegister):
    """學習者註冊"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 檢查學習者是否已存在
    cursor.execute("SELECT * FROM learners WHERE user_name = %s", (learner.user_name,))
    if cursor.fetchone():
        cursor.close()
        conn.close()
        raise HTTPException(status_code=400, detail="學習者姓名已存在")

    # 建立新學習者
    user_id = str(uuid.uuid4())
    password_hash = hash_password(learner.password)

    cursor.execute('''
        INSERT INTO learners (user_id, user_name, password_hash, last_count_reset_date)
        VALUES (%s, %s, %s, %s)
    ''', (user_id, learner.user_name, password_hash, datetime.now().date()))

    conn.commit()
    cursor.close()
    conn.close()

    return {"message": "學習者註冊成功", "user_id": user_id}

@app.post("/login", response_model=Token)
async def login_learner(learner: LearnerLogin):
    """學習者登入"""
    db_learner = get_learner_from_db(user_name=learner.user_name)

    if not db_learner or not verify_password(learner.password, db_learner['password_hash']):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="姓名或密碼錯誤"
        )

    # 更新造訪記錄
    update_visit_count(db_learner['user_id'])

    # 建立 JWT token
    access_token = create_access_token(data={"sub": db_learner['user_id']})

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_info": {
            "user_id": db_learner['user_id'],
            "user_name": db_learner['user_name'],
            "today_visit_count": db_learner['today_visit_count'] or 0,
            "total_questions": db_learner['total_questions'] or 0
        }
    }

@app.get("/me")
async def get_current_learner_info(current_user: str = Depends(get_current_user)):
    """取得目前登入學習者的資訊"""
    # 更新造訪記錄
    update_visit_count(current_user)
    
    db_learner = get_learner_from_db(user_id=current_user)
    if not db_learner:
        raise HTTPException(status_code=404, detail="學習者不存在")

    return {
        "user_id": db_learner['user_id'],
        "user_name": db_learner['user_name'],
        "last_visit_time": str(db_learner['last_visit_time']) if db_learner['last_visit_time'] else None,
        "today_visit_count": db_learner['today_visit_count'] or 0,
        "total_questions": db_learner['total_questions'] or 0,
        "created_at": str(db_learner['created_at']),
        "is_active": db_learner['is_active']
    }

@app.post("/initialize")
async def initialize_system(current_user: str = Depends(get_current_user)):
    """初始化 RAG 系統"""
    global rag_instance

    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="請在 .env 檔案中設定 OPENAI_API_KEY")

    if not os.path.exists("./pdfFiles"):
        raise HTTPException(status_code=500, detail="找不到 pdfFiles 資料夾")

    try:
        rag_instance = RAGHelper(pdf_folder="./pdfFiles", chunk_size=300, chunk_overlap=50)
        await rag_instance.load_and_prepare(['.pdf', '.txt', '.docx', '.md', '.csv'])
        rag_instance.setup_retrieval_chain()

        return StatusResponse(
            status="success",
            message="學習系統初始化完成"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"學習系統初始化失敗：{str(e)}")

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest, current_user: str = Depends(get_current_user)):
    """詢問學習問題"""
    global rag_instance

    # 檢查是否為圖片測試指令
    if re.match(r'^\d+$', request.question.strip()):
        try:
            image_id = int(request.question.strip())
            image_response = await get_test_image(image_id, current_user)

            start_time = datetime.now()
            response_time = (datetime.now() - start_time).total_seconds()
            log_learning_question(current_user, request.question, f"顯示圖片：{image_response.image_name}", 0, response_time)

            return AnswerResponse(
                answer=f"IMAGE:{image_response.image_url}|{image_response.message}",
                sources=[]
            )
        except HTTPException as e:
            return AnswerResponse(
                answer=f"❌ {e.detail}",
                sources=[]
            )

    if not rag_instance:
        raise HTTPException(status_code=400, detail="學習系統尚未初始化")

    try:
        start_time = datetime.now()
        answer, sources = rag_instance.ask(request.question)
        response_time = (datetime.now() - start_time).total_seconds()

        # 格式化來源資訊
        formatted_sources = []
        for doc in sources:
            source_info = {
                "source": os.path.basename(str(doc.metadata.get('source', '未知來源'))),
                "page": doc.metadata.get('page', 0) + 1,
                "content_preview": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
            }
            formatted_sources.append(source_info)

        # 記錄學習問題
        log_learning_question(current_user, request.question, answer, len(sources), response_time)

        return AnswerResponse(answer=answer, sources=formatted_sources)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"回答問題時發生錯誤：{str(e)}")

@app.get("/stats", response_model=LearnerStats)
async def get_learner_stats(current_user: str = Depends(get_current_user)):
    """取得學習者統計"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 獲取學習者基本資料
    cursor.execute('''
        SELECT user_name, today_visit_count, last_visit_time, total_questions
        FROM learners WHERE user_id = %s
    ''', (current_user,))
    learner_data = cursor.fetchone()

    # 今日問題數
    cursor.execute('''
        SELECT COUNT(*) as count
        FROM learning_questions
        WHERE user_id = %s AND DATE(asked_at) = CURRENT_DATE
    ''', (current_user,))
    questions_today = cursor.fetchone()['count']

    # 平均回應時間
    cursor.execute('''
        SELECT AVG(response_time) as avg
        FROM learning_questions WHERE user_id = %s
    ''', (current_user,))
    result = cursor.fetchone()
    avg_response_time = float(result['avg']) if result['avg'] else 0.0

    cursor.close()
    conn.close()

    return LearnerStats(
        user_name=learner_data['user_name'],
        total_questions=learner_data['total_questions'] or 0,
        questions_today=questions_today,
        today_visit_count=learner_data['today_visit_count'] or 0,
        last_visit_time=str(learner_data['last_visit_time']) if learner_data['last_visit_time'] else None,
        avg_response_time=round(avg_response_time, 2)
    )

@app.get("/learning/history", response_model=LearningHistoryResponse)
async def get_learning_history(
        limit: int = 50,
        offset: int = 0,
        current_user: str = Depends(get_current_user)
):
    """獲取學習歷史紀錄"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 獲取總筆數
    cursor.execute("SELECT COUNT(*) as count FROM learning_questions WHERE user_id = %s", (current_user,))
    total_count = cursor.fetchone()['count']

    # 獲取學習歷史紀錄
    cursor.execute('''
        SELECT question, answer, asked_at, response_time
        FROM learning_questions
        WHERE user_id = %s
        ORDER BY asked_at DESC
        LIMIT %s OFFSET %s
    ''', (current_user, limit, offset))

    records = cursor.fetchall()
    cursor.close()
    conn.close()

    # 格式化歷史紀錄
    history = []
    for record in records:
        history.append(LearningHistoryItem(
            question=record['question'],
            answer=record['answer'],
            timestamp=str(record['asked_at']),
            response_time=record['response_time']
        ))

    return LearningHistoryResponse(
        history=history,
        total_count=total_count
    )

@app.delete("/learning/history")
async def clear_learning_history(current_user: str = Depends(get_current_user)):
    """清除學習歷史紀錄"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM learning_questions WHERE user_id = %s", (current_user,))
    deleted_count = cursor.rowcount

    # 重置問題計數
    cursor.execute("UPDATE learners SET total_questions = 0 WHERE user_id = %s", (current_user,))

    conn.commit()
    cursor.close()
    conn.close()

    return {"message": f"已清除 {deleted_count} 筆學習紀錄"}

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

@app.get("/test/image/{image_id}")
async def get_test_image(image_id: int, current_user: str = Depends(get_current_user)):
    """測試圖片顯示功能"""
    images_folder = "./static/images"
    if not os.path.exists(images_folder):
        raise HTTPException(status_code=404, detail="圖片資料夾不存在")

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.webp']
    image_files = set()
    for ext in image_extensions:
        image_files.update(glob.glob(os.path.join(images_folder, ext)))
        image_files.update(glob.glob(os.path.join(images_folder, ext.upper())))

    image_files = sorted(list(image_files), key=natural_sort_key)

    if not image_files:
        raise HTTPException(status_code=404, detail="沒有找到任何圖片")

    if image_id < 1 or image_id > len(image_files):
        raise HTTPException(status_code=404, detail=f"圖片編號無效，請輸入 1 到 {len(image_files)} 之間的數字")

    selected_image = image_files[image_id - 1]
    image_name = os.path.basename(selected_image)

    return ImageResponse(
        image_url=f"/static/images/{image_name}",
        image_name=image_name,
        message=f"顯示第 {image_id} 張圖片：{image_name}"
    )

@app.get("/test/image-list")
async def get_image_list(current_user: str = Depends(get_current_user)):
    """獲取所有測試圖片列表"""
    images_folder = "./static/images"
    if not os.path.exists(images_folder):
        return {"images": [], "count": 0, "message": "圖片資料夾不存在"}

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_folder, ext)))
        image_files.extend(glob.glob(os.path.join(images_folder, ext.upper())))

    image_files.sort()
    image_names = [os.path.basename(f) for f in image_files]

    return {
        "images": image_names,
        "count": len(image_names),
        "message": f"找到 {len(image_names)} 張圖片"
    }

@app.get("/status")
async def get_status():
    """取得系統狀態"""
    global rag_instance
    return StatusResponse(
        status="ready" if rag_instance else "not_initialized",
        message="學習系統已就緒" if rag_instance else "學習系統尚未初始化"
    )

# 健康檢查端點
@app.get("/health")
async def health_check():
    """健康檢查端點"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected",
            "learning_system": "ready" if rag_instance else "not_initialized"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "database": "disconnected",
            "error": str(e)
        }

# 管理員端點 - 查看所有學習者狀態
@app.get("/admin/learners")
async def get_all_learners():
    """取得所有學習者狀態（管理員用）"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT user_name, user_id, last_visit_time, today_visit_count, 
               total_questions, created_at, is_active
        FROM learners 
        ORDER BY last_visit_time DESC NULLS LAST
    ''')
    
    learners = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return {
        "learners": [dict(learner) for learner in learners],
        "total_count": len(learners)
    }

if __name__ == "__main__":
    import uvicorn

    print("🚀 啟動學習追蹤 RAG 系統...")
    print("📱 網站網址：http://localhost:8080")
    print("📚 API 文件：http://localhost:8080/docs")
    print("📁 請確保 pdfFiles 資料夾中有要處理的檔案")
    print("🔑 請在 .env 檔案中設定 SECRET_KEY、OPENAI_API_KEY 和 DATABASE_URL")
    uvicorn.run(app, host="0.0.0.0", port=8080)
