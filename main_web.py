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
import json
from pathlib import Path

# 載入 .env 檔案
load_dotenv()   # 載入環境變數，像是 API 金鑰

# 安全設定
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this") # JWT 加密用
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30    #token 30 分鐘內有效

# 資料庫連線設定
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("請在 .env 檔案中設定 DATABASE_URL")

def get_db_connection():
    """取得 PostgreSQL 資料庫連線"""
    try:
        # 檢查是否為 Render 部署環境
        if "render.com" in DATABASE_URL or "amazonaws.com" in DATABASE_URL:
            # 雲端環境需要 SSL 連線
            conn = psycopg2.connect(
                DATABASE_URL,
                cursor_factory=RealDictCursor,
                sslmode='require'
            )
        else:
            # 本地環境不需要 SSL 連線
            conn = psycopg2.connect(
                DATABASE_URL,
                cursor_factory=RealDictCursor,
                sslmode='disable'  # 本地測試時停用 SSL
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

# 資料庫初始化
def init_database():
    """初始化 PostgreSQL 資料庫"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 使用者表 - PostgreSQL 語法
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,                    -- PostgreSQL 的自動遞增 ID
            user_id VARCHAR(255) UNIQUE NOT NULL,     -- 使用者唯一 ID（UUID）
            username VARCHAR(255) UNIQUE NOT NULL,    -- 使用者名稱（唯一）
            email VARCHAR(255),                       -- 電子信箱（可選）
            password_hash VARCHAR(255) NOT NULL,      -- 密碼的 SHA256 雜湊值
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- 註冊時間 使用 PostgreSQL 語法
            is_active BOOLEAN DEFAULT TRUE,           -- 是否啟用
            is_admin BOOLEAN DEFAULT FALSE            -- 是否為管理員
        )
    ''')

    # 問答紀錄表 - PostgreSQL 語法
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS questions_log (
            id SERIAL PRIMARY KEY,                    -- 問答紀錄編號
            user_id VARCHAR(255) NOT NULL,            -- 使用者 ID
            question TEXT NOT NULL,                   -- 問題內容
            answer TEXT NOT NULL,                     -- 回答內容
            sources_count INTEGER DEFAULT 0,          -- 來源段落數
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- 問答發生時間
            response_time REAL,                       -- 回答耗時（秒）
            FOREIGN KEY(user_id) REFERENCES users(user_id) -- 關聯到 users 表
        )
    ''')

    conn.commit()   # 儲存變更
    cursor.close()
    conn.close()    # 關閉連線

# 應用程式生命週期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 啟動時執行
    print("🔧 初始化資料庫...")
    test_db_connection()  # 測試連線
    init_database()
    print("✅ 資料庫初始化完成")
    yield
    # 關閉時執行（如果需要清理）

#這就是後端網站的主體
app = FastAPI(
    title="RAG 問答系統",
    description="基於文件的智能問答系統（含帳號管理）",
    lifespan=lifespan
)

# 掛載 /static 用來提供 CSS、JS 檔案
app.mount("/static", StaticFiles(directory="static"), name="static")


# 設定 CORS
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

# 資料模型（保持不變）
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

# 新增圖片回應模型
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
    """建立 JWT token，token 30 分鐘後過期"""
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

def get_user_from_db(user_id: str = None, username: str = None):
    """從資料庫取得使用者資料 - PostgreSQL 版本"""
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
    """記錄問答到資料庫 - PostgreSQL 版本"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO questions_log (user_id, question, answer, sources_count, response_time)
        VALUES (%s, %s, %s, %s, %s)
    ''', (user_id, question, answer, sources_count, response_time))  # 儲存完整的 answer，包含 CHARTS: 資訊
    conn.commit()
    cursor.close()
    conn.close()

def verify_admin(user_id: str):
    """驗證管理員權限"""
    user = get_user_from_db(user_id=user_id)
    if not user or not user['is_admin']:  # 使用字典方式存取
        raise HTTPException(status_code=403, detail="您沒有管理員權限")

def is_chart_relevant(chart_info, source_content):
    description = chart_info.get('generated_description', '').lower()
    caption = chart_info.get('original_caption', '').lower()

    # 使用更精確的關鍵字匹配
    description_keywords = [word for word in description.split() if len(word) > 3][:10]
    caption_keywords = [word for word in caption.split() if len(word) > 5]

    return (any(keyword in source_content for keyword in description_keywords) or
            any(keyword in source_content for keyword in caption_keywords))



# API 端點

@app.get("/", response_class=FileResponse)
async def serve_index():
    """顯示首頁"""
    return FileResponse("static/index.html")

@app.post("/register")
async def register_user(user: UserRegister):
    """使用者註冊 - PostgreSQL 版本"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 檢查使用者是否已存在
    cursor.execute("SELECT * FROM users WHERE username = %s", (user.username,))
    if cursor.fetchone():
        cursor.close()
        conn.close()
        raise HTTPException(status_code=400, detail="使用者名稱已存在")

    # 建立新使用者
    user_id = str(uuid.uuid4())
    password_hash = hash_password(user.password)

    cursor.execute('''
        INSERT INTO users (user_id, username, password_hash)
        VALUES (%s, %s, %s)
    ''', (user_id, user.username, password_hash))

    conn.commit()
    cursor.close()
    conn.close()

    return {"message": "註冊成功", "user_id": user_id}

@app.post("/login", response_model=Token)
async def login_user(user: UserLogin):
    """使用者登入 - PostgreSQL 版本"""
    db_user = get_user_from_db(username=user.username)

    # 比對密碼
    if not db_user or not verify_password(user.password, db_user['password_hash']):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="帳號或密碼錯誤"
        )

    # 建立 JWT token
    access_token = create_access_token(data={"sub": db_user['user_id']})

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_info": {
            "user_id": db_user['user_id'],
            "username": db_user['username'],
            "email": db_user['email'] if db_user['email'] else "",
            "is_admin": bool(db_user['is_admin'])
        }
    }

@app.get("/me")
async def get_current_user_info(current_user: str = Depends(get_current_user)):
    """取得目前登入使用者的資訊"""
    db_user = get_user_from_db(user_id=current_user)
    if not db_user:
        raise HTTPException(status_code=404, detail="使用者不存在")

    return {
        "user_id": db_user['user_id'],
        "username": db_user['username'],
        "email": db_user['email'] if db_user['email'] else "",
        "created_at": str(db_user['created_at']),
        "is_active": db_user['is_active'],
        "is_admin": db_user['is_admin']
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
            message="RAG 系統初始化完成"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"系統初始化失敗：{str(e)}")


# 在 ask_question 函數中新增圖片測試邏輯
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest, current_user: str = Depends(get_current_user)):
    """回答問題（需登入）"""
    global rag_instance

    """
    # 檢查是否為圖片測試指令
    if re.match(r'^\d+$', request.question.strip()):
        try:
            image_id = int(request.question.strip())
            image_response = await get_test_image(image_id, current_user)

            # 記錄問答
            start_time = datetime.now()
            response_time = (datetime.now() - start_time).total_seconds()
            log_question(current_user, request.question, f"顯示圖片：{image_response.image_name}", 0, response_time)

            return AnswerResponse(
                answer=f"IMAGE:{image_response.image_url}|{image_response.message}",
                sources=[]
            )
        except HTTPException as e:
            return AnswerResponse(
                answer=f"❌ {e.detail}",
                sources=[]
            )
    """

    if not rag_instance:
        raise HTTPException(status_code=400, detail="系統尚未初始化")

    # 原有的 RAG 問答邏輯...
    try:
        start_time = datetime.now()
        answer, sources = rag_instance.ask(request.question)
        response_time = (datetime.now() - start_time).total_seconds()

        # 檢查是否有圖表內容並載入圖表資訊
        chart_images = []
        chart_info_file = Path("pdfFiles/chart_metadata.json")  # 調整為你的 JSON 檔案路徑

        if chart_info_file.exists():
            try:
                with open(chart_info_file, 'r', encoding='utf-8') as f:
                    chart_data = json.load(f)

                # 檢查 Top 5 sources 中是否有包含 generated_description 的內容
                for doc in sources[:5]:  # 只檢查前5個
                    source_content = doc.page_content.lower()

                    # 遍歷所有圖表，檢查是否與檢索內容相關
                    for chart_id, chart_info in chart_data.items():
                        if 'generated_description' in chart_info:
                            # 檢查是否相關
                            if is_chart_relevant(chart_info, source_content):
                                # 構建圖片路徑
                                image_path = f"static/charts/{chart_id}.jpg"

                                # 只有圖片存在才加入
                                if os.path.exists(image_path):
                                    chart_images.append({
                                        'chart_id': chart_id,
                                        'image_url': f"/static/charts/{chart_id}.jpg",
                                        'caption': chart_info.get('original_caption', ''),
                                        'description': chart_info.get('generated_description', ''),
                                        'chart_type': chart_info.get('chart_type', ''),
                                        'chart_number': chart_info.get('chart_number', '')
                                    })

                                    # 找到一個有效的圖片就跳出
                                    if len(chart_images) >= 1:
                                        break

                    if len(chart_images) >= 1:
                        break

            except Exception as e:
                print(f"載入圖表資訊時發生錯誤：{e}")

        # 格式化來源資訊
        formatted_sources = []
        for doc in sources:
            source_info = {
                "source": os.path.basename(str(doc.metadata.get('source', '未知來源'))),
                "page": doc.metadata.get('page', 0) + 1,
                "content_preview": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
            }
            formatted_sources.append(source_info)

        # 如果有相關圖表，將圖片資訊加入回應
        final_answer = answer

        if chart_images:
            chart_info_text = "\n\n相關圖表：\n"
            """
            for i, chart in enumerate(chart_images, 1):
                chart_info_text += f"{i}. {chart['caption']} (圖 {chart['chart_number']})\n"
             """
            final_answer = answer + chart_info_text + f"\nCHARTS:{json.dumps(chart_images, ensure_ascii=False)}"

        # 記錄問答
        log_question(current_user, request.question, final_answer, len(sources), response_time)

        return AnswerResponse(answer=final_answer, sources=formatted_sources)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"回答問題時發生錯誤：{str(e)}")

@app.get("/stats", response_model=UserStats)
async def get_user_stats(current_user: str = Depends(get_current_user)):
    """取得使用者問答統計 - PostgreSQL 版本"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 總問題數
    cursor.execute("SELECT COUNT(*) FROM questions_log WHERE user_id = %s", (current_user,))
    total_questions = cursor.fetchone()['count']

    # 今日問題數 - 使用 PostgreSQL 語法
    cursor.execute('''
        SELECT COUNT(*)
        FROM questions_log
        WHERE user_id = %s AND DATE(created_at) = CURRENT_DATE
    ''', (current_user,))
    questions_today = cursor.fetchone()['count']

    # 平均回應時間
    cursor.execute("SELECT AVG(response_time) FROM questions_log WHERE user_id = %s", (current_user,))
    result = cursor.fetchone()
    avg_response_time = float(result['avg']) if result['avg'] else 0.0

    cursor.close()
    conn.close()

    return UserStats(
        total_questions=total_questions,
        questions_today=questions_today,
        avg_response_time=round(avg_response_time, 2)
    )

@app.get("/admin/stats")
async def get_admin_stats(current_user: str = Depends(get_current_user)):
    """管理員統計 - PostgreSQL 版本"""
    verify_admin(current_user)

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM users")
    total_users = cursor.fetchone()['count']

    cursor.execute("SELECT COUNT(*) FROM questions_log")
    total_questions = cursor.fetchone()['count']

    cursor.execute('''
        SELECT COUNT(*)
        FROM questions_log
        WHERE DATE(created_at) = CURRENT_DATE
    ''')
    questions_today = cursor.fetchone()['count']

    cursor.close()
    conn.close()

    return {
        "total_users": total_users,
        "total_questions": total_questions,
        "questions_today": questions_today
    }

@app.get("/status")
async def get_status():
    """取得系統狀態"""
    global rag_instance
    return StatusResponse(
        status="ready" if rag_instance else "not_initialized",
        message="系統已就緒" if rag_instance else "系統尚未初始化"
    )

@app.get("/chat/history", response_model=ChatHistoryResponse)
async def get_chat_history(
        limit: int = 50,
        offset: int = 0,
        current_user: str = Depends(get_current_user)
):
    """獲取使用者的聊天歷史紀錄 - PostgreSQL 版本"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 獲取總筆數
    cursor.execute("SELECT COUNT(*) FROM questions_log WHERE user_id = %s", (current_user,))
    total_count = cursor.fetchone()['count']

    # 獲取歷史紀錄 - 使用 PostgreSQL 的 LIMIT/OFFSET 語法
    cursor.execute('''
        SELECT question, answer, created_at, response_time
        FROM questions_log
        WHERE user_id = %s
        ORDER BY created_at DESC
        LIMIT %s OFFSET %s
    ''', (current_user, limit, offset))

    records = cursor.fetchall()
    cursor.close()
    conn.close()

    # 格式化歷史紀錄
    history = []
    for record in records:
        history.append(ChatHistoryItem(
            question=record['question'],
            answer=record['answer'],
            timestamp=str(record['created_at']),
            response_time=record['response_time']
        ))

    return ChatHistoryResponse(
        history=history,
        total_count=total_count
    )

@app.delete("/chat/history")
async def clear_chat_history(current_user: str = Depends(get_current_user)):
    """清除使用者的聊天歷史紀錄 - PostgreSQL 版本"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM questions_log WHERE user_id = %s", (current_user,))
    deleted_count = cursor.rowcount

    conn.commit()
    cursor.close()
    conn.close()

    return {"message": f"已清除 {deleted_count} 筆歷史紀錄"}

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

@app.get("/test/image/{image_id}")
async def get_test_image(image_id: int, current_user: str = Depends(get_current_user)):
    """測試圖片顯示功能"""

    # 檢查 images 資料夾是否存在
    images_folder = "./static/images"
    if not os.path.exists(images_folder):
        raise HTTPException(status_code=404, detail="圖片資料夾不存在")

    # 獲取資料夾中的所有圖片檔案
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.webp']
    image_files =set()
    for ext in image_extensions:
        image_files.update(glob.glob(os.path.join(images_folder, ext)))
        image_files.update(glob.glob(os.path.join(images_folder, ext.upper())))

    # 按檔名排序
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

# 新增健康檢查端點（防休眠用）
@app.get("/health")
async def health_check():
    """健康檢查端點"""
    try:
        # 測試資料庫連線
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected",
            "rag_system": "ready" if rag_instance else "not_initialized"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "database": "disconnected",
            "error": str(e)
        }

@app.get("/debug/database")
async def debug_database():
    """調試資料庫連線狀態"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 測試基本連線
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        
        # 檢查表格是否存在
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return {
            "status": "success",
            "database_version": version['version'],
            "tables": [table['table_name'] for table in tables],
            "connection": "OK"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "connection": "FAILED"
        }

@app.get("/debug/users-table")
async def debug_users_table():
    """調試 users 表格狀態"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 檢查 users 表格是否存在
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'users'
            );
        """)
        table_exists = cursor.fetchone()['exists']
        
        if not table_exists:
            cursor.close()
            conn.close()
            return {
                "status": "error",
                "message": "users 表格不存在",
                "table_exists": False
            }
        
        # 獲取表格結構
        cursor.execute("""
            SELECT column_name, data_type, is_nullable 
            FROM information_schema.columns 
            WHERE table_name = 'users'
            ORDER BY ordinal_position;
        """)
        columns = cursor.fetchall()
        
        # 獲取使用者數量
        cursor.execute("SELECT COUNT(*) as count FROM users")
        user_count = cursor.fetchone()['count']
        
        # 獲取前 5 個使用者（不包含敏感資訊）
        cursor.execute("""
            SELECT user_id, username, email, created_at, is_active, is_admin 
            FROM users 
            ORDER BY created_at DESC 
            LIMIT 5
        """)
        sample_users = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return {
            "status": "success",
            "table_exists": True,
            "user_count": user_count,
            "columns": [
                {
                    "name": col['column_name'],
                    "type": col['data_type'],
                    "nullable": col['is_nullable']
                } for col in columns
            ],
            "sample_users": [dict(user) for user in sample_users]
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "無法查詢 users 表格"
        }
    
@app.get("/debug/questions-log")
async def debug_questions_log():
    """問答紀錄"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, user_id, question, created_at, response_time
            FROM questions_log
            ORDER BY created_at DESC
            LIMIT 20
        """)
        records = cursor.fetchall()
        cursor.close()
        conn.close()
        return {"status": "success", "records": records}
    except Exception as e:
        return {"status": "error", "message": str(e)}
        
@app.get("/debug/init-check")
async def debug_init_check():
    """檢查初始化狀態和環境變數"""
    return {
        "database_url_set": bool(os.getenv("DATABASE_URL")),
        "openai_key_set": bool(os.getenv("OPENAI_API_KEY")),
        "secret_key_set": bool(os.getenv("SECRET_KEY")),
        "rag_initialized": rag_instance is not None,
        "pdf_folder_exists": os.path.exists("./pdfFiles")
    }

@app.post("/debug/force-init-db")
async def force_init_database():
    """強制重新初始化資料庫（僅供調試使用）"""
    try:
        init_database()
        return {
            "status": "success",
            "message": "資料庫重新初始化完成"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"資料庫初始化失敗：{str(e)}"
        }

if __name__ == "__main__":
    import uvicorn

    print("🚀 啟動 RAG 網站服務（PostgreSQL 版本）...")
    print("📱 網站網址：http://localhost:8080")
    print("📚 API 文件：http://localhost:8080/docs")
    print("📁 請確保 pdfFiles 資料夾中有要處理的檔案")
    print("🔑 請在 .env 檔案中設定 SECRET_KEY、OPENAI_API_KEY 和 DATABASE_URL")
    uvicorn.run(app, host="0.0.0.0", port=8080)
