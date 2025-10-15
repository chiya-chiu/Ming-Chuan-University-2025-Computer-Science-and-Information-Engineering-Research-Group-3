from openai import OpenAI
import os
from dotenv import load_dotenv

# 載入 .env 檔案
load_dotenv()

# 建立 OpenAI 客戶端
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print(f"目前載入的 API 金鑰為：{os.getenv('OPENAI_API_KEY')}")
key = os.getenv("OPENAI_API_KEY")
print(key.startswith("sk-"))
# 測試 API 連線
try:
    models = client.models.list()
    print("API 金鑰有效！")
except Exception as e:
    print(f"API 金鑰錯誤：{e}")
