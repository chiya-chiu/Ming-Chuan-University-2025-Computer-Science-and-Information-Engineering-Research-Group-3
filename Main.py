import os
import asyncio
from RAG_Helper import RAGHelper
from dotenv import load_dotenv

# 載入 .env 檔案
load_dotenv()

async def main():
    # 檢查是否有設定 API 金鑰
    if not os.getenv("OPENAI_API_KEY"):
        print("錯誤：請在 .env 檔案中設定 OPENAI_API_KEY")
        return

    try:

        rag = RAGHelper(pdf_folder=r"./pdfFiles", chunk_size=200, chunk_overlap=30)

        print("正在載入和處理文件...")
        await rag.load_and_prepare(['.pdf', '.txt', '.docx', '.md', '.csv'])  # 載入其他格式檔案：await rag.load_and_prepare(['.pdf', '.txt', '.docx'])

        print("設置問答系統...")
        rag.setup_retrieval_chain(k=5, similarity_threshold=1.1)

        print("\n=== RAG 問答系統已準備就緒 ===")
        print("輸入問題開始對話，輸入 'quit'、'exit' 或 'q' 結束程式")
        print("=" * 50)

        while True:
            try:
                question = input("\n請輸入您的問題：").strip()

                # 檢查退出條件
                if question.lower() in ['quit', 'exit', 'q', '']:
                    print("程式結束")
                    break

                answer, sources = rag.ask(question)
                print(f"回答：\n{answer}")

                print("\n來源頁面：")
                for i, doc in enumerate(sources, 1):
                    source = doc.metadata.get('source', '未知來源')
                    page = doc.metadata.get('page', '未知頁數') + 1
                    print(f"{i}. 來源：{os.path.basename(str(source))}, 頁數：{page}")

                    # 顯示部分內容（可選）
                    content_preview = doc.page_content[:300] + "..." if len(
                        doc.page_content) > 100 else doc.page_content
                    print(f"   內容預覽：\n{content_preview}")

            except KeyboardInterrupt:
                # 處理 Ctrl+C 中斷
                print("\n\n程式被使用者中斷")
                break
            except EOFError:
                # 處理 Ctrl+D (Unix) 或 Ctrl+Z (Windows) 的 EOF
                print("\n\n檢測到輸入結束")
                break
            except Exception as e:
                print(f"回答問題時發生錯誤：{e}")
                print("請重新輸入問題或輸入 'quit' 結束程式")

    except Exception as e:
        print(f"程式執行時發生錯誤：{e}")

# 如果直接執行此檔案
if __name__ == "__main__":
    asyncio.run(main())