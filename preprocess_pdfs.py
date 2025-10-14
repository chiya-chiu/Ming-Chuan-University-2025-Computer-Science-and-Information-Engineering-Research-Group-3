#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF 預處理統一腳本 - ABC 三階段整合

此腳本負責將 PDF 文件進行完整的預處理：
【階段 A】圖表識別與擷取（Caption Extraction）
【階段 B】LLM 圖表描述生成（LLM Description）
【階段 C】增強文檔生成（Enhanced Document Generation）

重要特性：
1. 只在 PDF 更新時才重新處理（檔案變更偵測）
2. 批量處理所有圖表（節省 API 成本）
3. 完整的 LOG 記錄系統
4. 支援手動強制重新處理
5. 產出可直接被 RAG_Helper 讀取的 enhanced_doc_*.txt

使用方式：
    python preprocess_pdfs.py                    # 自動處理有變更的PDF
    python preprocess_pdfs.py --force            # 強制重新處理所有PDF
    python preprocess_pdfs.py --use-mock         # 使用Mock LLM（不消耗API）
    python preprocess_pdfs.py --pdf-folder ./my_pdfs  # 指定PDF目錄
"""

import os
import sys
import json
import hashlib
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# 設置 Windows UTF-8 編碼支援
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# 匯入專案模組
from llm_config import LLMConfig

# 匯入 ABC 階段模組
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules', 'pdf_Cutting_TextReplaceImage'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules', 'pdf_Cutting_TextReplaceImage', 'enhanced_version', 'backend'))

# from pdf_chart_extractor import PDFChartExtractor  # 舊版，已封存為 pdf_chart_extractor_legacy.py
from caption_extractor_sA import PDFCaptionContextProcessor


class PreprocessLogger:
    """預處理專用的 Logger"""

    def __init__(self, log_dir: str = "preprocess_logs"):
        """
        初始化 Logger

        Args:
            log_dir: LOG 檔案存放目錄
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # 建立時間戳記
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"preprocess_{timestamp}.log"

        # 配置 Logger
        self.logger = logging.getLogger("PDFPreprocessor")
        self.logger.setLevel(logging.DEBUG)

        # 清除舊的 handlers
        self.logger.handlers.clear()

        # 檔案處理器（詳細）
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)

        # 控制台處理器（簡潔）
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)

        # 加入處理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.logger.info(f"📝 LOG 檔案: {log_file}")

    def info(self, message: str):
        """記錄資訊"""
        self.logger.info(message)

    def debug(self, message: str):
        """記錄除錯資訊"""
        self.logger.debug(message)

    def warning(self, message: str):
        """記錄警告"""
        self.logger.warning(f"⚠️ {message}")

    def error(self, message: str):
        """記錄錯誤"""
        self.logger.error(f"❌ {message}")

    def success(self, message: str):
        """記錄成功"""
        self.logger.info(f"✅ {message}")


class FileChangeDetector:
    """檔案變更偵測器"""

    def __init__(self, cache_file: str = ".preprocess_cache.json"):
        """
        初始化偵測器

        Args:
            cache_file: 快取檔案路徑
        """
        self.cache_file = Path(cache_file)
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        """載入快取"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        """儲存快取"""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def _get_file_hash(self, file_path: Path) -> str:
        """計算檔案的 MD5 雜湊值"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def is_changed(self, file_path: Path) -> bool:
        """
        檢查檔案是否有變更

        Args:
            file_path: 檔案路徑

        Returns:
            bool: True 表示檔案有變更或是新檔案
        """
        file_key = str(file_path)
        current_hash = self._get_file_hash(file_path)

        if file_key not in self.cache:
            # 新檔案
            return True

        cached_info = self.cache[file_key]
        if cached_info.get("hash") != current_hash:
            # 檔案已變更
            return True

        return False

    def update_file(self, file_path: Path, metadata: Optional[Dict] = None):
        """
        更新檔案的快取資訊

        Args:
            file_path: 檔案路徑
            metadata: 額外的中繼資料
        """
        file_key = str(file_path)
        self.cache[file_key] = {
            "hash": self._get_file_hash(file_path),
            "last_processed": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self._save_cache()


class PDFPreprocessor:
    """PDF 預處理器 - ABC 三階段整合"""

    def __init__(
            self,
            pdf_folder: str = "./pdfFiles",
            output_folder: str = "./pdfFiles/enhanced_docs",  # 修正：統一存放在子資料夾
            charts_folder: str = "./pdfFiles/charts",  # 修正：改為 pdfFiles/charts
            use_mock_llm: bool = False,
            logger: Optional[PreprocessLogger] = None
    ):
        """
        初始化預處理器

        Args:
            pdf_folder: PDF 檔案目錄
            output_folder: 輸出目錄（enhanced_doc_*.txt）
            charts_folder: 圖表圖片目錄
            use_mock_llm: 是否使用 Mock LLM（不消耗 API）
            logger: Logger 實例
        """
        self.pdf_folder = Path(pdf_folder)
        self.output_folder = Path(output_folder)
        self.charts_folder = Path(charts_folder)
        self.use_mock_llm = use_mock_llm
        self.logger = logger or PreprocessLogger()

        # 建立必要目錄
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.charts_folder.mkdir(parents=True, exist_ok=True)

        # 初始化檔案變更偵測器
        self.change_detector = FileChangeDetector()

        # 初始化統計資訊
        self.stats = {
            "total_pdfs": 0,
            "processed_pdfs": 0,
            "skipped_pdfs": 0,
            "total_charts": 0,
            "total_descriptions": 0,
            "total_api_calls": 0,
            "start_time": datetime.now()
        }

    def get_pdf_files(self) -> List[Path]:
        """取得所有 PDF 檔案"""
        pdf_files = list(self.pdf_folder.glob("*.pdf"))
        self.stats["total_pdfs"] = len(pdf_files)
        return pdf_files

    def stage_a_extract_charts(self, pdf_path: Path) -> Tuple[Dict, int]:
        """
        【階段 A】圖表識別與擷取（使用 caption_extractor_sA）

        Args:
            pdf_path: PDF 檔案路徑

        Returns:
            Tuple[Dict, int]: (圖表資訊字典, 圖表數量)
        """
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"【階段 A】圖表識別與擷取: {pdf_path.name}")
        self.logger.info(f"{'=' * 60}")

        try:
            # 使用 PDFCaptionContextProcessor（階段 A 完整版）
            extractor = PDFCaptionContextProcessor(
                context_window=200,
                min_caption_length=5,
                confidence_threshold=0.3
            )

            # 處理 PDF 並擷取圖表（包含 Caption 識別）
            caption_pairs = extractor.process_pdf(
                str(pdf_path),
                use_image_guided_search=True,
                charts_output_dir=str(self.charts_folder)
            )

            # 轉換為 metadata 格式
            chart_metadata = {}
            for i, pair in enumerate(caption_pairs, 1):
                chart_id = f"{pdf_path.stem}_chart_{i}"
                chart_metadata[chart_id] = {
                    "chart_id": chart_id,
                    "chart_type": pair.caption.caption_type,
                    "chart_number": pair.caption.number,
                    "original_caption": pair.caption.text,
                    "page_number": pair.caption.page_number,
                    "confidence_score": pair.caption.confidence,
                    "source_file": pdf_path.name,
                    "context_references": [ctx.text for ctx in pair.contexts[:3]]  # 取前3個相關段落
                }

            chart_count = len(chart_metadata)
            self.stats["total_charts"] += chart_count

            self.logger.success(f"階段 A 完成：識別了 {chart_count} 個圖表（含 Caption）")
            self.logger.debug(f"圖表資訊: {json.dumps(chart_metadata, ensure_ascii=False, indent=2)}")

            return chart_metadata, chart_count

        except Exception as e:
            self.logger.error(f"階段 A 失敗: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return {}, 0

    def stage_b_generate_descriptions(self, chart_metadata: Dict) -> Dict:
        """
        【階段 B】LLM 批量描述生成

        Args:
            chart_metadata: 階段 A 產生的圖表資訊

        Returns:
            Dict: 包含描述的圖表資訊
        """
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"【階段 B】LLM 批量描述生成")
        self.logger.info(f"{'=' * 60}")

        if not chart_metadata:
            self.logger.warning("沒有圖表需要生成描述")
            return {}

        try:
            # 動態匯入 LLM 模組
            try:
                from llm_providers_sB import LLMManager, LLMRequest
            except ImportError:
                self.logger.error("無法匯入 llm_providers_sB，請確保子模組已正確初始化")
                return chart_metadata

            # 初始化 LLM Manager
            # 優先使用 llm_config.py 的統一配置，否則使用命令列參數
            provider = LLMConfig.CHART_DESCRIPTION_PROVIDER if LLMConfig.CHART_DESCRIPTION_PROVIDER != "auto" else ("mock" if self.use_mock_llm else "auto")
            llm_manager = LLMManager(preferred_provider=provider)

            self.logger.info(f"使用 LLM 提供者: {llm_manager.get_current_provider()}")

            # 批量生成描述
            enriched_metadata = {}
            success_count = 0
            total_count = len(chart_metadata)

            for chart_id, chart_info in chart_metadata.items():
                self.logger.info(f"處理圖表 {success_count + 1}/{total_count}: {chart_id}")

                # 構建 prompt
                caption = chart_info.get('original_caption', '')
                chart_type = chart_info.get('chart_type', 'figure')

                prompt = f"""請為以下圖表生成詳細的學術描述：

圖表類型：{chart_type}
原始說明：{caption}

請用繁體中文生成 2-3 句專業且詳細的描述，說明此圖表的內容、意義和在文件中的作用。"""

                # 調用 LLM
                request = LLMRequest(
                    prompt=prompt,
                    max_tokens=LLMConfig.CHART_DESCRIPTION_MAX_TOKENS,
                    temperature=LLMConfig.CHART_DESCRIPTION_TEMPERATURE,
                    system_message="你是專業的學術文件分析助手，擅長解讀和描述圖表內容。"
                )

                response = llm_manager.generate(request)
                self.stats["total_api_calls"] += 1

                if response.success:
                    chart_info['generated_description'] = response.content
                    enriched_metadata[chart_id] = chart_info
                    success_count += 1

                    self.logger.debug(f"  描述: {response.content[:100]}...")
                    self.logger.debug(f"  Token 使用: {response.token_usage}")
                else:
                    self.logger.warning(f"  生成失敗: {response.error_message}")
                    # 仍然保留圖表資訊，但沒有描述
                    enriched_metadata[chart_id] = chart_info

            self.stats["total_descriptions"] += success_count
            self.logger.success(f"階段 B 完成：成功生成 {success_count}/{total_count} 個描述")

            return enriched_metadata

        except Exception as e:
            self.logger.error(f"階段 B 失敗: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return chart_metadata

    def stage_c_generate_enhanced_docs(
            self,
            pdf_path: Path,
            chart_metadata: Dict
    ) -> List[Path]:
        """
        【階段 C】增強文檔生成

        將 PDF 原文與圖表描述合併，產生 enhanced_doc_*.txt

        Args:
            pdf_path: PDF 檔案路徑
            chart_metadata: 包含描述的圖表資訊

        Returns:
            List[Path]: 產生的 enhanced_doc 檔案路徑列表
        """
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"【階段 C】增強文檔生成")
        self.logger.info(f"{'=' * 60}")

        try:
            import fitz  # PyMuPDF

            doc = fitz.open(pdf_path)
            enhanced_docs = []

            # 使用PDF檔名作為前綴
            pdf_basename = pdf_path.stem

            # 建立單一 enhanced 文檔內容
            all_content = f"""{'=' * 80}
{pdf_basename}.pdf - Enhanced Document
包含圖表: {len(chart_metadata)} 張
{'=' * 80}

"""

            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()

                # 檢查此頁是否有圖表
                page_charts = [
                    chart_info for chart_info in chart_metadata.values()
                    if chart_info.get('page_number') == page_num + 1
                ]

                # 添加頁面標題
                all_content += f"\n{'=' * 60}\n"
                all_content += f"頁面 {page_num + 1}\n"
                all_content += f"{'=' * 60}\n\n"

                # 添加原始文字
                all_content += page_text

                # 如果有圖表，附加描述
                if page_charts:
                    all_content += "\n\n--- 本頁圖表 ---\n"
                    for chart in page_charts:
                        chart_type = chart.get('chart_type', 'figure')
                        chart_num = chart.get('chart_number', '')
                        caption = chart.get('original_caption', '')
                        description = chart.get('generated_description', '')
                        chart_id = chart.get('chart_id', '')

                        all_content += f"\n[📊 圖表 {chart_num}: {caption}]\n"
                        all_content += f"類型: {chart_type}\n"
                        all_content += f"Chart ID: {chart_id}\n"
                        all_content += f"LLM 描述: {description}\n"
                        all_content += f"---\n"

                all_content += "\n"

            doc.close()

            # 儲存單一檔案，使用 PDF 檔名
            output_path = self.output_folder / f"{pdf_basename}_enhanced.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(all_content)

            self.logger.success(f"階段 C 完成：產生增強文檔 {output_path.name}")
            return [output_path]

        except Exception as e:
            self.logger.error(f"階段 C 失敗: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return []

    def process_pdf(self, pdf_path: Path, force: bool = False) -> bool:
        """
        處理單個 PDF 檔案（ABC 三階段）

        Args:
            pdf_path: PDF 檔案路徑
            force: 是否強制重新處理

        Returns:
            bool: 是否成功處理
        """
        self.logger.info(f"\n\n{'*' * 80}")
        self.logger.info(f"開始處理: {pdf_path.name}")
        self.logger.info(f"{'*' * 80}\n")

        # 檢查是否需要處理
        if not force and not self.change_detector.is_changed(pdf_path):
            self.logger.info(f"⏭️  跳過 {pdf_path.name}（檔案未變更）")
            self.stats["skipped_pdfs"] += 1
            return True

        try:
            # 階段 A: 圖表擷取
            chart_metadata, chart_count = self.stage_a_extract_charts(pdf_path)

            # 階段 B: LLM 描述生成
            if chart_metadata:
                chart_metadata = self.stage_b_generate_descriptions(chart_metadata)

            # 階段 C: 增強文檔生成
            enhanced_docs = self.stage_c_generate_enhanced_docs(pdf_path, chart_metadata)

            # 儲存 chart_metadata.json
            metadata_path = self.output_folder / "chart_metadata.json"

            # 如果已存在，則合併
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    existing_metadata = json.load(f)
                existing_metadata.update(chart_metadata)
                chart_metadata = existing_metadata

            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(chart_metadata, f, ensure_ascii=False, indent=2)

            # 更新檔案快取
            self.change_detector.update_file(pdf_path, {
                "chart_count": chart_count,
                "enhanced_docs_count": len(enhanced_docs)
            })

            self.stats["processed_pdfs"] += 1
            self.logger.success(f"\n✅ {pdf_path.name} 處理完成！")
            return True

        except Exception as e:
            self.logger.error(f"處理 {pdf_path.name} 時發生錯誤: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def run(self, force: bool = False):
        """
        執行預處理流程

        Args:
            force: 是否強制重新處理所有 PDF
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("PDF 預處理系統啟動")
        self.logger.info("=" * 80)

        # 印出配置
        self.logger.info(f"\n【配置資訊】")
        self.logger.info(f"PDF 目錄: {self.pdf_folder}")
        self.logger.info(f"輸出目錄: {self.output_folder}")
        self.logger.info(f"圖表目錄: {self.charts_folder}")
        self.logger.info(f"LLM 模式: {'Mock (不消耗API)' if self.use_mock_llm else 'OpenAI (消耗API)'}")
        self.logger.info(f"強制重新處理: {'是' if force else '否'}")

        # 取得 PDF 檔案
        pdf_files = self.get_pdf_files()
        self.logger.info(f"\n找到 {len(pdf_files)} 個 PDF 檔案")

        if not pdf_files:
            self.logger.warning("沒有找到 PDF 檔案，結束處理")
            return

        # 逐個處理
        for pdf_path in pdf_files:
            self.process_pdf(pdf_path, force=force)

        # 印出統計資訊
        self._print_statistics()

    def _print_statistics(self):
        """印出統計資訊"""
        elapsed_time = (datetime.now() - self.stats["start_time"]).total_seconds()

        self.logger.info("\n" + "=" * 80)
        self.logger.info("處理統計")
        self.logger.info("=" * 80)
        self.logger.info(f"總 PDF 數量: {self.stats['total_pdfs']}")
        self.logger.info(f"已處理: {self.stats['processed_pdfs']}")
        self.logger.info(f"已跳過: {self.stats['skipped_pdfs']}")
        self.logger.info(f"總圖表數: {self.stats['total_charts']}")
        self.logger.info(f"成功描述數: {self.stats['total_descriptions']}")
        self.logger.info(f"API 調用次數: {self.stats['total_api_calls']}")
        self.logger.info(f"總耗時: {elapsed_time:.2f} 秒")
        self.logger.info("=" * 80 + "\n")


def main():
    """主程式"""
    parser = argparse.ArgumentParser(
        description="PDF 預處理系統 - ABC 三階段整合",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例：
  python preprocess_pdfs.py                    # 自動處理有變更的PDF
  python preprocess_pdfs.py --force            # 強制重新處理所有PDF
  python preprocess_pdfs.py --use-mock         # 使用Mock LLM（不消耗API）
  python preprocess_pdfs.py --pdf-folder ./my_pdfs  # 指定PDF目錄
        """
    )

    parser.add_argument(
        "--pdf-folder",
        default="./pdfFiles",
        help="PDF 檔案目錄 (預設: ./pdfFiles)"
    )
    parser.add_argument(
        "--output-folder",
        default="./pdfFiles/enhanced_docs",
        help="輸出目錄 (預設: ./pdfFiles/enhanced_docs)"
    )
    parser.add_argument(
        "--charts-folder",
        default="./pdfFiles/charts",
        help="圖表圖片目錄 (預設: ./pdfFiles/charts)"
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="使用 Mock LLM（不消耗 API 額度）"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="強制重新處理所有 PDF"
    )

    args = parser.parse_args()

    # 初始化 Logger
    logger = PreprocessLogger()

    # 初始化預處理器
    preprocessor = PDFPreprocessor(
        pdf_folder=args.pdf_folder,
        output_folder=args.output_folder,
        charts_folder=args.charts_folder,
        use_mock_llm=args.use_mock,
        logger=logger
    )

    # 執行預處理
    preprocessor.run(force=args.force)


if __name__ == "__main__":
    main()
