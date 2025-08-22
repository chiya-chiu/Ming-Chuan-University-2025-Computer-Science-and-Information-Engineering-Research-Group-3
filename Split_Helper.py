import re
import fitz  # PyMuPDF
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class Document:
    """Document class for compatibility with langchain"""
    page_content: str
    metadata: Dict[str, Any]


class SplitHelper:
    """PDF document splitting helper with advanced header detection and chunking"""

    def __init__(self):
        # === 可調參數 ===
        self.CENTER_ONLY = False
        self.NUMBERED_HEADERS_ONLY = False

        self.HEADER_HI_PCT = 90
        self.HEADER_RATIO_MIN = 1.5
        self.REPETITIVE_MIN_PAGES = 3
        self.MARGIN_RATIO = 0.15

        self.THRESHOLD_MODE = "hybrid"  # "page" / "doc" / "hybrid"
        self.AUTO_THRESHOLD = True
        self.HEADER_ABS_MIN_PT = 16.0

        self.AUTO_TUNE_DOC = True
        self.TUNE_TARGET_RANGE = (0.8, 2.2)
        self.TUNE_MAX_STEPS = 8
        self.TUNE_STEP_UP = 1.08
        self.TUNE_STEP_DOWN = 0.92

        self.PAGE_FALLBACK_WHEN_NO_HEADER = True

        self.DROP_SMALLEST_PAGENUM = True
        self.SMALL_FONT_PCT = 20
        self.PAGENUM_MARGIN = 0.15

        self.PAGENUM_X_CENTER = True
        self.PAGENUM_X_LEFT = True
        self.PAGENUM_X_RIGHT = True
        self.PAGENUM_X_CENTER_TOL = 0.18
        self.PAGENUM_X_EDGE_BAND = 0.18

        self.PAGENUM_MAX_LEN = 12
        self.PAGENUM_DIGIT_RATIO = 0.5

        # 修正正則表達式
        self.PAGE_NUM_RE = re.compile(r"""
        ^\s*(
            第\s*[一二三四五六七八九十百千零〇0-9]+\s*页          |
            p(?:age|\.)?\s*\d+(?:\s*/\s*\d+)?                       |
            [ivxlcdm]+\s*(?:/\s*[ivxlcdm]+)?                        |
            \d+\s*/\s*\d+                                           |
            [—–-]?\s*\d+\s*[—–-]?                                   |
            \d+\s*[—–-]\s*\d+                                       |
            \d{1,3}
        )\s*$
        """, re.I | re.X)

        self.TOP_HEADER_BAND_RATIO = 0.22
        self.TOP_HEADER_NEAR_HEADER_THR = 1.30

        self.BULLET_PREFIX_RE = re.compile(r"^\s*[■●◆◼▪◦•‣∙·]\s*")
        self.SEP_CLASS = r"[-—–\-．．·]"

        self.HEADER_NUMBER_PATTERNS = [
            rf"^\s*\d+\s*{self.SEP_CLASS}\s*\d+(?:\s*{self.SEP_CLASS}\s*\d+)*\s*.+$",
            r"^\s*\d{1,4}\s+.+$",
            r"^\s*第\s*[一二三四五六七八九十百千零〇0-9]+\s*[章节篇]\s+.+$",
        ]
        self.NUMBER_TOKEN_PATTERN = rf"^\s*(?:\d+\s*{self.SEP_CLASS}\s*\d+(?:\s*{self.SEP_CLASS}\s*\d+)*|\d{{1,4}})\s*$"
        self._HEADER_BAD_TRAIL = tuple("，,。：:；;！!？?．.、")

        self.REPEAT_MIN_LEN = 3

        # 智慧合併參數
        self.SMART_CONSOLIDATE = True
        self.MERGE_TRIGGER_GAP = 200
        self.MERGE_MIN_RATIO = 0.5
        self.MERGE_MIN_ABS = 180
        self.ALLOW_OVERSHOOT = 0.02
        self.TAIL_OVERSHOOT = 0.1

        self.RIGHT_PUNCT = set("，,。．.！？!?：:；;）)]}》」％℃…'\"`")
        self.LEFT_PUNCT = set("（([{《「\"'「「")
        self._BODY_PUNCT = set("。！？；，,.!?;:")

    def soft_join(self, buf: str, s: str) -> str:
        """智能文本連接，處理標點符號間隔"""
        s = s.strip()
        if not buf:
            return s
        if not s:
            return buf

        last = buf[-1]
        first = s[0]

        if last.isspace() or last in self.LEFT_PUNCT:
            return buf + s
        if first in self.RIGHT_PUNCT:
            return buf + s
        return buf + " " + s

    def clean_spaces(self, s: str) -> str:
        """清理空格和特殊字符"""
        s = s.replace("\u3000", " ").replace("\xa0", " ")
        return re.sub(r"[ \t]{2,}", " ", s).strip()

    def normalize_text(self, s: str) -> str:
        """標準化文本格式"""
        s = s.replace('\r\n', '\n').replace('\r', '\n')
        s = re.sub(r'\n{2,}', '\n\n', s)
        s = s.replace('\n', ' ')
        s = re.sub(r'[ \t\u3000]+', ' ', s)
        s = re.sub(r'\s{2,}', ' ', s)
        return s.strip()

    def smart_sentence_split(self, text: str) -> List[str]:
        """智能句子分割"""
        pat = re.compile(r'[^。！？!?；;]*[。！？!?；;]')
        sentences = pat.findall(text)
        used = sum(len(s) for s in sentences)
        tail = text[used:].strip()
        if tail:
            sentences.append(tail)
        return [s.strip() for s in sentences if s.strip()]

    def smart_chunk_merge_by_chars(self, sentences: List[str], target_length: int = 400,
                                   tolerance: int = 200) -> List[str]:
        """按字符數智能合併句子"""
        chunks, buf, size = [], [], 0
        min_len = target_length - tolerance
        max_len = target_length + tolerance

        for s in sentences:
            s_len = len(s)
            if size > 0 and size + s_len > max_len:
                chunks.append(''.join(buf))
                buf, size = [s], s_len
            else:
                buf.append(s)
                size += s_len

        if buf:
            chunks.append(''.join(buf))

        if len(chunks) >= 2 and len(chunks[-1]) < min_len:
            chunks[-2] += chunks[-1]
            chunks.pop()

        return chunks

    def split_oversize_chunk(self, text: str, max_len: int) -> List[str]:
        """分割過大的文本塊"""
        parts = []
        t = text

        while len(t) > max_len:
            window = t[:max_len]
            cut = max((window.rfind(ch) for ch in "。！？；;.!?"), default=-1)
            if cut == -1:
                cut = max((window.rfind(ch) for ch in "，,、"), default=-1)
            if cut == -1:
                cut = max(window.rfind(" "), window.rfind("\n"))
            if cut == -1 or cut < max_len // 2:
                cut = max_len

            parts.append(t[:cut].strip())
            t = t[cut:].lstrip()

        if t.strip():
            parts.append(t.strip())

        return parts

    def make_fine_chunks(self, text: str, target_len: int, tol: int) -> List[str]:
        """製作精細分塊"""
        min_len = target_len - tol
        max_len = target_len + tol
        sents = self.smart_sentence_split(text)
        chunks = self.smart_chunk_merge_by_chars(sents, target_len, tol)
        final_chunks = []

        for ch in chunks:
            if len(ch) > max_len:
                final_chunks.extend(self.split_oversize_chunk(ch, max_len))
            else:
                final_chunks.append(ch)

        return final_chunks

    def _page_font_sizes(self, page) -> List[float]:
        """獲取頁面字體大小列表"""
        sizes = []
        pd = page.get_text("dict")

        for b in pd.get("blocks", []):
            for ln in b.get("lines", []):
                for sp in ln.get("spans", []):
                    t = sp.get("text", "")
                    if t and t.strip():
                        sizes.append(sp.get("size", 0))

        return sizes or [12.0]

    def _doc_fallback_header_min(self, doc) -> float:
        """計算文檔級別的標題最小字體大小"""
        all_sizes = []
        for p in doc:
            all_sizes.extend(self._page_font_sizes(p))

        arr = np.array(all_sizes, dtype=float) if all_sizes else np.array([12.0])
        p50 = float(np.percentile(arr, 50))
        p90 = float(np.percentile(arr, self.HEADER_HI_PCT))
        return max(p90, p50 * self.HEADER_RATIO_MIN)

    def _two_means_threshold(self, arr, max_iter: int = 25) -> Optional[float]:
        """使用兩均值聚類計算閾值"""
        a = np.asarray(arr, dtype=float)
        a = a[(a > 4) & (a < 200)]
        if a.size < 10:
            return None

        c1 = float(np.percentile(a, 40))
        c2 = float(np.percentile(a, 90))

        for _ in range(max_iter):
            d1 = np.abs(a - c1)
            d2 = np.abs(a - c2)
            m1 = a[d1 <= d2]
            m2 = a[d2 < d1]

            if m1.size == 0 or m2.size == 0:
                return None

            nc1 = float(m1.mean())
            nc2 = float(m2.mean())

            if abs(nc1 - c1) + abs(nc2 - c2) < 1e-3:
                break

            c1, c2 = nc1, nc2

        c1, c2 = sorted([c1, c2])
        return 0.5 * (c1 + c2)

    def _cluster_threshold_for_page(self, page) -> Optional[float]:
        """為單頁計算聚類閾值"""
        return self._two_means_threshold(self._page_font_sizes(page))

    def _page_header_min(self, page, fallback_min: float) -> float:
        """計算頁面標題最小字體大小"""
        arr = np.array(self._page_font_sizes(page), dtype=float)
        p50 = float(np.percentile(arr, 50)) if arr.size else 12.0
        p90 = float(np.percentile(arr, self.HEADER_HI_PCT)) if arr.size else 16.0

        thr_ratio = max(p90, p50 * self.HEADER_RATIO_MIN, float(fallback_min))
        thr_auto = self._cluster_threshold_for_page(page) if self.AUTO_THRESHOLD else None
        thr = max(thr_ratio, (thr_auto or 0.0))

        if self.HEADER_ABS_MIN_PT > 0:
            thr = max(thr, self.HEADER_ABS_MIN_PT)

        return thr

    def _matches_any(self, text: str, patterns: List[str]) -> bool:
        """檢查文本是否匹配任何模式"""
        return any(re.search(p, text) for p in patterns)

    def _looks_like_title(self, text: str, require_numbered: Optional[bool] = None) -> bool:
        """判斷文本是否像標題"""
        t = text.strip()
        if len(t) < 2 or len(t) > 60:
            return False
        if t.replace(" ", "").isdigit():
            return False
        if self.BULLET_PREFIX_RE.match(t):
            return False

        numbered_like = self._matches_any(t, self.HEADER_NUMBER_PATTERNS)
        if (not numbered_like) and t.endswith(self._HEADER_BAD_TRAIL):
            return False

        if require_numbered is None:
            require_numbered = self.NUMBERED_HEADERS_ONLY
        if require_numbered and not numbered_like:
            return False

        return True

    def _is_section_header(self, text: str, size: float, header_min_size: float,
                           page_width: Optional[float] = None, bbox: Optional[Tuple] = None,
                           require_center: bool = False, require_numbered: Optional[bool] = None) -> bool:
        """判斷是否為節標題"""
        if size < header_min_size:
            return False
        if not self._looks_like_title(text, require_numbered=require_numbered):
            return False

        if page_width and bbox and require_center:
            x0, y0, x1, y1 = bbox
            centered = abs(((x0 + x1) / 2) - (page_width / 2)) <= page_width * 0.18
            if not centered:
                return False

        return True

    def _iter_lines_in_reading_order(self, page, cluster_columns: bool = True):
        """按閱讀順序迭代頁面行"""
        pd = page.get_text("dict")
        page_width = page.rect.width
        rows = []

        for b in pd.get("blocks", []):
            for ln in b.get("lines", []):
                spans = [sp for sp in ln.get("spans", []) if sp.get("text", "").strip()]
                if not spans:
                    continue

                text = "".join(sp["text"] for sp in spans).strip()
                max_size = max(sp.get("size", 0) for sp in spans)
                xs0 = [sp["bbox"][0] for sp in spans]
                ys0 = [sp["bbox"][1] for sp in spans]
                xs1 = [sp["bbox"][2] for sp in spans]
                ys1 = [sp["bbox"][3] for sp in spans]
                bbox = (min(xs0), min(ys0), max(xs1), max(ys1))
                y_top = round(bbox[1], 2)
                x_left = round(bbox[0], 2)
                rows.append((y_top, x_left, text, max_size, bbox))

        rows.sort(key=lambda r: (r[0], r[1]))

        if not cluster_columns:
            for _, _, text, max_size, bbox in rows:
                yield text, max_size, bbox, page_width
            return

        # 列聚類處理
        buckets, y_tol = [], page.rect.height * 0.004
        for row in rows:
            for bucket in buckets:
                if abs(bucket[-1][0] - row[0]) <= y_tol:
                    bucket.append(row)
                    break
            else:
                buckets.append([row])

        for bucket in buckets:
            xs = sorted([r[1] for r in bucket])
            gaps = [xs[i] - xs[i - 1] for i in range(1, len(xs))]
            two_col = (gaps and max(gaps) > page_width * 0.15)

            if not two_col:
                cols = [bucket]
            else:
                midx = (min(xs) + max(xs)) / 2
                left = [r for r in bucket if r[1] <= midx]
                right = [r for r in bucket if r[1] > midx]
                cols = [left, right]

            for col in cols:
                for _, _, text, max_size, bbox in sorted(col, key=lambda r: r[1]):
                    yield text, max_size, bbox, page_width

    def _merge_number_badges(self, rows: List[Tuple], page_width: float, page_height: float,
                             y_tol_ratio: float = 0.018, x_gap_ratio: float = 0.02,
                             x_gap_ratio_top: float = 0.38, top_band_ratio: float = 0.22,
                             x_align_ratio: float = 0.03) -> List[Tuple]:
        """合併數字標記"""
        y_tol = page_height * y_tol_ratio
        base_gap = page_width * x_gap_ratio
        top_gap = page_width * x_gap_ratio_top
        x_align = page_width * x_align_ratio

        merged, i = [], 0
        while i < len(rows):
            t1, s1, b1, pw = rows[i]
            x0_1, y0_1, x1_1, y1_1 = b1
            nxt = rows[i + 1] if i + 1 < len(rows) else None

            if nxt:
                t2, s2, b2, _ = nxt
                x0_2, y0_2, x1_2, y1_2 = b2
                same_row = abs(y0_2 - y0_1) <= y_tol
                in_top = min(y0_1, y0_2) <= page_height * top_band_ratio
                gap_ok = (x0_2 >= (x1_1 - (top_gap if in_top else base_gap)))
                x_aligned = abs(x0_2 - x0_1) <= x_align

                if re.match(self.NUMBER_TOKEN_PATTERN, t1.strip()) and same_row and gap_ok:
                    combined_text = (t1.strip() + " " + t2.strip()).strip()
                    max_size = max(s1, s2)
                    union_bbox = (min(x0_1, x0_2), min(y0_1, y0_2), max(x1_1, x1_2), max(y1_1, y1_2))
                    merged.append((combined_text, max_size, union_bbox, page_width))
                    i += 2
                    continue

                below = (y0_2 > y0_1) and (y0_2 - y0_1 <= y_tol * 1.3) and x_aligned
                if re.match(self.NUMBER_TOKEN_PATTERN, t1.strip()) and below:
                    combined_text = (t1.strip() + " " + t2.strip()).strip()
                    max_size = max(s1, s2)
                    union_bbox = (min(x0_1, x0_2), min(y0_1, y0_2), max(x1_1, x1_2), max(y1_1, y1_2))
                    merged.append((combined_text, max_size, union_bbox, page_width))
                    i += 2
                    continue

                if re.match(self.NUMBER_TOKEN_PATTERN, t2.strip()) and same_row and (
                        (x0_2 - x1_1) <= (top_gap if in_top else base_gap)):
                    combined_text = (t2.strip() + " " + t1.strip()).strip()
                    max_size = max(s1, s2)
                    union_bbox = (min(x0_1, x0_2), min(y0_1, y0_2), max(x1_1, x1_2), max(y1_1, y1_2))
                    merged.append((combined_text, max_size, union_bbox, page_width))
                    i += 2
                    continue

            merged.append((t1, s1, b1, page_width))
            i += 1

        return merged

    def _repeat_canon(self, t: str) -> str:
        """標準化重複內容檢測"""
        s = re.sub(r'\s+', '', t).lower()
        s = s.strip('-—–·•—─=~:。．、.,')
        if not s:
            return s

        if ('www.' in s) or ('http://' in s) or ('https://' in s) \
                or re.search(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", s) \
                or re.search(r"\.[a-z]{2,4}\b", s):
            s = re.sub(r"\d+", "#", s)

        s = re.sub(r"([—–\-·•\s]*p\.?)?\s*[-—–·•]?\s*\d+(?:\s*[-—–]\s*\d+)*\s*$", "", s)

        if len(s) <= 8:
            digit_ratio = sum(ch.isdigit() for ch in s) / max(1, len(s))
            if digit_ratio >= 0.6:
                s = re.sub(r"\d+", "#", s)

        return s

    def _looks_like_body_text(self, t: str) -> bool:
        """判斷是否像正文文本"""
        s = re.sub(r'\s+', ' ', t).strip()
        if not s:
            return False
        if any(ch in self._BODY_PUNCT for ch in s):
            return True

        chinese_ratio = sum('\u4e00' <= ch <= '\u9fff' for ch in s) / max(1, len(s))
        if len(s) >= 20 and chinese_ratio >= 0.30:
            return True
        if (' ' in s) and (chinese_ratio >= 0.30):
            return True

        return False

    def _top_or_bottom(self, page, bbox) -> Optional[str]:
        """判斷位置是否在頁面頂部或底部"""
        y0 = bbox[1]
        h = page.rect.height
        if y0 <= h * self.MARGIN_RATIO:
            return 'top'
        if y0 >= h * (1 - self.MARGIN_RATIO):
            return 'bottom'
        return None

    def _small_font_thr_for_page(self, page) -> float:
        """計算頁面小字體閾值"""
        arr = np.array(self._page_font_sizes(page), dtype=float)
        if arr.size == 0:
            return 12.0
        p20 = float(np.percentile(arr, self.SMALL_FONT_PCT))
        p50 = float(np.percentile(arr, 50))
        return min(p20, p50 - 1.0)

    def _is_probable_page_number(self, page, text: str, size: float, bbox: Tuple) -> bool:
        """判斷是否可能是頁碼"""
        t = text.strip()
        if not t:
            return False

        h = page.rect.height
        y0 = bbox[1]
        if not (y0 <= h * self.PAGENUM_MARGIN or y0 >= h * (1 - self.PAGENUM_MARGIN)):
            return False

        w = page.rect.width
        x0, x1 = bbox[0], bbox[2]
        cx = (x0 + x1) / 2.0
        in_center = (abs(cx - w / 2.0) <= w * self.PAGENUM_X_CENTER_TOL)
        in_left = (x0 <= w * self.PAGENUM_X_EDGE_BAND)
        in_right = (x1 >= w * (1 - self.PAGENUM_X_EDGE_BAND))

        if not (in_center or in_left or in_right):
            return False

        compact = re.sub(r'\s+', '', t)
        has_cjk_or_alpha = re.search(r"[A-Za-z\u4e00-\u9fff]", compact) is not None
        special_ok = re.search(r"^(?:p(?:age|\.)?\d+|第[一二三四五六七八九十百千零〇0-9]+页)$", compact,
                               re.I) is not None

        if has_cjk_or_alpha and not special_ok:
            return False

        looks_regex = bool(self.PAGE_NUM_RE.match(t))
        allowed = set("0123456789ivxlcdmIVXLCDM-—–/· .")
        compact2 = compact.replace('·', '.')
        strong_like = (len(compact2) <= 7 and all(ch in allowed for ch in compact2))
        small_thr = self._small_font_thr_for_page(page)
        is_small = (size <= small_thr + 0.2)

        return (looks_regex or strong_like) and (is_small or strong_like)

    def _is_running_header_numbered(self, page, text: str, size: float, bbox: Tuple,
                                    header_min_for_page: float) -> bool:
        """判斷是否為帶編號的頁眉"""
        t = text.strip()
        if not t:
            return False

        y0 = bbox[1]
        if y0 > page.rect.height * self.TOP_HEADER_BAND_RATIO:
            return False

        has_number_token = bool(re.search(self.NUMBER_TOKEN_PATTERN, t))
        has_word = bool(re.search(r"[A-Za-z\u4e00-\u9fff]", t))

        if not (has_number_token and has_word):
            return False
        if size >= header_min_for_page * self.TOP_HEADER_NEAR_HEADER_THR:
            return False

        return True

    def _is_probable_running_header_strong(self, page, text: str, size: float, bbox: Tuple,
                                           header_min_for_page: float) -> bool:
        """判斷是否為強樣式頁眉"""
        t = text.strip()
        if not t:
            return False
        if self.BULLET_PREFIX_RE.match(t):
            return False
        if self._looks_like_body_text(t):
            return False

        h = page.rect.height
        y0 = bbox[1]
        if y0 > h * self.TOP_HEADER_BAND_RATIO:
            return False

        w = page.rect.width
        x0, x1 = bbox[0], bbox[2]
        cx = (x0 + x1) / 2.0
        in_left = cx <= w * 0.33
        in_right = cx >= w * 0.67

        if not (in_left or in_right):
            return False
        if len(t) > 24:
            return False
        if not re.search(r"\d", t):
            return False
        if not re.search(r"[A-Za-z\u4e00-\u9fff]", t):
            return False
        if size >= header_min_for_page * 1.40:
            return False

        return True

    def _is_badge_fragment(self, page, text: str, size: float, bbox: Tuple) -> bool:
        """判斷是否為徽章片段"""
        _BADGE_FRAGMENT_RE = re.compile(r"^\s*\d+\s*[-—–]\s*$")
        t = text.strip()
        if not t:
            return False

        y0 = bbox[1]
        if y0 > page.rect.height * self.TOP_HEADER_BAND_RATIO:
            return False

        return bool(_BADGE_FRAGMENT_RE.match(t))

    def _veto_topband_header(self, page, text: str, size: float, bbox: Tuple, header_min_for_page: float) -> bool:
        """否決頂部標題"""
        if not bbox:
            return False

        t = text.strip()
        if not t:
            return False

        h, w = page.rect.height, page.rect.width
        y0 = bbox[1]
        cx = (bbox[0] + bbox[2]) / 2.0
        in_top = (y0 <= h * self.TOP_HEADER_BAND_RATIO)
        left_or_right = (cx <= w * 0.35) or (cx >= w * 0.65)

        if not (in_top and left_or_right):
            return False
        if size >= header_min_for_page * 1.10:
            return False

        short = len(t) <= 24
        has_url = ('www.' in t.lower()) or ('http://' in t.lower()) or ('https://' in t.lower()) \
                  or re.search(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", t.lower()) \
                  or re.search(r"\.[a-z]{2,4}\b", t.lower())
        like_badge = bool(re.match(r"^\s*\d+\s*[-—–]\s*\d+\b", t))
        like_num_title = bool(re.match(r"^\s*\d{1,4}\s+[A-Za-z\u4e00-\u9fff]", t))

        if short and (has_url or like_badge or like_num_title):
            return True
        if self._is_running_header_numbered(page, text, size, bbox, header_min_for_page):
            return True
        if self._is_probable_running_header_strong(page, text, size, bbox, header_min_for_page):
            return True
        if self._is_badge_fragment(page, text, size, bbox):
            return True

        return False

    def _collect_repetitives(self, doc) -> set:
        """收集重複內容"""
        key_count = defaultdict(int)

        for page in doc:
            rows = list(self._iter_lines_in_reading_order(page, cluster_columns=False))
            rows = self._merge_number_badges(rows, page.rect.width, page.rect.height)
            seen = set()

            for text, size, bbox, _ in rows:
                if not text:
                    continue
                pos = self._top_or_bottom(page, bbox)
                if not pos:
                    continue

                canon = self._repeat_canon(text)
                if len(canon) < self.REPEAT_MIN_LEN:
                    continue

                k = (canon, pos)
                if k not in seen:
                    key_count[k] += 1
                    seen.add(k)

        return {k for k, c in key_count.items() if c >= self.REPETITIVE_MIN_PAGES}

    def _iter_candidate_lines_for_doc(self, doc):
        """為文檔迭代候選行"""
        for page_idx, page in enumerate(doc, start=1):
            rows = list(self._iter_lines_in_reading_order(page, cluster_columns=True))
            rows = self._merge_number_badges(rows, page.rect.width, page.rect.height)

            for text, size, bbox, pw in rows:
                t = text.strip()
                if not t:
                    continue
                if t.replace(" ", "").isdigit():
                    continue
                if self.BULLET_PREFIX_RE.match(t):
                    continue
                if t.endswith(self._HEADER_BAD_TRAIL):
                    continue

                yield page_idx, float(size), t, bbox, pw

    def _auto_tune_doc_threshold(self, doc, start_thr: float) -> float:
        """自動調整文檔閾值"""
        if not self.AUTO_TUNE_DOC:
            return start_thr

        thr = float(start_thr)
        for _ in range(self.TUNE_MAX_STEPS):
            counts = []
            cur_page = 1
            cnt = 0

            for page_idx, size, t, bbox, pw in self._iter_candidate_lines_for_doc(doc):
                if page_idx != cur_page:
                    counts.append(cnt)
                    cnt = 0
                    cur_page = page_idx
                if size >= thr:
                    cnt += 1
            counts.append(cnt)

            avg = (np.median(counts) if counts else 0.0)
            if avg < self.TUNE_TARGET_RANGE[0]:
                thr *= self.TUNE_STEP_DOWN
            elif avg > self.TUNE_TARGET_RANGE[1]:
                thr *= self.TUNE_STEP_UP
            else:
                break

        return thr

    def _consolidate_small_chunks(self, chunks: List[str], min_len: int, max_len: int,
                                  overshoot: float = 0.0, tail_overshoot: float = 0.35) -> List[str]:
        """合併小塊文本"""
        if not chunks:
            return chunks[:]

        hard_max_general = int(max_len * (1.0 + max(0.0, overshoot)))
        hard_max_tail = int(max_len * (1.0 + max(overshoot, tail_overshoot)))

        out = []
        i = 0
        n = len(chunks)

        while i < n:
            cur = chunks[i]
            cur_len = len(cur)

            if cur_len < min_len:
                has_next = (i + 1 < n)
                is_tail = (i == n - 1)
                is_head = (i == 0)

                if has_next and cur_len + len(chunks[i + 1]) <= hard_max_general:
                    chunks[i + 1] = cur + chunks[i + 1]
                elif out:
                    prev_len = len(out[-1])
                    cap = hard_max_tail if (is_tail or is_head) else hard_max_general
                    if prev_len + cur_len <= cap:
                        out[-1] = out[-1] + cur
                    else:
                        out.append(cur)
                else:
                    out.append(cur)
            else:
                out.append(cur)
            i += 1

        if len(out) >= 2 and len(out[0]) < min_len and len(out[0]) + len(out[1]) <= hard_max_tail:
            out[1] = out[0] + out[1]
            del out[0]

        return out

    def chunk_pdf_full_page(self, pdf_path: str, target_len: int, tol: int) -> List[Document]:
        """完整頁面PDF分塊處理"""
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            raise RuntimeError(f"無法打開PDF文件 {pdf_path}: {e}")

        try:
            # 收集重複內容
            repetitives = self._collect_repetitives(doc)
            fallback_min = self._doc_fallback_header_min(doc)

            # 計算文檔級別閾值
            all_sizes = []
            for p in doc:
                all_sizes.extend(self._page_font_sizes(p))

            arr = np.array(all_sizes, dtype=float) if all_sizes else np.array([12.0])
            doc_p50 = float(np.percentile(arr, 50)) if arr.size else 12.0
            doc_p90 = float(np.percentile(arr, self.HEADER_HI_PCT)) if arr.size else 16.0
            doc_start = max(doc_p90, doc_p50 * self.HEADER_RATIO_MIN, float(fallback_min))
            doc_auto = self._two_means_threshold(arr) or 0.0
            doc_thr_start = max(doc_start, doc_auto)

            if self.HEADER_ABS_MIN_PT > 0:
                doc_thr_start = max(doc_thr_start, float(self.HEADER_ABS_MIN_PT))

            doc_thr = self._auto_tune_doc_threshold(doc, doc_thr_start)

            # 處理各頁面
            coarse_sections, fine_chunks = [], []
            current_section = "前言"
            current_page = 1
            buffer = ""
            header_hits = 0

            for page_idx, page in enumerate(doc, start=1):
                page_thr = self._page_header_min(page, fallback_min)

                if self.THRESHOLD_MODE == "doc":
                    header_min = doc_thr
                elif self.THRESHOLD_MODE == "hybrid":
                    header_min = max(page_thr, doc_thr)
                else:
                    header_min = page_thr

                rows = list(self._iter_lines_in_reading_order(page, cluster_columns=True))
                rows = self._merge_number_badges(rows, page.rect.width, page.rect.height)

                for line_text, max_size, bbox, page_width in rows:
                    if not line_text.strip():
                        continue

                    # 過濾各種噪音內容
                    if self._is_badge_fragment(page, line_text, max_size, bbox):
                        continue

                    if self.DROP_SMALLEST_PAGENUM and self._is_probable_page_number(page, line_text, max_size, bbox):
                        continue

                    if self._is_running_header_numbered(page, line_text, max_size, bbox, header_min):
                        continue

                    if self._is_probable_running_header_strong(page, line_text, max_size, bbox, header_min):
                        continue

                    # 處理重複頁眉/頁腳
                    pos = self._top_or_bottom(page, bbox)
                    if pos is not None:
                        key = self._repeat_canon(line_text)
                        if len(key) >= self.REPEAT_MIN_LEN and (key, pos) in repetitives:
                            continue

                    # 判斷是否為標題
                    ok = self._is_section_header(
                        line_text, max_size, header_min, page_width, bbox,
                        require_center=self.CENTER_ONLY, require_numbered=self.NUMBERED_HEADERS_ONLY
                    )

                    if ok and self._veto_topband_header(page, line_text, max_size, bbox, header_min):
                        ok = False

                    if ok:
                        if buffer.strip():
                            coarse_sections.append((current_page, current_section, self.clean_spaces(buffer)))
                            buffer = ""
                        current_section = line_text.strip()
                        current_page = page_idx
                        header_hits += 1
                    else:
                        buffer = self.soft_join(buffer, line_text)

            if buffer.strip():
                coarse_sections.append((current_page, current_section, self.clean_spaces(buffer)))

            # 保底方案：按頁面分割
            if header_hits == 0 and self.PAGE_FALLBACK_WHEN_NO_HEADER:
                coarse_sections = []
                for page_idx, page in enumerate(doc, start=1):
                    full_text = self.normalize_text(page.get_text())
                    if full_text.strip():
                        coarse_sections.append((page_idx, f"第{page_idx}頁", full_text.strip()))

            # 細切分塊處理
            min_len = target_len - tol
            max_len = target_len + tol

            for page_no, section_title, content in coarse_sections:
                if min_len <= len(content) <= max_len:
                    fine_chunks.append((page_no, section_title, content))
                    continue

                sents = self.smart_sentence_split(content)
                if not sents:
                    fine_chunks.append((page_no, section_title, content))
                    continue

                # 先切分
                sec_chunks = []
                for ch in self.smart_chunk_merge_by_chars(sents, target_len, tol):
                    if len(ch) > max_len:
                        sec_chunks.extend(self.split_oversize_chunk(ch, max_len))
                    else:
                        sec_chunks.append(ch)

                # 智能合併過短塊
                if self.SMART_CONSOLIDATE and sec_chunks:
                    merge_floor = max(min_len, self.MERGE_MIN_ABS)
                    need_merge = any(len(c) < min_len for c in sec_chunks)
                    if need_merge:
                        sec_chunks = self._consolidate_small_chunks(
                            sec_chunks, min_len=merge_floor, max_len=max_len,
                            overshoot=self.ALLOW_OVERSHOOT, tail_overshoot=self.TAIL_OVERSHOOT
                        )

                for ch in sec_chunks:
                    fine_chunks.append((page_no, section_title, ch))

            # 轉換為 Document 對象
            docs = []
            for pg, sec, txt in fine_chunks:
                if txt.strip():
                    docs.append(Document(
                        page_content=txt,
                        metadata={"source": pdf_path, "page": pg, "section": sec}
                    ))

            return docs

        finally:
            doc.close()

    def process_multiple_pdfs(self, pdf_paths: List[str], target_len: int = 400, tol: int = 200) -> List[Document]:
        """處理多個PDF文件"""
        all_docs = []

        for pdf_path in pdf_paths:
            try:
                docs = self.chunk_pdf_full_page(pdf_path, target_len, tol)
                all_docs.extend(docs)
                print(f"✓ 已處理 {pdf_path}: {len(docs)} 個分塊")
            except Exception as e:
                print(f"✗ 處理 {pdf_path} 時發生錯誤: {e}")
                continue

        return all_docs

    def get_statistics(self, docs: List[Document]) -> Dict[str, Any]:
        """獲取文檔統計信息"""
        if not docs:
            return {"總文檔數": 0}

        chunk_lengths = [len(doc.page_content) for doc in docs]
        sources = set(doc.metadata.get("source", "未知") for doc in docs)
        sections = set(doc.metadata.get("section", "未知") for doc in docs)

        return {
            "總文檔數": len(docs),
            "來源文件數": len(sources),
            "章節數": len(sections),
            "平均長度": int(np.mean(chunk_lengths)) if chunk_lengths else 0,
            "最短長度": min(chunk_lengths) if chunk_lengths else 0,
            "最長長度": max(chunk_lengths) if chunk_lengths else 0,
            "長度標準差": int(np.std(chunk_lengths)) if chunk_lengths else 0,
            "來源文件": list(sources)[:5],  # 顯示前5個
        }


"""
# 使用示例
if __name__ == "__main__":
    # 初始化分割器
    splitter = SplitHelper()

    # 可調參數示例
    splitter.CENTER_ONLY = False  # 不要求標題居中
    splitter.NUMBERED_HEADERS_ONLY = False  # 不要求標題有編號
    splitter.SMART_CONSOLIDATE = True  # 啟用智能合併

    # 處理單個PDF
    try:
        docs = splitter.chunk_pdf_full_page("example.pdf", target_len=400, tol=200)
        stats = splitter.get_statistics(docs)
        print("處理統計:", stats)
    except Exception as e:
        print(f"處理失敗: {e}")

    # 處理多個PDF
    pdf_list = ["file1.pdf", "file2.pdf", "file3.pdf"]
    all_docs = splitter.process_multiple_pdfs(pdf_list)
    print(f"總共處理了 {len(all_docs)} 個文檔分塊")
"""