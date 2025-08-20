#import asyncio
import glob  # 用來找多個檔案
import os
from pathlib import Path
#langchain 相關套件
from langchain.text_splitter import RecursiveCharacterTextSplitter  #切割文字
from langchain_community.vectorstores import FAISS                  # FAISS : Facebook 開發的向量資料庫，用來做快速相似度搜尋。
from langchain_openai import OpenAIEmbeddings, ChatOpenAI           # embeddings 用來將文字轉換成向量
from langchain.chains import create_retrieval_chain                 #建立 RAG 架構中的「檢索＋問答」流程。
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

#可以讀取不同的檔案格式
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
)
# =========================
# PDF 智慧切割
# =========================
import re
import fitz  # PyMuPDF

# —— 可調參數 ——
TARGET_LEN = 500
TOLERANCE  = 100
CENTER_ONLY = False
NUMBERED_HEADERS_ONLY = False

HEADER_HI_PCT    = 90
HEADER_RATIO_MIN = 1.5
REPETITIVE_MIN_PAGES = 3
MARGIN_RATIO        = 0.15

THRESHOLD_MODE    = "hybrid"   # "page" / "doc" / "hybrid"
AUTO_THRESHOLD    = True
HEADER_ABS_MIN_PT = 16.0

AUTO_TUNE_DOC       = True
TUNE_TARGET_RANGE   = (0.8, 2.2)
TUNE_MAX_STEPS      = 8
TUNE_STEP_UP        = 1.08
TUNE_STEP_DOWN      = 0.92

PAGE_FALLBACK_WHEN_NO_HEADER = True

DROP_SMALLEST_PAGENUM = True
SMALL_FONT_PCT = 20
PAGENUM_MARGIN = 0.15

PAGENUM_X_CENTER = True
PAGENUM_X_LEFT   = True
PAGENUM_X_RIGHT  = True

PAGENUM_X_CENTER_TOL = 0.25
PAGENUM_X_EDGE_BAND  = 0.18

PAGENUM_MAX_LEN = 12
PAGENUM_DIGIT_RATIO = 0.5
PAGE_NUM_RE = re.compile(r"""
^\s*(
    第\s*[一二三四五六七八九十百千零〇0-9]+\s*頁         |
    p(?:age|\.)?\s*\d+(?:\s*/\s*\d+)?                       |
    [ivxlcdm]+\s*(?:/\s*[ivxlcdm]+)?                        |
    \d+\s*/\s*\d+                                           |
    [–—-]?\s*\d+\s*[–—-]?                                   |
    \d+\s*[–—-]\s*\d+                                       |
    \d{1,3}
)\s*$
""", re.I | re.X)

TOP_HEADER_BAND_RATIO      = 0.22
TOP_HEADER_NEAR_HEADER_THR = 1.30

BULLET_PREFIX_RE = re.compile(r"^\s*[■●◆◼▪◦•‣∙·]\s*")
SEP_CLASS = r"[-–—\-－\.．·]"

HEADER_NUMBER_PATTERNS = [
    rf"^\s*\d+\s*{SEP_CLASS}\s*\d+(?:\s*{SEP_CLASS}\s*\d+)*\s*.+$",
    r"^\s*\d{1,4}\s+.+$",
    r"^\s*第\s*[一二三四五六七八九十百千零〇0-9]+\s*[章節篇]\s+.+$",
]
NUMBER_TOKEN_PATTERN = rf"^\s*(?:\d+\s*{SEP_CLASS}\s*\d+(?:\s*{SEP_CLASS}\s*\d+)*|\d{{1,4}})\s*$"
_HEADER_BAD_TRAIL = tuple("，,。：:；;！!？?．.、")

REPEAT_MIN_LEN = 3

SMART_CONSOLIDATE = True
MERGE_TRIGGER_GAP  = 200
MERGE_MIN_RATIO    = 0.5
MERGE_MIN_ABS      = 180
ALLOW_OVERSHOOT    = 0.02
TAIL_OVERSHOOT     = 0.1

RIGHT_PUNCT = set("，,。．.！？!?：:；;、）)]}〉》％℃…’”」』")
LEFT_PUNCT  = set("（([ {〈《“‘「『")

def soft_join(buf: str, s: str) -> str:
    s = s.strip()
    if not buf: return s
    if not s:   return buf
    last = buf[-1]; first = s[0]
    if last.isspace() or last in LEFT_PUNCT: return buf + s
    if first in RIGHT_PUNCT:                return buf + s
    return buf + " " + s

def clean_spaces(s: str) -> str:
    s = s.replace("\u3000", " ").replace("\xa0", " ")
    return re.sub(r"[ \t]{2,}", " ", s).strip()

def normalize_text(s: str) -> str:
    s = s.replace('\r\n', '\n').replace('\r', '\n')
    s = re.sub(r'\n{2,}', '\n\n', s)
    s = s.replace('\n', ' ')
    s = re.sub(r'[ \t\u3000]+', ' ', s)
    s = re.sub(r'\s{2,}', ' ', s)
    return s.strip()

def smart_sentence_split(text: str):
    pat = re.compile(r'[^。！？!?；;]*[。！？!?；;]')
    sentences = pat.findall(text)
    used = sum(len(s) for s in sentences)
    tail = text[used:].strip()
    if tail: sentences.append(tail)
    return [s.strip() for s in sentences if s.strip()]

def smart_chunk_merge_by_chars(sentences, target_length=400, tolerance=200):
    chunks, buf, size = [], [], 0
    min_len = target_length - tolerance
    max_len = target_length + tolerance
    for s in sentences:
        s_len = len(s)
        if size > 0 and size + s_len > max_len:
            chunks.append(''.join(buf)); buf, size = [s], s_len
        else:
            buf.append(s); size += s_len
    if buf: chunks.append(''.join(buf))
    if len(chunks) >= 2 and len(chunks[-1]) < min_len:
        chunks[-2] += chunks[-1]; chunks.pop()
    return chunks

def split_oversize_chunk(text, max_len):
    parts = []; t = text
    while len(t) > max_len:
        window = t[:max_len]
        cut = max((window.rfind(ch) for ch in "。！？；;.!?"), default=-1)
        if cut == -1: cut = max((window.rfind(ch) for ch in "，,、"), default=-1)
        if cut == -1: cut = max(window.rfind(" "), window.rfind("\n"))
        if cut == -1 or cut < max_len // 2: cut = max_len
        parts.append(t[:cut].strip()); t = t[cut:].lstrip()
    if t.strip(): parts.append(t.strip())
    return parts

def make_fine_chunks(text: str, target_len: int, tol: int):
    min_len = target_len - tol; max_len = target_len + tol
    sents = smart_sentence_split(text)
    chunks = smart_chunk_merge_by_chars(sents, target_len, tol)
    final_chunks = []
    for ch in chunks:
        if len(ch) > max_len:
            final_chunks.extend(split_oversize_chunk(ch, max_len))
        else:
            final_chunks.append(ch)
    return final_chunks

def _page_font_sizes(page):
    sizes = []
    pd = page.get_text("dict")
    for b in pd.get("blocks", []):
        for ln in b.get("lines", []):
            for sp in ln.get("spans", []):
                t = sp.get("text", "")
                if t and t.strip():
                    sizes.append(sp.get("size", 0))
    return sizes or [12.0]

def _doc_fallback_header_min(doc):
    all_sizes = []
    for p in doc: all_sizes.extend(_page_font_sizes(p))
    arr = np.array(all_sizes, dtype=float) if all_sizes else np.array([12.0])
    p50 = float(np.percentile(arr, 50)); p90 = float(np.percentile(arr, HEADER_HI_PCT))
    return max(p90, p50 * HEADER_RATIO_MIN)

def _two_means_threshold(arr, max_iter=25):
    a = np.asarray(arr, dtype=float); a = a[(a > 4) & (a < 200)]
    if a.size < 10: return None
    c1 = float(np.percentile(a, 40)); c2 = float(np.percentile(a, 90))
    for _ in range(max_iter):
        d1 = np.abs(a - c1); d2 = np.abs(a - c2)
        m1 = a[d1 <= d2]; m2 = a[d2 < d1]
        if m1.size == 0 or m2.size == 0: return None
        nc1 = float(m1.mean()); nc2 = float(m2.mean())
        if abs(nc1-c1) + abs(nc2-c2) < 1e-3: break
        c1, c2 = nc1, nc2
    c1, c2 = sorted([c1, c2])
    return 0.5 * (c1 + c2)

def _cluster_threshold_for_page(page): return _two_means_threshold(_page_font_sizes(page))

def _page_header_min(page, fallback_min):
    arr = np.array(_page_font_sizes(page), dtype=float)
    p50 = float(np.percentile(arr, 50)) if arr.size else 12.0
    p90 = float(np.percentile(arr, HEADER_HI_PCT)) if arr.size else 16.0
    thr_ratio = max(p90, p50 * HEADER_RATIO_MIN, float(fallback_min))
    thr_auto  = _cluster_threshold_for_page(page) if AUTO_THRESHOLD else None
    thr = max(thr_ratio, (thr_auto or 0.0))
    if HEADER_ABS_MIN_PT > 0: thr = max(thr, HEADER_ABS_MIN_PT)
    return thr

def _matches_any(text: str, patterns) -> bool:
    return any(re.search(p, text) for p in patterns)

def _looks_like_title(text: str, require_numbered: bool | None = None) -> bool:
    t = text.strip()
    if len(t) < 2 or len(t) > 60: return False
    if t.replace(" ", "").isdigit(): return False
    if BULLET_PREFIX_RE.match(t):    return False
    numbered_like = _matches_any(t, HEADER_NUMBER_PATTERNS)
    if (not numbered_like) and t.endswith(_HEADER_BAD_TRAIL): return False
    if require_numbered is None: require_numbered = NUMBERED_HEADERS_ONLY
    if require_numbered and not numbered_like: return False
    return True

def _is_section_header(text, size, header_min_size, page_width=None, bbox=None,
                       require_center=False, require_numbered: bool | None = None):
    if size < header_min_size: return False
    if not _looks_like_title(text, require_numbered=require_numbered): return False
    if page_width and bbox and require_center:
        x0,y0,x1,y1 = bbox
        centered = abs(((x0+x1)/2) - (page_width/2)) <= page_width * 0.18
        if not centered: return False
    return True

def _iter_lines_in_reading_order(page, cluster_columns=True):
    pd = page.get_text("dict"); page_width = page.rect.width
    rows = []
    for b in pd.get("blocks", []):
        for ln in b.get("lines", []):
            spans = [sp for sp in ln.get("spans", []) if sp.get("text", "").strip()]
            if not spans: continue
            text = "".join(sp["text"] for sp in spans).strip()
            max_size = max(sp.get("size", 0) for sp in spans)
            xs0 = [sp["bbox"][0] for sp in spans]; ys0 = [sp["bbox"][1] for sp in spans]
            xs1 = [sp["bbox"][2] for sp in spans]; ys1 = [sp["bbox"][3] for sp in spans]
            bbox = (min(xs0), min(ys0), max(xs1), max(ys1))
            y_top = round(bbox[1], 2); x_left = round(bbox[0], 2)
            rows.append((y_top, x_left, text, max_size, bbox))
    rows.sort(key=lambda r: (r[0], r[1]))
    if not cluster_columns:
        for _, _, text, max_size, bbox in rows: yield text, max_size, bbox, page_width
        return
    buckets, y_tol = [], page.rect.height * 0.004
    for row in rows:
        for bucket in buckets:
            if abs(bucket[-1][0] - row[0]) <= y_tol: bucket.append(row); break
        else:
            buckets.append([row])
    for bucket in buckets:
        xs = sorted([r[1] for r in bucket]); gaps = [xs[i]-xs[i-1] for i in range(1, len(xs))]
        two_col = (gaps and max(gaps) > page_width * 0.15)
        if not two_col: cols = [bucket]
        else:
            midx = (min(xs)+max(xs))/2
            left  = [r for r in bucket if r[1] <= midx]
            right = [r for r in bucket if r[1] >  midx]
            cols = [left, right]
        for col in cols:
            for _, _, text, max_size, bbox in sorted(col, key=lambda r: r[1]):
                yield text, max_size, bbox, page_width

def _merge_number_badges(
    rows, page_width, page_height,
    y_tol_ratio=0.018, x_gap_ratio=0.02,
    x_gap_ratio_top=0.38, top_band_ratio=0.22, x_align_ratio=0.03
):
    y_tol = page_height * y_tol_ratio
    base_gap = page_width * x_gap_ratio
    top_gap  = page_width * x_gap_ratio_top
    x_align  = page_width * x_align_ratio
    merged, i = [], 0
    while i < len(rows):
        t1, s1, b1, pw = rows[i]; x0_1,y0_1,x1_1,y1_1 = b1
        nxt = rows[i+1] if i+1 < len(rows) else None
        if nxt:
            t2, s2, b2, _ = nxt; x0_2,y0_2,x1_2,y1_2 = b2
            same_row = abs(y0_2 - y0_1) <= y_tol
            in_top = min(y0_1, y0_2) <= page_height * top_band_ratio
            gap_ok = (x0_2 >= (x1_1 - (top_gap if in_top else base_gap)))
            x_aligned = abs(x0_2 - x0_1) <= x_align
            if re.match(NUMBER_TOKEN_PATTERN, t1.strip()) and same_row and gap_ok:
                combined_text = (t1.strip() + " " + t2.strip()).strip()
                max_size = max(s1, s2)
                union_bbox = (min(x0_1,x0_2), min(y0_1,y0_2), max(x1_1,x1_2), max(y1_1,y1_2))
                merged.append((combined_text, max_size, union_bbox, page_width)); i += 2; continue
            below = (y0_2 > y0_1) and (y0_2 - y0_1 <= y_tol*1.3) and x_aligned
            if re.match(NUMBER_TOKEN_PATTERN, t1.strip()) and below:
                combined_text = (t1.strip() + " " + t2.strip()).strip()
                max_size = max(s1, s2)
                union_bbox = (min(x0_1,x0_2), min(y0_1,y0_2), max(x1_1,x1_2), max(y1_1,y1_2))
                merged.append((combined_text, max_size, union_bbox, page_width)); i += 2; continue
            if re.match(NUMBER_TOKEN_PATTERN, t2.strip()) and same_row and ((x0_2 - x1_1) <= (top_gap if in_top else base_gap)):
                combined_text = (t2.strip() + " " + t1.strip()).strip()
                max_size = max(s1, s2)
                union_bbox = (min(x0_1,x0_2), min(y0_1,y0_2), max(x1_1,x1_2), max(y1_1,y1_2))
                merged.append((combined_text, max_size, union_bbox, page_width)); i += 2; continue
        merged.append((t1, s1, b1, page_width)); i += 1
    return merged

def _repeat_canon(t: str) -> str:
    s = re.sub(r'\s+', '', t).lower()
    s = s.strip('-–—·•—─=~:。．、.,')
    if not s:
        return s
    if ('www.' in s) or ('http://' in s) or ('https://' in s) \
       or re.search(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", s) \
       or re.search(r"\.[a-z]{2,4}\b", s):
        s = re.sub(r"\d+", "#", s)
    s = re.sub(r"([–—\-·•\s]*p\.?)?\s*[-–—·•]?\s*\d+(?:\s*[-–—]\s*\d+)*\s*$", "", s)
    if len(s) <= 8:
        digit_ratio = sum(ch.isdigit() for ch in s) / max(1, len(s))
        if digit_ratio >= 0.6:
            s = re.sub(r"\d+", "#", s)
    return s

_BODY_PUNCT = set("。！？；，、,.!?;:")

def _looks_like_body_text(t: str) -> bool:
    s = re.sub(r'\s+', ' ', t).strip()
    if not s: return False
    if any(ch in _BODY_PUNCT for ch in s): return True
    chinese_ratio = sum('\u4e00' <= ch <= '\u9fff' for ch in s) / max(1, len(s))
    if len(s) >= 20 and chinese_ratio >= 0.30: return True
    if (' ' in s) and (chinese_ratio >= 0.30): return True
    return False

def _top_or_bottom(page, bbox):
    y0 = bbox[1]; h = page.rect.height
    if y0 <= h * MARGIN_RATIO: return 'top'
    if y0 >= h * (1 - MARGIN_RATIO): return 'bottom'
    return None

def _small_font_thr_for_page(page) -> float:
    arr = np.array(_page_font_sizes(page), dtype=float)
    if arr.size == 0: return 12.0
    p20 = float(np.percentile(arr, SMALL_FONT_PCT))
    p50 = float(np.percentile(arr, 50))
    return min(p20, p50 - 1.0)

def _is_probable_page_number(page, text, size, bbox) -> bool:
    t = text.strip()
    if not t: return False
    h = page.rect.height; y0 = bbox[1]
    if not (y0 <= h * PAGENUM_MARGIN or y0 >= h * (1 - PAGENUM_MARGIN)):
        return False
    w = page.rect.width; x0, x1 = bbox[0], bbox[2]
    cx = (x0 + x1) / 2.0
    in_center = (abs(cx - w/2.0) <= w * PAGENUM_X_CENTER_TOL)
    in_left   = (x0 <= w * PAGENUM_X_EDGE_BAND)
    in_right  = (x1 >= w * (1 - PAGENUM_X_EDGE_BAND))
    if not (in_center or in_left or in_right): return False
    compact = re.sub(r'\s+', '', t)
    has_cjk_or_alpha = re.search(r"[A-Za-z\u4e00-\u9fff]", compact) is not None
    special_ok = re.search(r"^(?:p(?:age|\.)?\d+|第[一二三四五六七八九十百千零〇0-9]+頁)$", compact, re.I) is not None
    if has_cjk_or_alpha and not special_ok: return False
    looks_regex = bool(PAGE_NUM_RE.match(t))
    allowed = set("0123456789ivxlcdmIVXLCDM-–—/· .")
    compact2 = compact.replace('·', '.')
    strong_like = (len(compact2) <= 7 and all(ch in allowed for ch in compact2))
    small_thr = _small_font_thr_for_page(page)
    is_small  = (size <= small_thr + 0.2)
    return (looks_regex or strong_like) and (is_small or strong_like)

def _is_running_header_numbered(page, text, size, bbox, header_min_for_page) -> bool:
    t = text.strip()
    if not t: return False
    y0 = bbox[1]
    if y0 > page.rect.height * TOP_HEADER_BAND_RATIO: return False
    has_number_token = bool(re.search(NUMBER_TOKEN_PATTERN, t))
    has_word = bool(re.search(r"[A-Za-z\u4e00-\u9fff]", t))
    if not (has_number_token and has_word): return False
    if size >= header_min_for_page * TOP_HEADER_NEAR_HEADER_THR: return False
    return True

def _is_probable_running_header_strong(page, text, size, bbox, header_min_for_page) -> bool:
    t = text.strip()
    if not t: return False
    if BULLET_PREFIX_RE.match(t): return False
    if _looks_like_body_text(t):  return False
    h = page.rect.height; y0 = bbox[1]
    if y0 > h * TOP_HEADER_BAND_RATIO: return False
    w = page.rect.width; x0, x1 = bbox[0], bbox[2]
    cx = (x0 + x1) / 2.0
    in_left  = cx <= w * 0.33
    in_right = cx >= w * 0.67
    if not (in_left or in_right): return False
    if len(t) > 24: return False
    if not re.search(r"\d", t): return False
    if not re.search(r"[A-Za-z\u4e00-\u9fff]", t): return False
    if size >= header_min_for_page * 1.40: return False
    return True

_BADGE_FRAGMENT_RE = re.compile(r"^\s*\d+\s*[-–—]\s*$")
def _is_badge_fragment(page, text, size, bbox) -> bool:
    t = text.strip()
    if not t: return False
    y0 = bbox[1]
    if y0 > page.rect.height * TOP_HEADER_BAND_RATIO: return False
    return bool(_BADGE_FRAGMENT_RE.match(t))

def _veto_topband_header(page, text, size, bbox, header_min_for_page) -> bool:
    if not bbox: return False
    t = text.strip()
    if not t: return False
    h, w = page.rect.height, page.rect.width
    y0 = bbox[1]
    cx = (bbox[0] + bbox[2]) / 2.0
    in_top = (y0 <= h * TOP_HEADER_BAND_RATIO)
    left_or_right = (cx <= w * 0.35) or (cx >= w * 0.65)
    if not (in_top and left_or_right): return False
    if size >= header_min_for_page * 1.10: return False
    short = len(t) <= 24
    has_url = ('www.' in t.lower()) or ('http://' in t.lower()) or ('https://' in t.lower()) \
              or re.search(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", t.lower()) \
              or re.search(r"\.[a-z]{2,4}\b", t.lower())
    like_badge = bool(re.match(r"^\s*\d+\s*[-–—]\s*\d+\b", t))
    like_num_title = bool(re.match(r"^\s*\d{1,4}\s+[A-Za-z\u4e00-\u9fff]", t))
    if short and (has_url or like_badge or like_num_title): return True
    if _is_running_header_numbered(page, text, size, bbox, header_min_for_page): return True
    if _is_probable_running_header_strong(page, text, size, bbox, header_min_for_page): return True
    if _is_badge_fragment(page, text, size, bbox): return True
    return False

def _collect_repetitives(doc):
    from collections import defaultdict
    key_count = defaultdict(int)
    for page in doc:
        rows = list(_iter_lines_in_reading_order(page, cluster_columns=False))
        rows = _merge_number_badges(rows, page.rect.width, page.rect.height)
        seen = set()
        for text, size, bbox, _ in rows:
            if not text: continue
            pos = _top_or_bottom(page, bbox)
            if not pos: continue
            canon = _repeat_canon(text)
            if len(canon) < REPEAT_MIN_LEN: continue
            k = (canon, pos)
            if k not in seen:
                key_count[k] += 1
                seen.add(k)
    return {k for k, c in key_count.items() if c >= REPETITIVE_MIN_PAGES}

def _iter_candidate_lines_for_doc(doc):
    for page_idx, page in enumerate(doc, start=1):
        rows = list(_iter_lines_in_reading_order(page, cluster_columns=True))
        rows = _merge_number_badges(rows, page.rect.width, page.rect.height)
        for text, size, bbox, pw in rows:
            t = text.strip()
            if not t: continue
            if t.replace(" ", "").isdigit(): continue
            if BULLET_PREFIX_RE.match(t):    continue
            if t.endswith(_HEADER_BAD_TRAIL): continue
            yield page_idx, float(size), t, bbox, pw

def _auto_tune_doc_threshold(doc, start_thr):
    if not AUTO_TUNE_DOC: return start_thr
    thr = float(start_thr)
    for _ in range(TUNE_MAX_STEPS):
        counts = []; cur_page = 1; cnt = 0
        for page_idx, size, t, bbox, pw in _iter_candidate_lines_for_doc(doc):
            if page_idx != cur_page:
                counts.append(cnt); cnt = 0; cur_page = page_idx
            if size >= thr: cnt += 1
        counts.append(cnt)
        avg = (np.median(counts) if counts else 0.0)
        if avg < TUNE_TARGET_RANGE[0]:   thr *= TUNE_STEP_DOWN
        elif avg > TUNE_TARGET_RANGE[1]: thr *= TUNE_STEP_UP
        else: break
    return thr

def _consolidate_small_chunks(chunks: list[str], min_len: int, max_len: int,
                              overshoot: float = 0.0, tail_overshoot: float = 0.35) -> list[str]:
    if not chunks: return chunks[:]
    hard_max_general = int(max_len * (1.0 + max(0.0, overshoot)))
    hard_max_tail    = int(max_len * (1.0 + max(overshoot, tail_overshoot)))
    out = []
    i = 0; n = len(chunks)
    while i < n:
        cur = chunks[i]; cur_len = len(cur)
        if cur_len < min_len:
            has_next = (i + 1 < n)
            is_tail  = (i == n - 1)
            is_head  = (i == 0)
            if has_next and cur_len + len(chunks[i+1]) <= hard_max_general:
                chunks[i+1] = cur + chunks[i+1]
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

def chunk_pdf_full_page(pdf_path: str, target_len: int, tol: int):
    doc = fitz.open(pdf_path)
    repetitives = _collect_repetitives(doc)
    fallback_min = _doc_fallback_header_min(doc)

    all_sizes = []
    for p in doc: all_sizes.extend(_page_font_sizes(p))
    arr = np.array(all_sizes, dtype=float) if all_sizes else np.array([12.0])
    doc_p50 = float(np.percentile(arr, 50)) if arr.size else 12.0
    doc_p90 = float(np.percentile(arr, HEADER_HI_PCT)) if arr.size else 16.0
    doc_start = max(doc_p90, doc_p50 * HEADER_RATIO_MIN, float(fallback_min))
    doc_auto  = _two_means_threshold(arr) or 0.0
    doc_thr_start = max(doc_start, doc_auto)
    if HEADER_ABS_MIN_PT > 0: doc_thr_start = max(doc_thr_start, float(HEADER_ABS_MIN_PT))
    doc_thr = _auto_tune_doc_threshold(doc, doc_thr_start)

    coarse_sections, fine_chunks = [], []
    current_section = "前言"; current_page = 1; buffer = ""; header_hits = 0

    for page_idx, page in enumerate(doc, start=1):
        page_thr = _page_header_min(page, fallback_min)
        if THRESHOLD_MODE == "doc":          header_min = doc_thr
        elif THRESHOLD_MODE == "hybrid":     header_min = max(page_thr, doc_thr)
        else:                                header_min = page_thr

        rows = list(_iter_lines_in_reading_order(page, cluster_columns=True))
        rows = _merge_number_badges(rows, page.rect.width, page.rect.height)

        for line_text, max_size, bbox, page_width in rows:
            if not line_text.strip(): continue

            # 0) 徽章殘片
            if _is_badge_fragment(page, line_text, max_size, bbox): continue
            # 1) 頁碼
            if DROP_SMALLEST_PAGENUM and _is_probable_page_number(page, line_text, max_size, bbox): continue
            # 2) 上邊帶頁眉（編號+文字）
            if _is_running_header_numbered(page, line_text, max_size, bbox, header_min): continue
            # 2.5) 強樣式頁眉
            if _is_probable_running_header_strong(page, line_text, max_size, bbox, header_min): continue

            # 3) 規範化重複頁眉/頁腳
            pos = _top_or_bottom(page, bbox)
            if pos is not None:
                key = _repeat_canon(line_text)
                if len(key) >= REPEAT_MIN_LEN and (key, pos) in repetitives:
                    continue

            ok = _is_section_header(
                line_text, max_size, header_min, page_width, bbox,
                require_center=CENTER_ONLY, require_numbered=NUMBERED_HEADERS_ONLY
            )
            if ok and _veto_topband_header(page, line_text, max_size, bbox, header_min):
                ok = False

            if ok:
                if buffer.strip():
                    coarse_sections.append((current_page, current_section, clean_spaces(buffer)))
                    buffer = ""
                current_section = line_text.strip(); current_page = page_idx
                header_hits += 1
            else:
                buffer = soft_join(buffer, line_text)

    if buffer.strip():
        coarse_sections.append((current_page, current_section, clean_spaces(buffer)))

    # 保底
    if header_hits == 0 and PAGE_FALLBACK_WHEN_NO_HEADER:
        coarse_sections = []
        for page_idx, page in enumerate(doc, start=1):
            full_text = normalize_text(page.get_text())
            if full_text.strip():
                coarse_sections.append((page_idx, f"第{page_idx}頁", full_text.strip()))

    # 細切 + 章內智慧合併
    min_len = target_len - tol; max_len = target_len + tol
    for page_no, section_title, content in coarse_sections:
        if min_len <= len(content) <= max_len:
            fine_chunks.append((page_no, section_title, content)); continue

        sents = smart_sentence_split(content)
        if not sents:
            fine_chunks.append((page_no, section_title, content)); continue

        # 先切
        sec_chunks = []
        for ch in smart_chunk_merge_by_chars(sents, target_len, tol):
            if len(ch) > max_len:
                sec_chunks.extend(split_oversize_chunk(ch, max_len))
            else:
                sec_chunks.append(ch)

        # 有過短才合併
        if SMART_CONSOLIDATE and sec_chunks:
            merge_floor = max(min_len, MERGE_MIN_ABS)
            need_merge = any(len(c) < min_len for c in sec_chunks)
            if need_merge:
                sec_chunks = _consolidate_small_chunks(
                    sec_chunks, min_len=merge_floor, max_len=max_len, overshoot=ALLOW_OVERSHOOT
                )

        for ch in sec_chunks:
            fine_chunks.append((page_no, section_title, ch))

    # → langchain Document
    docs = []
    for pg, sec, txt in fine_chunks:
        if txt.strip():
            docs.append(Document(page_content=txt, metadata={"source": pdf_path, "page": pg, "section": sec}))
    return docs
class RAGHelper:
    def __init__(self, pdf_folder, chunk_size=300, chunk_overlap=50,pdf_target_len=TARGET_LEN, pdf_tolerance=TOLERANCE):    #__init__ 是 python 的建構子
        self.pdf_folder = pdf_folder    # 儲存 PDF 檔案的 PATH
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.pdf_target_len = pdf_target_len       
        self.pdf_tolerance = pdf_tolerance  
        self.vectorstore = None
        self.retrieval_chain = None

    def get_loader(self,path: str):
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

    async def load_any_file_async(self,path: str):
        loader = self.get_loader(path)
        # 有些 loader 是 async 的，有些不是
        if hasattr(loader, "alazy_load"):
            pages = []
            async for page in loader.alazy_load():
                pages.append(page)
            return pages
        else:
            return loader.load()  # 同步方式載入

    #切割檔案
    def _split_documents(self, documents):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
            length_function=len,
        )
        return splitter.split_documents(documents)

    def _build_vectorstore(self, documents):
        print(f"建立向量資料庫... 共 {len(documents)} 個段落")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small") # 或是 model="text-embedding-3-large"
        self.vectorstore = FAISS.from_documents(documents, embeddings)

    async def load_and_prepare(self, file_extensions=None):
        print("開始載入檔案...")

        if os.path.exists("my_faiss_index"):    #如果本地有向量資料庫，載入本地的向量資料庫
            print("已偵測到現有向量資料庫，直接載入...")
            self.vectorstore = FAISS.load_local(
                "my_faiss_index",
                OpenAIEmbeddings(model="text-embedding-3-small"),
                allow_dangerous_deserialization=True
            )

        else:

            """
            載入並準備文件
            file_extensions: 要載入的檔案副檔名列表，例如 ['.pdf', '.txt', '.docx']
            如果為 None，則只載入 PDF 檔案（保持原有行為）
            """
            print("正在建立和讀取向量資料庫")

            if file_extensions is None:
                file_extensions = ['.pdf']  # 預設只載入 PDF

            all_chunks = []

            # 根據指定的副檔名載入檔案
            for ext in file_extensions:
                pattern = f"*{ext}"
                file_paths = glob.glob(os.path.join(self.pdf_folder, pattern))

                for path in file_paths:
                    try:
                        fname = os.path.basename(path)
                        print(f"讀取中: {fname}")

                        if Path(path).suffix.lower() == ".pdf":
                            # ★ PDF 使用智慧切割（標題偵測 / 頁眉頁腳過濾 / 細切 + 智慧合併）
                            docs = chunk_pdf_full_page(
                                pdf_path=path,
                                target_len=self.pdf_target_len,
                                tol=self.pdf_tolerance
                            )
                            all_chunks.extend(docs)
                            print(f" {fname}（PDF 智慧切）完成，共 {len(docs)} 段")
                        else:
                            # 其它副檔名維持原本流程
                            pages = await self.load_any_file_async(path)
                            chunks = self._split_documents(pages)
                            all_chunks.extend(chunks)
                            print(f" {fname} 分割完成，共 {len(chunks)} 段")

                    except Exception as e:
                        print(f"載入 {os.path.basename(path)} 時發生錯誤: {e}")

            print(f"所有檔案段落總數：{len(all_chunks)}")

            if len(all_chunks) == 0:
                raise ValueError("沒有成功載入任何文件")

            self._build_vectorstore(all_chunks)  # 將文字轉成向量，並建立向量資料庫
            self.vectorstore.save_local("my_faiss_index")   #將向量資料庫存到本地

    def setup_retrieval_chain(self):
        if not self.vectorstore:
            raise ValueError("請先執行 load_and_prepare()")

        llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
        # 創建檢索器
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 5}  # 只取前5個最相關的段落
        )
        # 創建提示詞模板
        system_prompt = (
            "你是一個基於 RAG 系統的計算機概論家教。請參考以下提供的內容來回答問題。"
            "用詞上請多使用正向鼓勵的詞語，並基於現有問題延伸出更多相關的問題。"
            "請針對問題舉出簡單好懂的比喻或例子。"
            "如果不知道如何回答問題，請說出來。"
            "如果問題和計算機概論無關，請將主題拉回計算機概論。"
            "使用 LaTeX 時，請使用 $ 符號作為塊級公式"
            "請用繁體中文回答。\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        # 創建文檔合併鏈
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        # 創建檢索鏈
        self.retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)

    def ask(self, query):
        if not self.retrieval_chain:
            raise ValueError("請先執行 setup_retrieval_chain()")
        try:
            result = self.retrieval_chain.invoke({"input": query})    #將使用者的問題傳給問答鏈，鏈內部會檢索並將檢索到的段落和問題交給大語言模型
            return result["answer"], result["context"]     # result["answer"] 是 語言模型給的答案，result["context"]  是檢索到的原始段落
        except Exception as e:
            if "max_tokens_per_request" in str(e):
                print("內容過長，嘗試使用較短的上下文...")
                self.setup_retrieval_chain_with_shorter_context()
                result = self.retrieval_chain.invoke({"input": query})
                return result["answer"], result["context"]
            else:
                raise e

    def setup_retrieval_chain_with_shorter_context(self):
        """設置更短上下文的檢索鏈"""
        if not self.vectorstore:
            raise ValueError("請先執行 load_and_prepare()")

        llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
        # 更嚴格的檢索配置
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
        system_prompt = (
            "你是一個問答助手。基於以下提供的內容來回答問題。"
            "如果內容中沒有相關資訊，請說「根據提供的資料無法回答這個問題」。"
            "請用繁體中文簡潔回答。\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        self.retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)
