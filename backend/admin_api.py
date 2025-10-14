#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
助教後台管理系統 API

提供圖表管理、描述編輯、預處理控制等功能
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

# 初始化路由
router = APIRouter(prefix="/api/admin", tags=["admin"])

# ==================== 資料模型 ====================

class ChartMetadata(BaseModel):
    """圖表 Metadata 模型"""
    chart_id: str
    chart_type: str = Field(..., description="圖表類型: figure/table/chart")
    chart_number: str
    original_caption: str
    page_number: int
    confidence_score: float
    source_file: str
    context_references: List[str] = []
    llm_description: Optional[str] = None
    image_path: Optional[str] = None
    manually_annotated: bool = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ChartListResponse(BaseModel):
    """圖表列表回應"""
    total: int
    page: int
    page_size: int
    charts: List[ChartMetadata]


class UpdateDescriptionRequest(BaseModel):
    """更新描述請求"""
    llm_description: str
    manually_annotated: bool = True


class UpdateCaptionRequest(BaseModel):
    """更新 Caption 請求"""
    original_caption: str
    chart_type: str
    chart_number: str


class BatchActionRequest(BaseModel):
    """批量操作請求"""
    action: str = Field(..., description="delete|move_to_filtered")
    chart_ids: List[str]


class StatsResponse(BaseModel):
    """統計資訊回應"""
    total_charts: int
    charts_with_caption: int
    charts_without_caption: int
    total_pdfs: int
    total_images_extracted: int
    filtered_images: int
    chart_types: Dict[str, int]
    avg_confidence_score: float


# ==================== 輔助函數 ====================

def get_metadata_path() -> Path:
    """取得 chart_metadata.json 路徑"""
    return Path("pdfFiles/enhanced_docs/chart_metadata.json")


def load_chart_metadata() -> Dict[str, Dict]:
    """載入圖表 metadata"""
    metadata_path = get_metadata_path()
    if not metadata_path.exists():
        return {}

    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_chart_metadata(metadata: Dict[str, Dict]):
    """儲存圖表 metadata"""
    metadata_path = get_metadata_path()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def get_image_path(chart_id: str, metadata: Dict) -> Optional[str]:
    """根據 chart_id 推測圖片路徑"""
    # 從 metadata 中查找對應的圖片
    charts_dir = Path("pdfFiles/charts")

    if not charts_dir.exists():
        return None

    # 嘗試匹配檔名（通常包含 chart_id 的一部分）
    source_file = metadata.get('source_file', '')
    page_num = metadata.get('page_number', 0)

    # 搜尋可能的圖片檔案
    for img_file in charts_dir.glob("*.png"):
        if f"_p{page_num:03d}_" in img_file.name:
            return str(img_file)

    return None


def verify_admin_role(user: Dict = Depends(lambda: {"role": "admin"})) -> Dict:
    """驗證管理員角色（簡化版，實際應從 JWT 取得）"""
    if user.get("role") not in ["admin", "ta"]:
        raise HTTPException(status_code=403, detail="需要管理員權限")
    return user


# ==================== API 端點 ====================

@router.get("/charts", response_model=ChartListResponse)
async def list_charts(
    page: int = Query(1, ge=1, description="頁碼"),
    page_size: int = Query(20, ge=1, le=100, description="每頁數量"),
    source_file: Optional[str] = Query(None, description="來源 PDF 檔名篩選"),
    has_caption: Optional[str] = Query("all", description="是否有 Caption: true/false/all"),
    chart_type: str = Query("all", description="圖表類型: figure/table/chart/all"),
    sort_by: str = Query("page_number", description="排序欄位"),
    order: str = Query("asc", description="排序方向: asc/desc"),
    user: Dict = Depends(verify_admin_role)
):
    """
    取得圖表列表（支援分頁和篩選）
    """
    metadata = load_chart_metadata()

    # 轉換為列表
    charts = []
    for chart_id, chart_data in metadata.items():
        chart_data['chart_id'] = chart_id
        chart_data['image_path'] = get_image_path(chart_id, chart_data)
        charts.append(chart_data)

    # 篩選
    if source_file:
        charts = [c for c in charts if c.get('source_file') == source_file]

    if has_caption != "all":
        has_caption_bool = has_caption.lower() == "true"
        charts = [c for c in charts if bool(c.get('original_caption')) == has_caption_bool]

    if chart_type != "all":
        charts = [c for c in charts if c.get('chart_type') == chart_type]

    # 排序
    reverse = (order == "desc")
    charts.sort(key=lambda x: x.get(sort_by, 0), reverse=reverse)

    # 分頁
    total = len(charts)
    start = (page - 1) * page_size
    end = start + page_size
    paginated_charts = charts[start:end]

    return ChartListResponse(
        total=total,
        page=page,
        page_size=page_size,
        charts=[ChartMetadata(**c) for c in paginated_charts]
    )


@router.get("/charts/{chart_id}", response_model=ChartMetadata)
async def get_chart(
    chart_id: str,
    user: Dict = Depends(verify_admin_role)
):
    """
    取得單一圖表詳情
    """
    metadata = load_chart_metadata()

    if chart_id not in metadata:
        raise HTTPException(status_code=404, detail=f"圖表 {chart_id} 不存在")

    chart_data = metadata[chart_id].copy()
    chart_data['chart_id'] = chart_id
    chart_data['image_path'] = get_image_path(chart_id, chart_data)

    return ChartMetadata(**chart_data)


@router.patch("/charts/{chart_id}/description")
async def update_chart_description(
    chart_id: str,
    request: UpdateDescriptionRequest,
    user: Dict = Depends(verify_admin_role)
):
    """
    更新圖表描述
    """
    metadata = load_chart_metadata()

    if chart_id not in metadata:
        raise HTTPException(status_code=404, detail=f"圖表 {chart_id} 不存在")

    # 更新描述
    metadata[chart_id]['llm_description'] = request.llm_description
    metadata[chart_id]['manually_annotated'] = request.manually_annotated
    metadata[chart_id]['updated_at'] = datetime.now().isoformat()

    save_chart_metadata(metadata)

    return {
        "success": True,
        "message": "描述已更新",
        "updated_at": metadata[chart_id]['updated_at']
    }


@router.patch("/charts/{chart_id}/caption")
async def update_chart_caption(
    chart_id: str,
    request: UpdateCaptionRequest,
    user: Dict = Depends(verify_admin_role)
):
    """
    更新圖表 Caption
    """
    metadata = load_chart_metadata()

    if chart_id not in metadata:
        raise HTTPException(status_code=404, detail=f"圖表 {chart_id} 不存在")

    # 更新 Caption
    metadata[chart_id]['original_caption'] = request.original_caption
    metadata[chart_id]['chart_type'] = request.chart_type
    metadata[chart_id]['chart_number'] = request.chart_number
    metadata[chart_id]['manually_annotated'] = True
    metadata[chart_id]['updated_at'] = datetime.now().isoformat()

    save_chart_metadata(metadata)

    return {
        "success": True,
        "message": "Caption 已更新",
        "updated_at": metadata[chart_id]['updated_at']
    }


@router.delete("/charts/{chart_id}")
async def delete_chart(
    chart_id: str,
    user: Dict = Depends(verify_admin_role)
):
    """
    刪除圖表（移至 charts_deleted 資料夾）
    """
    metadata = load_chart_metadata()

    if chart_id not in metadata:
        raise HTTPException(status_code=404, detail=f"圖表 {chart_id} 不存在")

    # 移動圖片檔案
    image_path = get_image_path(chart_id, metadata[chart_id])
    if image_path and os.path.exists(image_path):
        deleted_dir = Path("pdfFiles/charts_deleted")
        deleted_dir.mkdir(parents=True, exist_ok=True)

        dest_path = deleted_dir / Path(image_path).name
        shutil.move(image_path, str(dest_path))

    # 從 metadata 中移除
    del metadata[chart_id]
    save_chart_metadata(metadata)

    return {
        "success": True,
        "message": f"圖表 {chart_id} 已刪除並移至 charts_deleted/"
    }


@router.post("/charts/batch")
async def batch_action(
    request: BatchActionRequest,
    user: Dict = Depends(verify_admin_role)
):
    """
    批量操作圖表
    """
    metadata = load_chart_metadata()
    results = {
        "success": [],
        "failed": []
    }

    for chart_id in request.chart_ids:
        try:
            if chart_id not in metadata:
                results["failed"].append({"chart_id": chart_id, "reason": "不存在"})
                continue

            if request.action == "delete":
                # 移動圖片
                image_path = get_image_path(chart_id, metadata[chart_id])
                if image_path and os.path.exists(image_path):
                    deleted_dir = Path("pdfFiles/charts_deleted")
                    deleted_dir.mkdir(parents=True, exist_ok=True)
                    dest_path = deleted_dir / Path(image_path).name
                    shutil.move(image_path, str(dest_path))

                # 從 metadata 中移除
                del metadata[chart_id]
                results["success"].append(chart_id)

            elif request.action == "move_to_filtered":
                # 移至 filtered_out
                image_path = get_image_path(chart_id, metadata[chart_id])
                if image_path and os.path.exists(image_path):
                    filtered_dir = Path("pdfFiles/charts_filtered_out")
                    filtered_dir.mkdir(parents=True, exist_ok=True)
                    dest_path = filtered_dir / Path(image_path).name
                    shutil.move(image_path, str(dest_path))

                del metadata[chart_id]
                results["success"].append(chart_id)

            else:
                results["failed"].append({"chart_id": chart_id, "reason": "未知操作"})

        except Exception as e:
            results["failed"].append({"chart_id": chart_id, "reason": str(e)})

    save_chart_metadata(metadata)

    return {
        "success": len(results["success"]) > 0,
        "message": f"成功處理 {len(results['success'])} 個，失敗 {len(results['failed'])} 個",
        "results": results
    }


@router.get("/stats", response_model=StatsResponse)
async def get_stats(user: Dict = Depends(verify_admin_role)):
    """
    取得統計資訊
    """
    metadata = load_chart_metadata()

    # 統計圖表類型
    chart_types = {}
    total_confidence = 0
    charts_with_caption = 0

    for chart_data in metadata.values():
        # 類型統計
        chart_type = chart_data.get('chart_type', 'unknown')
        chart_types[chart_type] = chart_types.get(chart_type, 0) + 1

        # Caption 統計
        if chart_data.get('original_caption'):
            charts_with_caption += 1

        # 信心度累加
        total_confidence += chart_data.get('confidence_score', 0)

    total_charts = len(metadata)
    avg_confidence = total_confidence / total_charts if total_charts > 0 else 0

    # 統計檔案數量
    charts_dir = Path("pdfFiles/charts")
    filtered_dir = Path("pdfFiles/charts_filtered_out")

    total_images = len(list(charts_dir.glob("*.png"))) if charts_dir.exists() else 0
    filtered_images = len(list(filtered_dir.glob("*.png"))) if filtered_dir.exists() else 0

    # 統計 PDF 數量（從來源檔案）
    pdf_files = set(c.get('source_file') for c in metadata.values())

    return StatsResponse(
        total_charts=total_charts,
        charts_with_caption=charts_with_caption,
        charts_without_caption=total_charts - charts_with_caption,
        total_pdfs=len(pdf_files),
        total_images_extracted=total_images + filtered_images,
        filtered_images=filtered_images,
        chart_types=chart_types,
        avg_confidence_score=round(avg_confidence, 3)
    )


# ==================== 掛載到主應用 ====================

def setup_admin_routes(app):
    """
    將 admin 路由掛載到主 FastAPI 應用

    使用方式：
        from admin_api import setup_admin_routes
        app = FastAPI()
        setup_admin_routes(app)
    """
    app.include_router(router)
