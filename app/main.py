"""海缆整数微应变联合反演 HTTP API。"""

from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .solver import InfeasibleError, ValidationErrors, invert_payload

app = FastAPI(
    title="Submarine Cable Integer Strain Inversion API",
    version="1.0.0",
    description=(
        "对多个重叠标距的累计伸长量观测窗做精确整数联合反演，"
        "依次最小化相邻段最大应变差、相邻差绝对值之和与应变序列字典序。"
    ),
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/api/v1/schema")
async def schema_hint() -> dict:
    """返回输入输出字段约定，便于调用方自查。"""
    return {
        "request": {
            "segment_lengths": "6..12 个正整数，按顺序排列的缆段长度",
            "segment_length_intervals": (
                "可选；与 segment_lengths 等长，每项 "
                '{"min": 正整数, "max": 正整数} 且 min<=max、包含对应名义长度。'
                "提交后启用稳健模式：每个观测窗必须对各段长度在各自区间内的"
                "任意独立整数取值都满足累计伸长闭区间；未提交时行为与固定长度一致"
            ),
            "strain_bounds": {"min": "整数微应变闭区间下端", "max": "上端"},
            "windows": [
                {
                    "start_segment": "起始段（1 基，含）",
                    "end_segment": "结束段（1 基，含，不小于 start_segment）",
                    "min_elongation": "累计伸长量闭区间下端（整数）",
                    "max_elongation": "上端（整数，不小于下端）",
                }
            ],
            "window_count": "8..20",
        },
        "success_response": {
            "segment_count": "int",
            "strains": "逐段整数微应变（字典序最优）",
            "adjacent_diffs": "逐相邻段应变差，可直接复核两级平滑指标",
            "objectives": {
                "max_adjacent_diff": "第一级指标 = max(|adjacent_diffs|)",
                "sum_adjacent_abs_diff": "第二级指标 = sum(|adjacent_diffs|)",
            },
            "window_checks": [
                {
                    "weighted_strain_sum": "= Σ 窗内名义长度×应变，须落入提交的闭区间",
                    "total_length": "= Σ 窗内名义段长",
                    "satisfied": (
                        "固定长度模式：min<=weighted_strain_sum<=max；"
                        "稳健模式：min<=min_possible 且 max_possible<=max"
                    ),
                    "min_possible_weighted_sum": (
                        "仅稳健模式：按应变正负取长度端点推导的最小可能回算和（精确整数）"
                    ),
                    "max_possible_weighted_sum": "仅稳健模式：最大可能回算和",
                    "lengths_at_min": (
                        "仅稳健模式：取得最小和时窗内各段所用的长度端点（与窗内段一一对应）"
                    ),
                    "lengths_at_max": "仅稳健模式：取得最大和时窗内各段所用的长度端点",
                }
            ],
            "length_mode": "仅稳健模式出现，值为 interval；未提交长度区间时响应不含该字段",
            "criteria_order": [
                "max_adjacent_diff",
                "sum_adjacent_abs_diff",
                "lexicographic",
            ],
        },
        "errors": {
            "INVALID_INPUT": {"fields": [{"field": "字段路径", "message": "原因"}]},
            "INFEASIBLE": (
                "观测窗彼此冲突（稳健模式下：无法对所有长度取值同时满足），"
                "不存在满足全部闭区间的整数应变序列"
            ),
        },
    }


@app.post("/api/v1/invert")
async def invert(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={
                "code": "INVALID_INPUT",
                "message": "request body must be valid JSON",
                "fields": [{"field": ".", "message": "invalid JSON body"}],
            },
        )
    try:
        result = invert_payload(payload)
    except ValidationErrors as exc:
        return JSONResponse(
            status_code=422,
            content={
                "code": "INVALID_INPUT",
                "message": "input validation failed",
                "fields": exc.fields,
            },
        )
    except InfeasibleError as exc:
        return JSONResponse(
            status_code=409,
            content={"code": "INFEASIBLE", "message": str(exc)},
        )
    return JSONResponse(status_code=200, content={"code": "OK", "result": result})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=os.environ.get("APP_HOST", "0.0.0.0"),
        port=int(os.environ.get("APP_PORT", "8000")),
    )
