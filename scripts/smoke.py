"""对运行中的反演 API 做端到端冒烟（仅标准库）。

由 verify 一次性服务在 api 健康后执行。覆盖：
1. 固定长度兼容：成功反演并逐项回算每个观测窗的长度加权应变和与两级平滑指标；
2. 固定长度不可行（409）与非法输入（422）；
3. 稳健模式成功：复核每窗最小/最大可能回算和与取得极值时各段长度端点见证；
4. 稳健模式不可行（名义可行但无法稳健满足 -> 409）；
5. 非法长度区间 -> 422 且定位到对应缆段字段。
全部由提交数据直接复核；任一不符即以非零退出。
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def request(method: str, path: str, body: object | None = None) -> tuple[int, object]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        BASE + path,
        data=data,
        method=method,
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def check(condition: bool, message: str) -> None:
    if not condition:
        print(f"  [FAIL] {message}")
        raise SystemExit(1)
    print(f"  [ok]   {message}")


def feasible_payload() -> dict:
    lengths = [10, 12, 11, 13, 10, 14]
    truth = [3, 5, 4, 2, 6, 1]
    spans = [(0, 5), (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (0, 2)]
    windows = []
    for s, e in spans:
        total = sum(lengths[i] * truth[i] for i in range(s, e + 1))
        windows.append(
            {
                "start_segment": s + 1,
                "end_segment": e + 1,
                "min_elongation": total - 5,
                "max_elongation": total + 30,
            }
        )
    return {
        "segment_lengths": lengths,
        "strain_bounds": {"min": -50, "max": 50},
        "windows": windows,
    }


def interval_payload() -> dict:
    """稳健模式样例：为各段提交包含名义长度的正整数闭区间。"""
    base = feasible_payload()
    lengths = base["segment_lengths"]
    truth = [3, 5, 4, 2, 6, 1]
    intervals = [{"min": l - 1, "max": l + 1} for l in lengths]
    # 窗区间放宽到足以覆盖长度不确定性（spread = Σ (max-min)·|应变|）。
    windows = []
    for win in base["windows"]:
        s = win["start_segment"] - 1
        e = win["end_segment"] - 1
        total = sum(lengths[i] * truth[i] for i in range(s, e + 1))
        spread = sum(2 * abs(truth[i]) for i in range(s, e + 1))
        windows.append(
            {
                "start_segment": win["start_segment"],
                "end_segment": win["end_segment"],
                "min_elongation": total - spread - 5,
                "max_elongation": total + spread + 30,
            }
        )
    return {
        "segment_lengths": lengths,
        "segment_length_intervals": intervals,
        "strain_bounds": {"min": -50, "max": 50},
        "windows": windows,
    }


def robust_infeasible_payload() -> dict:
    """名义长度可行、但长度区间内某取值破坏窗约束（稳健不可行）。"""
    windows = [
        {"start_segment": 1, "end_segment": 1,
         "min_elongation": 15, "max_elongation": 15},
    ]
    for i in range(2, 7):
        windows.append(
            {"start_segment": i, "end_segment": i,
             "min_elongation": 10, "max_elongation": 10}
        )
    windows.append(
        {"start_segment": 1, "end_segment": 6,
         "min_elongation": 25, "max_elongation": 70}
    )
    windows.append(
        {"start_segment": 2, "end_segment": 6,
         "min_elongation": 50, "max_elongation": 50}
    )
    return {
        "segment_lengths": [15, 10, 10, 10, 10, 10],
        "segment_length_intervals": [{"min": 10, "max": 20}]
        + [{"min": 10, "max": 10}] * 5,
        "strain_bounds": {"min": 1, "max": 1},
        "windows": windows,
    }


def main() -> None:
    print(f"smoke against {BASE}")

    print("1) health")
    status, body = request("GET", "/health")
    check(status == 200, f"GET /health -> 200 (got {status})")
    check(body.get("status") == "ok", "health payload status == ok")

    print("2) fixed-length 兼容：成功反演 + 证据回算")
    payload = feasible_payload()
    status, body = request("POST", "/api/v1/invert", payload)
    check(status == 200, f"POST /api/v1/invert -> 200 (got {status}: {body})")
    check(body.get("code") == "OK", "response code == OK")
    result = body["result"]
    lengths = payload["segment_lengths"]
    strains = result["strains"]
    check(len(strains) == 6, "返回 6 段应变")
    check(
        "length_mode" not in result,
        "未提交长度区间时响应不含稳健模式字段（兼容）",
    )
    check(
        all(payload["strain_bounds"]["min"] <= v <= payload["strain_bounds"]["max"]
            for v in strains),
        "每段应变均落入统一应变闭区间",
    )
    for check_item, win in zip(result["window_checks"], payload["windows"]):
        total = sum(
            lengths[i] * strains[i]
            for i in range(win["start_segment"] - 1, win["end_segment"])
        )
        check(
            check_item["weighted_strain_sum"] == total,
            f"窗[{win['start_segment']},{win['end_segment']}] 回算和 {total} 与响应一致",
        )
        check(
            win["min_elongation"] <= total <= win["max_elongation"],
            f"窗[{win['start_segment']},{win['end_segment']}] {total} ∈ 提交闭区间",
        )
    diffs = [strains[i] - strains[i - 1] for i in range(1, 6)]
    check(
        result["adjacent_diffs"] == diffs,
        "adjacent_diffs 可由应变序列直接复核",
    )
    check(
        result["objectives"]["max_adjacent_diff"] == max(abs(d) for d in diffs),
        "一级指标 max_adjacent_diff 可由相邻差复核",
    )
    check(
        result["objectives"]["sum_adjacent_abs_diff"] == sum(abs(d) for d in diffs),
        "二级指标 sum_adjacent_abs_diff 可由相邻差复核",
    )

    print("3) conflicting windows -> INFEASIBLE")
    bad = feasible_payload()
    bad["windows"][0] = {
        "start_segment": 1,
        "end_segment": 6,
        "min_elongation": 10**9,
        "max_elongation": 10**9,
    }
    status, body = request("POST", "/api/v1/invert", bad)
    check(status == 409, f"冲突观测 -> 409 (got {status})")
    check(body.get("code") == "INFEASIBLE", "错误码 == INFEASIBLE")

    print("4) invalid input -> 字段错误")
    bad = feasible_payload()
    bad["segment_lengths"] = [1, 2, 3]
    status, body = request("POST", "/api/v1/invert", bad)
    check(status == 422, f"非法输入 -> 422 (got {status})")
    check(body.get("code") == "INVALID_INPUT", "错误码 == INVALID_INPUT")
    check(bool(body.get("fields")), "返回明确 fields 列表")

    print("5) 稳健模式成功：窗口极值与端点见证复核")
    payload = interval_payload()
    status, body = request("POST", "/api/v1/invert", payload)
    check(status == 200, f"稳健反演 -> 200 (got {status}: {body})")
    result = body["result"]
    check(result.get("length_mode") == "interval", "length_mode == interval")
    check(
        result.get("segment_length_intervals")
        == payload["segment_length_intervals"],
        "回显的长度区间与提交一致",
    )
    strains = result["strains"]
    intervals = [
        (iv["min"], iv["max"]) for iv in payload["segment_length_intervals"]
    ]
    for item, win in zip(result["window_checks"], payload["windows"]):
        segs = list(range(win["start_segment"] - 1, win["end_segment"]))
        check(
            len(item["lengths_at_min"]) == len(segs)
            and len(item["lengths_at_max"]) == len(segs),
            f"窗[{win['start_segment']},{win['end_segment']}] 见证与窗内段一一对应",
        )
        # 端点见证必须取自提交区间的端点。
        for k, i in enumerate(segs):
            lo_i, hi_i = intervals[i]
            check(
                item["lengths_at_min"][k] in (lo_i, hi_i)
                and item["lengths_at_max"][k] in (lo_i, hi_i),
                f"窗[{win['start_segment']},{win['end_segment']}] 段{i + 1} 见证取自区间端点",
            )
        # 见证加权和必须等于响应给出的两端极值。
        w_min = sum(item["lengths_at_min"][k] * strains[i] for k, i in enumerate(segs))
        w_max = sum(item["lengths_at_max"][k] * strains[i] for k, i in enumerate(segs))
        check(
            w_min == item["min_possible_weighted_sum"]
            and w_max == item["max_possible_weighted_sum"],
            f"窗[{win['start_segment']},{win['end_segment']}] 极值与见证加权和一致",
        )
        # 按应变正负独立推导的理论极值必须一致（不经过服务端见证）。
        t_min = sum(
            (intervals[i][0] if strains[i] >= 0 else intervals[i][1]) * strains[i]
            for i in segs
        )
        t_max = sum(
            (intervals[i][1] if strains[i] >= 0 else intervals[i][0]) * strains[i]
            for i in segs
        )
        check(
            w_min == t_min and w_max == t_max,
            f"窗[{win['start_segment']},{win['end_segment']}] 极值可按应变正负独立复核",
        )
        check(
            win["min_elongation"] <= w_min and w_max <= win["max_elongation"],
            f"窗[{win['start_segment']},{win['end_segment']}] 任意长度取值均满足提交区间",
        )
        check(item["satisfied"] is True, "satisfied 标志为真")
    diffs = [strains[i] - strains[i - 1] for i in range(1, 6)]
    check(
        result["objectives"]["max_adjacent_diff"] == max(abs(d) for d in diffs)
        and result["objectives"]["sum_adjacent_abs_diff"] == sum(abs(d) for d in diffs),
        "稳健模式两级平滑指标可由相邻差复核",
    )

    print("6) 稳健模式不可行 -> INFEASIBLE")
    payload = robust_infeasible_payload()
    # 佐证：同一请求去掉区间（固定名义长度）是可行的，不可行来自稳健要求。
    fixed = {k: v for k, v in payload.items() if k != "segment_length_intervals"}
    status, body = request("POST", "/api/v1/invert", fixed)
    check(status == 200, f"同名请求固定长度下可行 -> 200 (got {status})")
    status, body = request("POST", "/api/v1/invert", payload)
    check(status == 409, f"稳健不可行 -> 409 (got {status})")
    check(body.get("code") == "INFEASIBLE", "错误码 == INFEASIBLE")

    print("7) 非法长度区间 -> 422 并定位缆段字段")
    bad = interval_payload()
    bad["segment_length_intervals"][2] = {"min": 20, "max": 9}  # min > max
    status, body = request("POST", "/api/v1/invert", bad)
    check(status == 422, f"非法长度区间 -> 422 (got {status})")
    check(body.get("code") == "INVALID_INPUT", "错误码 == INVALID_INPUT")
    check(
        any(
            f["field"].startswith("segment_length_intervals[2]")
            for f in body.get("fields", [])
        ),
        "错误定位到 segment_length_intervals[2] 字段",
    )
    bad = interval_payload()
    bad["segment_length_intervals"][4] = {"min": 1, "max": 3}  # 不含名义长度 10
    status, body = request("POST", "/api/v1/invert", bad)
    check(status == 422, f"区间不含名义长度 -> 422 (got {status})")
    check(
        any(
            f["field"].startswith("segment_length_intervals[4]")
            for f in body.get("fields", [])
        ),
        "错误定位到 segment_length_intervals[4] 字段",
    )

    print("ALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
