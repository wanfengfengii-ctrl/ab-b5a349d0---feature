"""对运行中的反演 API 做端到端冒烟（仅标准库）。

由 verify 一次性服务在 api 健康后执行：
1. health
2. 固定长度兼容：成功回算 + 证据回算
3. 稳健成功：提交 length_intervals，复核每窗最小/最大可能和与端点见证
4. 稳健不可行：合法但无稳健解 -> 409 INFEASIBLE
5. 非法输入 -> 422 INVALID_INPUT（字段定位到具体缆段）
"""

import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")


def request(method: str, path: str, body: object | None = None) -> tuple[int, object]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        BASE + path,
        data=data,
        method=method,
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def check(condition: bool, message: str) -> None:
    if not condition:
        print(f"  [FAIL] {message}")
        raise SystemExit(1)
    print(f"  [ok]   {message}")


def fixed_payload() -> dict:
    lengths = [10, 12, 11, 13, 10, 14]
    truth = [3, 5, 4, 2, 6, 1]
    spans = [(0, 5), (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (0, 2)]
    windows = []
    for s, e in spans:
        total = sum(lengths[i] * truth[i] for i in range(s, e + 1))
        windows = windows or []
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


def robust_payload(slack=0) -> dict:
    """以固定 payload 为基础，提交每段长度闭区间 [L-1,L+1]。"""
    p = fixed_payload()
    p["length_intervals"] = [
        {"min": L - 1, "max": L + 1} for L in p["segment_lengths"]
    ]
    # 放宽窗口以容纳 ±1 的标距误差（真实场景由现场给出）。
    p["windows"] = [
        {
            **w,
            "min_elongation": w["min_elongation"] - slack - 60,
            "max_elongation": w["max_elongation"] + slack + 60,
        }
        for w in p["windows"]
    ]
    return p


def main() -> None:
    print(f"smoke against {BASE}")

    print("1) health")
    status, body = request("GET", "/health")
    check(status == 200, f"GET /health -> 200 (got {status})")
    check(body.get("status") == "ok", "health payload status == ok")

    print("2) fixed-length compat: success + 证据回算")
    payload = fixed_payload()
    status, body = request("POST", "/api/v1/invert", payload)
    check(status == 200, f"POST /api/v1/invert -> 200 (got {status}: {body})")
    check(body.get("code") == "OK", "response code == OK")
    check("mode" not in body, "固定长度响应不含新增 mode 字段（兼容）")
    result = body["result"]
    lengths = payload["segment_lengths"]
    strains = result["strains"]
    check(len(strains) == 6, "返回 6 段应变")
    all_in_bounds = all(-50 <= v <= 50 for v in strains)
    check(all_in_bounds, "每段应变均落入统一应变闭区间")
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
        check(check_item["satisfied"] is True, "satisfied is True")
    diffs = [strains[i] - strains[i - 1] for i in range(1, 6)]
    check(
        result["adjacent_diffs"] == diffs,
        "adjacent_diffs 可由应变序列直接复核",
    )
    check(
        result["objectives"]["max_adjacent_diff"] == max(abs(d) for d in diffs),
        "一级指标 max_adjacent_diff 可复核",
    )
    check(
        result["objectives"]["sum_adjacent_abs_diff"] == sum(abs(d) for d in diffs),
        "二级指标 sum_adjacent_abs_diff 可复核",
    )

    print("3) robust success: 逐窗最小/最大可能和 + 长度端点见证")
    payload = robust_payload()
    status, body = request("POST", "/api/v1/invert", payload)
    check(status == 200, f"robust POST -> 200 (got {status}: {body})")
    result = body["result"]
    check(result.get("mode") == "robust_interval", "mode == robust_interval")
    intervals = [(d["min"], d["max"]) for d in payload["length_intervals"]]
    xs = result["strains"]
    for c in result["window_checks"]:
        s, e = c["start_segment"] - 1, c["end_segment"] - 1
        span = e - s + 1
        fmin = sum(c["min_extremum_lengths"][k] * xs[s + k] for k in range(span))
        fmax = sum(c["max_extremum_lengths"][k] * xs[s + k] for k in range(span))
        check(
            fmin == c["robust_min_weighted_sum"],
            f"窗[{s+1},{e+1}] 最小极值由见证长度直接复核",
        )
        check(
            fmax == c["robust_max_weighted_sum"],
            f"窗[{s+1},{e+1}] 最大极值由见证长度直接复核",
        )
        check(
            c["min_elongation"] <= fmin and fmax <= c["max_elongation"],
            f"窗[{s+1},{e+1}] 极值闭区间落在提交伸长区间内",
        )
        for k in range(span):
            a, b = intervals[s + k]
            check(
                c["min_extremum_lengths"][k] in (a, b)
                and c["max_extremum_lengths"][k] in (a, b),
                f"窗[{s+1},{e+1}] 第{k+1}段见证为合法长度端点",
            )

    print("4) robust infeasible -> 409 INFEASIBLE")
    bad = robust_payload()
    # 第一段单段窗：长度至多 L+1、应变上界 50，却要求 ≥ 10^6，稳健不可行。
    bad["windows"] = [
        {
            "start_segment": 1,
            "end_segment": 1,
            "min_elongation": 10**6,
            "max_elongation": 10**9,
        }
    ] + [
        {
            "start_segment": i + 1,
            "end_segment": i + 1,
            "min_elongation": -10**9,
            "max_elongation": 10**9,
        }
        for i in range(1, 6)
    ] + [
        {
            "start_segment": 1,
            "end_segment": 6,
            "min_elongation": -10**12,
            "max_elongation": 10**12,
        },
        {
            "start_segment": 2,
            "end_segment": 5,
            "min_elongation": -10**12,
            "max_elongation": 10**12,
        },
    ]
    status, body = request("POST", "/api/v1/invert", bad)
    check(status == 409, f"robust conflict -> 409 (got {status})")
    check(body.get("code") == "INFEASIBLE", "robust 错误码 == INFEASIBLE")

    print("5) invalid interval field -> 422 定位到缆段")
    bad2 = robust_payload()
    bad2["length_intervals"][3] = {"min": 99, "max": 1}
    status, body = request("POST", "/api/v1/invert", bad2)
    check(status == 422, f"illegal interval -> 422 (got {status})")
    check(body.get("code") == "INVALID_INPUT", "错误码 == INVALID_INPUT")
    check(
        bool(body.get("fields"))
        and "length_intervals[3].min" in {f["field"] for f in body["fields"]},
        "返回明确 length_intervals[3] 字段",
    )

    print("ALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
