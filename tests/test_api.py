"""API 层测试：健康检查、成功反演、422 字段错误、409 不可行、坏 JSON。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def sample_payload():
    return {
        "segment_lengths": [10, 12, 11, 13, 10, 14],
        "strain_bounds": {"min": -100, "max": 100},
        "windows": [
            {"start_segment": 1, "end_segment": 6, "min_elongation": 60, "max_elongation": 700},
            {"start_segment": 1, "end_segment": 1, "min_elongation": 0, "max_elongation": 1000},
            {"start_segment": 2, "end_segment": 2, "min_elongation": 0, "max_elongation": 1000},
            {"start_segment": 3, "end_segment": 3, "min_elongation": 0, "max_elongation": 1000},
            {"start_segment": 4, "end_segment": 4, "min_elongation": 0, "max_elongation": 1000},
            {"start_segment": 5, "end_segment": 5, "min_elongation": 0, "max_elongation": 1000},
            {"start_segment": 6, "end_segment": 6, "min_elongation": 0, "max_elongation": 1000},
            {"start_segment": 1, "end_segment": 3, "min_elongation": 0, "max_elongation": 2000},
        ],
    }


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_schema_hint_documents_fields():
    resp = client.get("/api/v1/schema")
    assert resp.status_code == 200
    body = resp.json()
    assert "window_checks" in body["success_response"]
    assert "INFEASIBLE" in body["errors"]


def test_invert_success_and_recompute():
    resp = client.post("/api/v1/invert", json=sample_payload())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["code"] == "OK"
    result = body["result"]
    assert len(result["strains"]) == 6
    assert all(-100 <= v <= 100 for v in result["strains"])
    # 全部回算和均可由提交数据复核且落在提交区间内。
    payload = sample_payload()
    lengths = payload["segment_lengths"]
    strains = result["strains"]
    for check, win in zip(result["window_checks"], payload["windows"]):
        total = sum(
            lengths[i] * strains[i]
            for i in range(win["start_segment"] - 1, win["end_segment"])
        )
        assert check["weighted_strain_sum"] == total
        assert win["min_elongation"] <= total <= win["max_elongation"]
        assert check["satisfied"] is True
    diffs = [strains[i] - strains[i - 1] for i in range(1, 6)]
    assert result["objectives"]["max_adjacent_diff"] == max(abs(d) for d in diffs)
    assert result["objectives"]["sum_adjacent_abs_diff"] == sum(abs(d) for d in diffs)


def test_invert_invalid_fields_422():
    payload = sample_payload()
    payload["segment_lengths"] = [1, 2]  # 少于 6 段
    resp = client.post("/api/v1/invert", json=payload)
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "INVALID_INPUT"
    assert any(f["field"] == "segment_lengths" for f in body["fields"])


def test_invert_infeasible_409():
    payload = sample_payload()
    payload["windows"][0] = {
        "start_segment": 1,
        "end_segment": 6,
        "min_elongation": 10**9,
        "max_elongation": 10**9,
    }
    resp = client.post("/api/v1/invert", json=payload)
    assert resp.status_code == 409
    assert resp.json()["code"] == "INFEASIBLE"


def test_bad_json_400():
    resp = client.post(
        "/api/v1/invert",
        content=b"{not json",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "INVALID_INPUT"


def interval_payload():
    """稳健模式样例：各段提交包含名义长度的闭区间。"""
    lengths = [10, 12, 11, 13, 10, 14]
    truth = [3, 5, 4, 2, 6, 1]
    intervals = [{"min": l - 1, "max": l + 1} for l in lengths]
    spans = [(0, 5), (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (0, 2)]
    windows = []
    for s, e in spans:
        total = sum(lengths[i] * truth[i] for i in range(s, e + 1))
        spread = sum(2 * abs(truth[i]) for i in range(s, e + 1))
        windows.append(
            {
                "start_segment": s + 1,
                "end_segment": e + 1,
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


def test_invert_interval_mode_success_and_recompute():
    payload = interval_payload()
    resp = client.post("/api/v1/invert", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["code"] == "OK"
    result = body["result"]
    assert result["length_mode"] == "interval"
    assert result["segment_length_intervals"] == payload["segment_length_intervals"]
    strains = result["strains"]
    intervals = [
        (iv["min"], iv["max"]) for iv in payload["segment_length_intervals"]
    ]
    # 窗口极值与端点见证均可由请求数据直接复核。
    for check, win in zip(result["window_checks"], payload["windows"]):
        segs = list(range(win["start_segment"] - 1, win["end_segment"]))
        assert len(check["lengths_at_min"]) == len(segs)
        assert len(check["lengths_at_max"]) == len(segs)
        for k, i in enumerate(segs):
            assert check["lengths_at_min"][k] in intervals[i]
            assert check["lengths_at_max"][k] in intervals[i]
        w_min = sum(check["lengths_at_min"][k] * strains[i] for k, i in enumerate(segs))
        w_max = sum(check["lengths_at_max"][k] * strains[i] for k, i in enumerate(segs))
        assert check["min_possible_weighted_sum"] == w_min
        assert check["max_possible_weighted_sum"] == w_max
        # 按应变正负独立推导的理论极值必须一致。
        t_min = sum(
            (intervals[i][0] if strains[i] >= 0 else intervals[i][1]) * strains[i]
            for i in segs
        )
        t_max = sum(
            (intervals[i][1] if strains[i] >= 0 else intervals[i][0]) * strains[i]
            for i in segs
        )
        assert w_min == t_min
        assert w_max == t_max
        assert win["min_elongation"] <= w_min and w_max <= win["max_elongation"]
        assert check["satisfied"] is True
    diffs = [strains[i] - strains[i - 1] for i in range(1, 6)]
    assert result["objectives"]["max_adjacent_diff"] == max(abs(d) for d in diffs)
    assert result["objectives"]["sum_adjacent_abs_diff"] == sum(abs(d) for d in diffs)


def test_invert_interval_mode_robust_infeasible_409():
    """名义长度可行，但区间内某取值破坏窗约束 -> 409。"""
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
    payload = {
        "segment_lengths": [15, 10, 10, 10, 10, 10],
        "segment_length_intervals": [{"min": 10, "max": 20}]
        + [{"min": 10, "max": 10}] * 5,
        "strain_bounds": {"min": 1, "max": 1},
        "windows": windows,
    }
    # 佐证：去掉区间的同一请求在固定长度下可行。
    fixed_payload = {
        k: v for k, v in payload.items() if k != "segment_length_intervals"
    }
    assert client.post("/api/v1/invert", json=fixed_payload).status_code == 200
    resp = client.post("/api/v1/invert", json=payload)
    assert resp.status_code == 409
    assert resp.json()["code"] == "INFEASIBLE"


def test_invert_interval_validation_422_locates_segment():
    payload = interval_payload()
    payload["segment_length_intervals"][2] = {"min": 20, "max": 9}  # min>max
    resp = client.post("/api/v1/invert", json=payload)
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "INVALID_INPUT"
    assert any(
        f["field"].startswith("segment_length_intervals[2]") for f in body["fields"]
    )
    # 区间必须包含名义长度。
    payload = interval_payload()
    payload["segment_length_intervals"][4] = {"min": 1, "max": 3}  # 名义 10 不在内
    resp = client.post("/api/v1/invert", json=payload)
    assert resp.status_code == 422
    assert any(
        f["field"].startswith("segment_length_intervals[4]")
        for f in resp.json()["fields"]
    )
    # 数量必须与缆段数一致。
    payload = interval_payload()
    payload["segment_length_intervals"] = payload["segment_length_intervals"][:5]
    resp = client.post("/api/v1/invert", json=payload)
    assert resp.status_code == 422
    assert any(
        f["field"] == "segment_length_intervals" for f in resp.json()["fields"]
    )
