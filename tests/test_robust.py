"""稳健长度区间（length_intervals）模式的单元测试。

包含与全枚举暴力解的三级最优对照、稳健极值/见证复核、字段校验与不可行码。
"""

import itertools
import random

import pytest

from app.solver import InfeasibleError, ValidationErrors, invert_payload, validate


def brute_robust(intervals, strain_min, strain_max, windows_raw):
    """对所有整数应变序列全枚举，返回三级字典序最优 (key, strains) 或 None。"""
    n = len(intervals)
    best = None
    for xs in itertools.product(range(strain_min, strain_max + 1), repeat=n):
        ok = True
        for s, e, lo, hi in windows_raw:
            f_min = sum(
                (intervals[i][0] if xs[i] >= 0 else intervals[i][1]) * xs[i]
                for i in range(s, e + 1)
            )
            f_max = sum(
                (intervals[i][1] if xs[i] >= 0 else intervals[i][0]) * xs[i]
                for i in range(s, e + 1)
            )
            if not (lo <= f_min and f_max <= hi):
                ok = False
                break
        if not ok:
            continue
        diffs = [xs[i] - xs[i - 1] for i in range(1, n)]
        key = (max(abs(d) for d in diffs), sum(abs(d) for d in diffs), xs)
        if best is None or key < best[0]:
            best = (key, list(xs))
    return best


def make_payload(seed, n, slack_choices, bounds_choice):
    rng = random.Random(seed)
    nominal = [rng.randint(1, 6) for _ in range(n)]
    intervals = [
        (max(1, L - rng.randint(0, 2)), L + rng.randint(0, 2)) for L in nominal
    ]
    strain_min, strain_max = bounds_choice
    truth = [rng.randint(strain_min, strain_max) for _ in range(n)]
    pairs = [(s, e) for s in range(n) for e in range(s, n)]
    rng.shuffle(pairs)
    windows_raw = []
    for s, e in pairs[: (10 if n == 6 else 12)]:
        f_min = sum(
            (intervals[i][0] if truth[i] >= 0 else intervals[i][1]) * truth[i]
            for i in range(s, e + 1)
        )
        f_max = sum(
            (intervals[i][1] if truth[i] >= 0 else intervals[i][0]) * truth[i]
            for i in range(s, e + 1)
        )
        slack = rng.choice(slack_choices)
        windows_raw.append((s, e, f_min - slack, f_max + slack))
    payload = {
        "segment_lengths": nominal,
        "length_intervals": [{"min": a, "max": b} for a, b in intervals],
        "strain_bounds": {"min": strain_min, "max": strain_max},
        "windows": [
            {
                "start_segment": s + 1,
                "end_segment": e + 1,
                "min_elongation": lo,
                "max_elongation": hi,
            }
            for s, e, lo, hi in windows_raw
        ],
    }
    return payload, intervals, windows_raw


@pytest.mark.parametrize("seed", range(24))
def test_robust_matches_bruteforce_n6(seed):
    payload, intervals, raw = make_payload(
        seed, 6, [0, 1, 2, 4], (-2, 2)
    )
    result = invert_payload(payload)
    expected = brute_robust(intervals, -2, 2, raw)
    assert expected is not None
    assert tuple(result["strains"]) == tuple(expected[1])
    assert result["objectives"]["max_adjacent_diff"] == expected[0][0]
    assert result["objectives"]["sum_adjacent_abs_diff"] == expected[0][1]


@pytest.mark.parametrize("seed", range(12))
def test_robust_matches_bruteforce_sign_bounds(seed):
    bounds = [(-1, 3), (2, 5), (-4, -1), (0, 4)]
    payload, intervals, raw = make_payload(
        seed, 6, [0, 1, 3], bounds[seed % len(bounds)]
    )
    smin, smax = bounds[seed % len(bounds)]
    result = invert_payload(payload)
    expected = brute_robust(intervals, smin, smax, raw)
    assert expected is not None
    assert tuple(result["strains"]) == tuple(expected[1])


def test_robust_extrema_and_witnesses_recomputable():
    payload, intervals, _raw = make_payload(123, 6, [1, 2], (-3, 3))
    result = invert_payload(payload)
    assert result["mode"] == "robust_interval"
    strains = result["strains"]
    for check in result["window_checks"]:
        s = check["start_segment"] - 1
        e = check["end_segment"] - 1
        span = e - s + 1
        assert len(check["min_extremum_lengths"]) == span
        assert len(check["max_extremum_lengths"]) == span
        f_min = sum(
            check["min_extremum_lengths"][k] * strains[s + k] for k in range(span)
        )
        f_max = sum(
            check["max_extremum_lengths"][k] * strains[s + k] for k in range(span)
        )
        assert f_min == check["robust_min_weighted_sum"]
        assert f_max == check["robust_max_weighted_sum"]
        assert check["robust_min_weighted_sum"] <= check["robust_max_weighted_sum"]
        assert check["min_elongation"] <= f_min
        assert f_max <= check["max_elongation"]
        assert check["robustly_satisfied"] is True
        # 见证长度必须是对应段的合法端点。
        for k in range(span):
            a, b = intervals[s + k]
            assert check["min_extremum_lengths"][k] in (a, b)
            assert check["max_extremum_lengths"][k] in (a, b)
        # 见证端点必须与应变符号一致：非负段 min 取下界、max 取上界。
        for k in range(span):
            a, b = intervals[s + k]
            x = strains[s + k]
            if x >= 0:
                assert check["min_extremum_lengths"][k] == a
                assert check["max_extremum_lengths"][k] == b
            else:
                assert check["min_extremum_lengths"][k] == b
                assert check["max_extremum_lengths"][k] == a


def test_robust_holds_against_all_length_vertices():
    """枚举每窗内各段长度的全部独立取值，确认提交区间确实闭包成立。"""
    payload, intervals, _raw = make_payload(77, 6, [1, 2], (-2, 2))
    result = invert_payload(payload)
    xs = result["strains"]
    for check in result["window_checks"]:
        s = check["start_segment"] - 1
        e = check["end_segment"] - 1
        local = [intervals[i] for i in range(s, e + 1)]
        for choice in itertools.product(*[(a, b) for a, b in local]):
            total = sum(choice[k] * xs[s + k] for k in range(len(choice)))
            assert check["min_elongation"] <= total <= check["max_elongation"]


def test_robust_infeasible_returns_code():
    # 单段窗要求正伸长 100，但长度区间至多 12、应变上界 10 ⇒ 稳健不可行。
    payload = {
        "segment_lengths": [10] * 6,
        "length_intervals": [{"min": 8, "max": 12}] * 6,
        "strain_bounds": {"min": -10, "max": 10},
        "windows": [
            {"start_segment": 1, "end_segment": 1,
             "min_elongation": 100, "max_elongation": 200},
        ]
        + [
            {
                "start_segment": i + 1,
                "end_segment": i + 1,
                "min_elongation": -10**6,
                "max_elongation": 10**6,
            }
            for i in range(1, 6)
        ]
        + [
            {
                "start_segment": 1,
                "end_segment": 6,
                "min_elongation": -10**9,
                "max_elongation": 10**9,
            },
            {
                "start_segment": 2,
                "end_segment": 5,
                "min_elongation": -10**9,
                "max_elongation": 10**9,
            },
        ],
    }
    with pytest.raises(InfeasibleError):
        invert_payload(payload)


def test_robust_infeasible_matches_bruteforce():
    payload, intervals, raw = make_payload(5, 6, [0], (-2, 2))
    # 收紧一个窗到不可能区间。
    s, e, _lo, hi = raw[0]
    raw[0] = (s, e, hi + 40, hi + 50)
    payload["windows"][0]["min_elongation"] = hi + 40
    payload["windows"][0]["max_elongation"] = hi + 50
    assert brute_robust(intervals, -2, 2, raw) is None
    with pytest.raises(InfeasibleError):
        invert_payload(payload)


def test_degenerate_intervals_equal_fixed_length():
    """所有区间退化为单点（a==b==名义长度）时，稳健解应等于固定长度解。"""
    lengths = [11, 12, 10, 13, 9, 14]
    base = {
        "segment_lengths": lengths,
        "strain_bounds": {"min": -50, "max": 50},
        "windows": [
            {"start_segment": i + 1, "end_segment": i + 1,
             "min_elongation": -1000, "max_elongation": 1000}
            for i in range(6)
        ]
        + [
            {"start_segment": 1, "end_segment": 6,
             "min_elongation": -500, "max_elongation": 500},
            {"start_segment": 1, "end_segment": 3,
             "min_elongation": -300, "max_elongation": 300},
        ],
    }
    fixed = invert_payload(base)
    robust_payload = dict(base)
    robust_payload["length_intervals"] = [
        {"min": L, "max": L} for L in lengths
    ]
    robust = invert_payload(robust_payload)
    assert robust["strains"] == fixed["strains"]
    assert robust["objectives"] == fixed["objectives"]


# ----------------------------- 字段校验 -----------------------------
def _valid_payload():
    return {
        "segment_lengths": [10] * 6,
        "length_intervals": [{"min": 9, "max": 11}] * 6,
        "strain_bounds": {"min": -10, "max": 10},
        "windows": [
            {"start_segment": i + 1, "end_segment": i + 1,
             "min_elongation": -5, "max_elongation": 5}
            for i in range(6)
        ]
        + [
            {"start_segment": 1, "end_segment": 6,
             "min_elongation": -10, "max_elongation": 10},
            {"start_segment": 2, "end_segment": 4,
             "min_elongation": -10, "max_elongation": 10},
        ],
    }


def test_interval_field_errors_point_to_segment():
    p = _valid_payload()
    p["length_intervals"][2] = {"min": 20, "max": 5}
    with pytest.raises(ValidationErrors) as exc:
        validate(p)
    fields = {f["field"] for f in exc.value.fields}
    assert "length_intervals[2].min" in fields

    p = _valid_payload()
    p["length_intervals"][0] = {"min": 0, "max": 3}
    with pytest.raises(ValidationErrors) as exc:
        validate(p)
    fields = {f["field"] for f in exc.value.fields}
    assert "length_intervals[0].min" in fields

    p = _valid_payload()
    p["length_intervals"][4] = {"min": 7}
    with pytest.raises(ValidationErrors) as exc:
        validate(p)
    assert any(f["field"] == "length_intervals[4].max" for f in exc.value.fields)

    p = _valid_payload()
    p["length_intervals"] = [{"min": 9, "max": 11}] * 5
    with pytest.raises(ValidationErrors) as exc:
        validate(p)
    assert any("length_intervals" == f["field"] for f in exc.value.fields)

    p = _valid_payload()
    p["length_intervals"] = "not-a-list"
    with pytest.raises(ValidationErrors):
        validate(p)

    p = _valid_payload()
    p["length_intervals"][1] = [9, 11]
    with pytest.raises(ValidationErrors) as exc:
        validate(p)
    assert any("length_intervals[1]" in f["field"] for f in exc.value.fields)


def test_interval_optional_absent_keeps_fixed_mode():
    p = _valid_payload()
    del p["length_intervals"]
    result = invert_payload(p)
    assert "mode" not in result  # 固定长度响应严格保持原结构
    assert "weighted_strain_sum" in result["window_checks"][0]
