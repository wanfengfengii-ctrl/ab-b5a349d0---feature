"""稳健（长度闭区间）模式测试。

- 与全枚举暴力参考实现对照三级最优解（稳健可行性按定义逐点判定）；
- 窗口极值与端点见证必须能由请求数据直接复核；
- 区间退化为点时与固定长度模式完全一致（兼容性）；
- 名义可行但稳健不可行 -> InfeasibleError；
- 非法长度区间 -> ValidationErrors 且定位到对应缆段字段。
"""

from __future__ import annotations

import itertools
import random
import time

import pytest

from app.solver import InfeasibleError, ValidationErrors, invert_payload, validate


def brute_force_robust(intervals, strain_min, strain_max, windows):
    """全枚举参考实现：稳健可行（窗内任意长度取值均满足）下按 (M, S, x) 取最小。"""
    n = len(intervals)
    best = None
    for xs in itertools.product(range(strain_min, strain_max + 1), repeat=n):
        ok = True
        for s, e, lo, hi in windows:
            w_min = 0
            w_max = 0
            for i in range(s, e + 1):
                li, ui = intervals[i]
                x = xs[i]
                if x >= 0:
                    w_min += li * x
                    w_max += ui * x
                else:
                    w_min += ui * x
                    w_max += li * x
            if not (lo <= w_min and w_max <= hi):
                ok = False
                break
        if not ok:
            continue
        diffs = [xs[i] - xs[i - 1] for i in range(1, n)]
        key = (max(abs(d) for d in diffs), sum(abs(d) for d in diffs), xs)
        if best is None or key < best[0]:
            best = (key, list(xs))
    return best


def make_interval_payload(lengths, intervals, strain_min, strain_max, windows_raw):
    return {
        "segment_lengths": lengths,
        "segment_length_intervals": [
            {"min": lo, "max": hi} for lo, hi in intervals
        ],
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


def assert_witnesses(result, intervals, windows_raw):
    """窗口极值与端点见证必须能由请求数据（区间端点 + 应变序列）直接复核。"""
    strains = result["strains"]
    assert result["length_mode"] == "interval"
    assert result["segment_length_intervals"] == [
        {"min": lo, "max": hi} for lo, hi in intervals
    ]
    for check, (s, e, lo, hi) in zip(result["window_checks"], windows_raw):
        segs = list(range(s, e + 1))
        assert len(check["lengths_at_min"]) == len(segs)
        assert len(check["lengths_at_max"]) == len(segs)
        # 见证必须取自提交区间的端点。
        for k, i in enumerate(segs):
            li, ui = intervals[i]
            assert check["lengths_at_min"][k] in (li, ui)
            assert check["lengths_at_max"][k] in (li, ui)
        # 见证加权和必须等于响应给出的两端极值。
        w_min = sum(
            check["lengths_at_min"][k] * strains[i] for k, i in enumerate(segs)
        )
        w_max = sum(
            check["lengths_at_max"][k] * strains[i] for k, i in enumerate(segs)
        )
        assert check["min_possible_weighted_sum"] == w_min
        assert check["max_possible_weighted_sum"] == w_max
        # 按应变正负独立推导的理论极值必须一致（不经过服务端见证）。
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
        # 稳健满足：任意长度取值的回算和都落在提交闭区间内。
        assert lo <= w_min and w_max <= hi
        assert check["satisfied"] is True


@pytest.mark.parametrize("seed", range(10))
def test_robust_matches_bruteforce(seed):
    rng = random.Random(1000 + seed)
    n = 6
    strain_min, strain_max = -2, 2
    truth = [rng.randint(strain_min, strain_max) for _ in range(n)]
    lengths = []
    intervals = []
    for _ in range(n):
        nominal = rng.randint(2, 5)
        lo = max(1, nominal - rng.randint(0, 1))
        hi = nominal + rng.randint(0, 1)
        lengths.append(nominal)
        intervals.append((lo, hi))
    # 真值长度取区间内随机值；窗宽覆盖长度不确定性 + 额外松弛 => 真值稳健可行。
    actual = [rng.randint(lo, hi) for lo, hi in intervals]
    windows_raw = []
    seen = set()
    while len(windows_raw) < 10:
        s = rng.randint(0, n - 1)
        e = rng.randint(s, n - 1)
        if (s, e) in seen:
            continue
        seen.add((s, e))
        total = sum(actual[i] * truth[i] for i in range(s, e + 1))
        spread = sum(
            (intervals[i][1] - intervals[i][0]) * abs(truth[i])
            for i in range(s, e + 1)
        )
        slack = rng.randint(0, 2)
        windows_raw.append((s, e, total - spread - slack, total + spread + slack))
    payload = make_interval_payload(
        lengths, intervals, strain_min, strain_max, windows_raw
    )
    result = invert_payload(payload)
    expected = brute_force_robust(intervals, strain_min, strain_max, windows_raw)
    assert expected is not None
    assert result["strains"] == expected[1]
    assert result["objectives"]["max_adjacent_diff"] == expected[0][0]
    assert result["objectives"]["sum_adjacent_abs_diff"] == expected[0][1]
    assert_witnesses(result, intervals, windows_raw)


@pytest.mark.parametrize("seed", range(4))
def test_robust_n7_matches_bruteforce(seed):
    rng = random.Random(2000 + seed)
    n = 7
    strain_min, strain_max = -1, 1  # 3^7 = 2187 个枚举点
    truth = [rng.randint(strain_min, strain_max) for _ in range(n)]
    lengths = [rng.randint(1, 3) for _ in range(n)]
    intervals = [(max(1, l - 1), l + 1) for l in lengths]
    actual = [rng.randint(lo, hi) for lo, hi in intervals]
    windows_raw = []
    pairs = [(s, e) for s in range(n) for e in range(s, n)]
    rng.shuffle(pairs)
    for s, e in pairs[:12]:
        total = sum(actual[i] * truth[i] for i in range(s, e + 1))
        spread = sum(
            (intervals[i][1] - intervals[i][0]) * abs(truth[i])
            for i in range(s, e + 1)
        )
        windows_raw.append((s, e, total - spread - 1, total + spread + 1))
    payload = make_interval_payload(
        lengths, intervals, strain_min, strain_max, windows_raw
    )
    result = invert_payload(payload)
    expected = brute_force_robust(intervals, strain_min, strain_max, windows_raw)
    assert expected is not None
    assert result["strains"] == expected[1]
    assert_witnesses(result, intervals, windows_raw)


def test_degenerate_intervals_match_fixed_mode():
    """区间退化为点（=名义长度）时，稳健模式必须与固定长度模式给出同一最优解。"""
    rng = random.Random(5)
    n = 6
    lengths = [3, 1, 4, 1, 5, 9]
    strain_min, strain_max = -3, 3
    truth = [rng.randint(-2, 2) for _ in range(n)]
    windows_raw = []
    pairs = [(s, e) for s in range(n) for e in range(s, n)]
    rng.shuffle(pairs)
    for s, e in pairs[:10]:
        total = sum(lengths[i] * truth[i] for i in range(s, e + 1))
        windows_raw.append((s, e, total - 2, total + 2))
    windows = [
        {
            "start_segment": s + 1,
            "end_segment": e + 1,
            "min_elongation": lo,
            "max_elongation": hi,
        }
        for s, e, lo, hi in windows_raw
    ]
    fixed_payload = {
        "segment_lengths": lengths,
        "strain_bounds": {"min": strain_min, "max": strain_max},
        "windows": windows,
    }
    robust_payload = {
        **fixed_payload,
        "segment_length_intervals": [{"min": l, "max": l} for l in lengths],
    }
    r_fixed = invert_payload(fixed_payload)
    r_robust = invert_payload(robust_payload)
    assert r_robust["strains"] == r_fixed["strains"]
    assert r_robust["adjacent_diffs"] == r_fixed["adjacent_diffs"]
    assert r_robust["objectives"] == r_fixed["objectives"]
    # 退化区间下极值即名义回算，两端见证唯一且相等。
    for check in r_robust["window_checks"]:
        assert check["min_possible_weighted_sum"] == check["weighted_strain_sum"]
        assert check["max_possible_weighted_sum"] == check["weighted_strain_sum"]
        assert check["lengths_at_min"] == check["lengths_at_max"]
        s, e = check["start_segment"] - 1, check["end_segment"] - 1
        assert check["lengths_at_min"] == lengths[s : e + 1]


def test_fixed_mode_response_unchanged():
    """未提交长度区间时，响应不得出现稳健模式字段（兼容性）。"""
    payload = {
        "segment_lengths": [10, 12, 11, 13, 10, 14],
        "strain_bounds": {"min": -100, "max": 100},
        "windows": [
            {"start_segment": 1, "end_segment": 6,
             "min_elongation": 60, "max_elongation": 700}
        ]
        + [
            {"start_segment": i, "end_segment": i,
             "min_elongation": 0, "max_elongation": 1000}
            for i in range(1, 7)
        ]
        + [
            {"start_segment": 1, "end_segment": 3,
             "min_elongation": 0, "max_elongation": 2000},
        ],
    }
    result = invert_payload(payload)
    assert set(result) == {
        "segment_count",
        "strains",
        "adjacent_diffs",
        "objectives",
        "window_checks",
        "criteria_order",
    }
    for check in result["window_checks"]:
        assert set(check) == {
            "index",
            "start_segment",
            "end_segment",
            "total_length",
            "min_elongation",
            "max_elongation",
            "weighted_strain_sum",
            "satisfied",
        }


def test_robust_infeasible_but_nominally_feasible():
    """名义长度可行、但区间内某取值破坏窗约束 -> 不可行（409 语义）。"""
    # x 全钉为 1；窗 1 要求和恒为 15，而 L_1 ∈ [10,20] 时和 ∈ [10,20]。
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
    # 同一请求去掉区间（固定名义长度）是可行的，佐证不可行来自稳健要求。
    fixed_payload = {
        k: v for k, v in payload.items() if k != "segment_length_intervals"
    }
    assert invert_payload(fixed_payload)["strains"] == [1] * 6
    with pytest.raises(InfeasibleError):
        invert_payload(payload)


def test_robust_infeasible_randomized():
    """随机实例中收紧窗区间至不含稳健极值，应判不可行且暴力一致。"""
    rng = random.Random(3100)
    n = 6
    lengths = [rng.randint(2, 4) for _ in range(n)]
    intervals = [(l, l + 1) for l in lengths]
    strain_min, strain_max = -2, 2
    windows_raw = []
    for s in range(n):
        # 单段窗：L_i·x_i ∈ [lo_i·x, hi_i·x]，要求恒等于一个取不到的值。
        windows_raw.append((s, s, 1, 1))
    while len(windows_raw) < 9:
        s = rng.randint(0, n - 1)
        e = rng.randint(s, n - 1)
        windows_raw.append((s, e, -50, 50))
    payload = make_interval_payload(
        lengths, intervals, strain_min, strain_max, windows_raw
    )
    # 单段窗要求 L_i·x_i == 1 对所有 L_i ∈ {l, l+1}：l>=2 不可能。
    assert brute_force_robust(intervals, strain_min, strain_max, windows_raw) is None
    with pytest.raises(InfeasibleError):
        invert_payload(payload)


def test_interval_validation_errors_locate_segment_fields():
    base = {
        "segment_lengths": [10, 12, 11, 13, 10, 14],
        "strain_bounds": {"min": -10, "max": 10},
        "windows": [
            {"start_segment": 1, "end_segment": 6,
             "min_elongation": -100000, "max_elongation": 100000}
        ]
        * 8,
    }

    def fields_of(intervals):
        payload = dict(base)
        payload["segment_length_intervals"] = intervals
        with pytest.raises(ValidationErrors) as exc:
            validate(payload)
        return {f["field"] for f in exc.value.fields}

    good = [{"min": l - 1, "max": l + 1} for l in base["segment_lengths"]]

    # min > max
    bad = [dict(iv) for iv in good]
    bad[2] = {"min": 20, "max": 9}
    assert "segment_length_intervals[2].min" in fields_of(bad)
    # 非正整数端点
    bad = [dict(iv) for iv in good]
    bad[0] = {"min": 0, "max": 11}
    bad[4] = {"min": -3, "max": -1}
    fields = fields_of(bad)
    assert "segment_length_intervals[0].min" in fields
    assert "segment_length_intervals[4].min" in fields
    # 区间不含名义长度
    bad = [dict(iv) for iv in good]
    bad[1] = {"min": 1, "max": 5}  # 名义 12 不在内
    assert "segment_length_intervals[1]" in fields_of(bad)
    # 非整数 / 布尔端点
    bad = [dict(iv) for iv in good]
    bad[3] = {"min": 12.5, "max": 14}
    bad[5] = {"min": True, "max": 15}
    fields = fields_of(bad)
    assert "segment_length_intervals[3].min" in fields
    assert "segment_length_intervals[5].min" in fields
    # 缺键 / 未知键 / 非对象
    bad = [dict(iv) for iv in good]
    del bad[0]["max"]
    assert "segment_length_intervals[0].max" in fields_of(bad)
    bad = [dict(iv) for iv in good]
    bad[2]["note"] = 1
    assert "segment_length_intervals[2].note" in fields_of(bad)
    bad = [dict(iv) for iv in good]
    bad[4] = 7
    assert "segment_length_intervals[4]" in fields_of(bad)
    # 数量不匹配 / 非数组
    assert "segment_length_intervals" in fields_of(good[:5])
    assert "segment_length_intervals" in fields_of({"min": 1, "max": 2})


def test_intervals_absent_is_none_and_present_is_normalized():
    payload = {
        "segment_lengths": [10, 12, 11, 13, 10, 14],
        "strain_bounds": {"min": -10, "max": 10},
        "windows": [
            {"start_segment": 1, "end_segment": 6,
             "min_elongation": -100000, "max_elongation": 100000}
        ]
        * 8,
    }
    assert validate(payload)[4] is None
    payload["segment_length_intervals"] = [
        {"min": l - 1, "max": l + 1} for l in payload["segment_lengths"]
    ]
    assert validate(payload)[4] == [(l - 1, l + 1) for l in payload["segment_lengths"]]


def test_robust_exact_large_integers():
    """大整数区间端点与回算极值必须逐位精确。"""
    big = 10**25
    lengths = [big + i for i in range(6)]
    intervals = [(l - 10**20, l + 10**20) for l in lengths]
    truth = [(-1) ** i * (i + 1) for i in range(6)]
    windows_raw = []
    pairs = [(s, e) for s in range(6) for e in range(s, 6)]
    for s, e in pairs[:12]:
        total = sum(lengths[i] * truth[i] for i in range(s, e + 1))
        spread = sum(
            (intervals[i][1] - intervals[i][0]) * abs(truth[i])
            for i in range(s, e + 1)
        )
        windows_raw.append((s, e, total - spread, total + spread))
    payload = make_interval_payload(lengths, intervals, -100, 100, windows_raw)
    result = invert_payload(payload)
    assert_witnesses(result, intervals, windows_raw)


def test_robust_max_size_performance():
    rng = random.Random(777)
    n = 12
    lengths = [rng.randint(1, 100) for _ in range(n)]
    intervals = [(l, l + 1) for l in lengths]
    truth = [rng.randint(-120, 120) for _ in range(n)]
    windows = []
    starts = list(range(n))
    rng.shuffle(starts)
    for s in starts:
        e = rng.randint(s, n - 1)
        total = sum(lengths[i] * truth[i] for i in range(s, e + 1))
        spread = sum(
            (intervals[i][1] - intervals[i][0]) * abs(truth[i])
            for i in range(s, e + 1)
        )
        windows.append(
            {"start_segment": s + 1, "end_segment": e + 1,
             "min_elongation": total - spread - 30,
             "max_elongation": total + spread + 30}
        )
    while len(windows) < 20:
        s = rng.randint(0, n - 1)
        e = rng.randint(s, n - 1)
        total = sum(lengths[i] * truth[i] for i in range(s, e + 1))
        spread = sum(
            (intervals[i][1] - intervals[i][0]) * abs(truth[i])
            for i in range(s, e + 1)
        )
        windows.append(
            {"start_segment": s + 1, "end_segment": e + 1,
             "min_elongation": total - spread - 30,
             "max_elongation": total + spread + 30}
        )
    payload = {
        "segment_lengths": lengths,
        "segment_length_intervals": [
            {"min": lo, "max": hi} for lo, hi in intervals
        ],
        "strain_bounds": {"min": -100000, "max": 100000},
        "windows": windows,
    }
    start = time.monotonic()
    result = invert_payload(payload)
    elapsed = time.monotonic() - start
    assert elapsed < 10.0
    assert len(result["strains"]) == 12
    assert all(c["satisfied"] for c in result["window_checks"])
    assert_witnesses(
        result,
        intervals,
        [
            (w["start_segment"] - 1, w["end_segment"] - 1,
             w["min_elongation"], w["max_elongation"])
            for w in windows
        ],
    )


def test_robust_deterministic():
    rng = random.Random(88)
    n = 6
    lengths = [rng.randint(2, 6) for _ in range(n)]
    intervals = [(max(1, l - 1), l + 1) for l in lengths]
    truth = [rng.randint(-3, 3) for _ in range(n)]
    windows_raw = []
    pairs = [(s, e) for s in range(n) for e in range(s, n)]
    rng.shuffle(pairs)
    for s, e in pairs[:9]:
        total = sum(lengths[i] * truth[i] for i in range(s, e + 1))
        spread = sum(
            (intervals[i][1] - intervals[i][0]) * abs(truth[i])
            for i in range(s, e + 1)
        )
        windows_raw.append((s, e, total - spread - 1, total + spread + 1))
    payload = make_interval_payload(lengths, intervals, -5, 5, windows_raw)
    assert invert_payload(payload) == invert_payload(payload)
