"""海缆连续缆段整数微应变联合反演核心（仅使用 Python 标准库）。

输入
----
- 6..12 段按顺序排列的缆段名义长度（正整数）
- 可选的每段长度正整数闭区间 length_intervals（提交后进入稳健模式：
  每个观测窗对各段长度在区间内任意独立取值都须成立）
- 统一应变闭区间 [strain_min, strain_max]（整数微应变）
- 8..20 个观测窗，每窗给出连续起止段（1 基，含端点）与累计伸长量闭区间

模型（所有比较与运算均为 Python 任意精度精确整数）
--------------------------------------------------
待求逐段应变为 x_0..x_{n-1}（整数微应变）。

观测窗 [s,e] 约束：Σ_{i=s..e} L_i·x_i ∈ [lo, hi]。
每段还须满足 strain_min ≤ x_i ≤ strain_max。

引入前缀和 P_0=0, P_i = Σ_{j<i} L_j·x_j，则
    P_{e+1} - P_s ∈ [lo, hi]
    P_{i+1} - P_i ∈ [L_i·strain_min, L_i·strain_max]
全部是 P 上的**差分约束**，其最长路闭包（Floyd-Warshall）：
- 出现正环 ⇒ 整体不可行（可靠的充分必要判定，针对松弛后的实值域；
  整数可行性仍以下层整数搜索为准）；
- 给出每个 P_i 的精确上下界，进而导出每段 x_i 的紧整数域：
      x_i ≥ ceil( D[i][i+1] / L_i )
      x_i ≤ floor( -D[i+1][i] / L_i )

优选准则（字典序三级，逐级不可放宽）
1. 最小化相邻段最大应变差 M = max_k |x_k - x_{k-1}|
   —— 在 x 空间就是差分约束 x_k - x_{k-1} ∈ [-M, M]，同样并入最长路闭包。
2. 在 1 的最优解中最小化 S = Σ_k |x_k - x_{k-1}|
   —— 辅助变量 e_k ≥ |Δ_k|，以 Σ e_k ≤ S 的线性松弛精确等价判定。
3. 在 1、2 的最优解中取应变序列 (x_0,...,x_{n-1}) 字典序最小者。

求解：最长路闭包缩域 + 通用整数界传播 + MRV/折半回溯；
最小 M、最小 S 及字典序各值均以「可行性关于阈值单调」二分得到。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

MIN_SEGMENTS = 6
MAX_SEGMENTS = 12
MIN_WINDOWS = 8
MAX_WINDOWS = 20


class ValidationErrors(ValueError):
    """输入字段错误（可一次性包含多个字段问题）。"""

    def __init__(self, fields: list[dict[str, str]]):
        self.fields = fields
        super().__init__("; ".join(f"{f['field']}: {f['message']}" for f in fields))


class InfeasibleError(ValueError):
    """全部输入合法，但观测窗彼此冲突（含统一应变界），无可行解。"""


@dataclass(frozen=True)
class Window:
    start: int  # 0 基，含端点
    end: int  # 0 基，含端点
    lo: int  # 累计伸长量（长度加权应变和）闭区间下端
    hi: int  # 闭区间上端


# --------------------------------------------------------------------------- #
# 输入校验
# --------------------------------------------------------------------------- #
def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _err(fields: list[dict[str, str]], field: str, message: str) -> None:
    fields.append({"field": field, "message": message})


def _validate_lengths(payload: dict, fields: list[dict[str, str]]) -> list[int] | None:
    raw = payload.get("segment_lengths")
    if raw is None:
        _err(fields, "segment_lengths", "field is required")
        return None
    if not isinstance(raw, list):
        _err(fields, "segment_lengths", "must be an array of positive integers")
        return None
    lengths: list[int] = []
    for i, item in enumerate(raw):
        if not _is_int(item) or item < 1:
            _err(fields, f"segment_lengths[{i}]", "must be a positive integer")
        else:
            lengths.append(item)
    if not (MIN_SEGMENTS <= len(raw) <= MAX_SEGMENTS):
        _err(
            fields,
            "segment_lengths",
            f"must contain between {MIN_SEGMENTS} and {MAX_SEGMENTS} segments"
            f" (got {len(raw)})",
        )
    return lengths if len(lengths) == len(raw) else None


def _validate_strain_bounds(
    payload: dict, fields: list[dict[str, str]]
) -> tuple[int, int] | None:
    raw = payload.get("strain_bounds")
    if raw is None:
        _err(fields, "strain_bounds", "field is required")
        return None
    if not isinstance(raw, dict):
        _err(fields, "strain_bounds", 'must be an object {"min": int, "max": int}')
        return None
    for key in raw:
        if key not in ("min", "max"):
            _err(fields, f"strain_bounds.{key}", "unknown field")
    lo = raw.get("min")
    hi = raw.get("max")
    ok = True
    if not _is_int(lo):
        _err(fields, "strain_bounds.min", "must be an integer")
        ok = False
    if not _is_int(hi):
        _err(fields, "strain_bounds.max", "must be an integer")
        ok = False
    if ok and lo > hi:
        _err(fields, "strain_bounds", "min must be less than or equal to max")
        return None
    return (lo, hi) if ok else None


def _validate_length_intervals(
    payload: dict, n: int | None, fields: list[dict[str, str]]
) -> list[tuple[int, int]] | None:
    """可选的逐段长度闭区间（正整数）；未提交返回 None（固定长度模式）。"""
    if "length_intervals" not in payload:
        return None
    raw = payload["length_intervals"]
    if raw is None:
        return None
    if not isinstance(raw, list):
        _err(
            fields,
            "length_intervals",
            "must be an array of {\"min\": positive int, \"max\": positive int}",
        )
        return None
    intervals: list[tuple[int, int]] = []
    all_ok = True
    if n is not None and len(raw) != n:
        _err(
            fields,
            "length_intervals",
            f"must contain one interval per segment (expected {n}, got {len(raw)})",
        )
    for i, item in enumerate(raw):
        prefix = f"length_intervals[{i}]"
        if not isinstance(item, dict):
            _err(fields, prefix, 'must be an object {"min": int, "max": int}')
            all_ok = False
            continue
        ok = True
        for key in ("min", "max"):
            if key not in item:
                _err(fields, f"{prefix}.{key}", "field is required")
                ok = False
            elif not _is_int(item[key]):
                _err(fields, f"{prefix}.{key}", "must be an integer")
                ok = False
            elif item[key] < 1:
                _err(fields, f"{prefix}.{key}", "must be a positive integer")
                ok = False
        for key in item:
            if key not in ("min", "max"):
                _err(fields, f"{prefix}.{key}", "unknown field")
                ok = False
        if ok and item["min"] > item["max"]:
            _err(fields, f"{prefix}.min", "must be less than or equal to max")
            ok = False
        if ok:
            intervals.append((item["min"], item["max"]))
        else:
            all_ok = False
    if not all_ok:
        return None
    if n is not None and len(intervals) != n:
        return None
    return intervals


def _validate_windows(
    payload: dict, n: int | None, fields: list[dict[str, str]]
) ->list[Window] | None:
    raw = payload.get("windows")
    if raw is None:
        _err(fields, "windows", "field is required")
        return None
    if not isinstance(raw, list):
        _err(fields, "windows", "must be an array of observation windows")
        return None
    if not (MIN_WINDOWS <= len(raw) <= MAX_WINDOWS):
        _err(
            fields,
            "windows",
            f"must contain between {MIN_WINDOWS} and {MAX_WINDOWS} windows"
            f" (got {len(raw)})",
        )
    windows: list[Window] = []
    all_ok = True
    for i, item in enumerate(raw):
        prefix = f"windows[{i}]"
        if not isinstance(item, dict):
            _err(fields, prefix, "must be an object")
            all_ok = False
            continue
        ok = True
        for key in ("start_segment", "end_segment", "min_elongation", "max_elongation"):
            if key not in item:
                _err(fields, f"{prefix}.{key}", "field is required")
                ok = False
            elif not _is_int(item[key]):
                _err(fields, f"{prefix}.{key}", "must be an integer")
                ok = False
        for key in item:
            if key not in ("start_segment", "end_segment", "min_elongation", "max_elongation"):
                _err(fields, f"{prefix}.{key}", "unknown field")
                ok = False
        if not ok:
            all_ok = False
            continue
        s = item["start_segment"]
        e = item["end_segment"]
        lo = item["min_elongation"]
        hi = item["max_elongation"]
        if n is not None:
            if not (1 <= s <= n):
                _err(fields, f"{prefix}.start_segment", f"must be between 1 and {n}")
                ok = False
            if not (1 <= e <= n):
                _err(fields, f"{prefix}.end_segment", f"must be between 1 and {n}")
                ok = False
            if ok and s > e:
                _err(fields, f"{prefix}.start_segment", "must be <= end_segment")
                ok = False
        if lo > hi:
            _err(
                fields,
                f"{prefix}.min_elongation",
                "must be less than or equal to max_elongation",
            )
            ok = False
        if ok and n is not None:
            windows.append(Window(start=s - 1, end=e - 1, lo=lo, hi=hi))
        else:
            all_ok = False
    return windows if all_ok else None


_ALLOWED_TOP_LEVEL = {
    "segment_lengths",
    "length_intervals",
    "strain_bounds",
    "windows",
}


def validate(
    payload: object,
) -> tuple[list[int], list[tuple[int, int]] | None, int, int, list[Window]]:
    """校验并归一化输入；非法时抛 ValidationErrors。

    返回 (名义长度, 可选长度区间, 应变下界, 应变上界, 观测窗)。
    未提交 length_intervals 时第二元素为 None，调用方须保持固定长度语义。
    """
    fields: list[dict[str, str]] = []
    if not isinstance(payload, dict):
        raise ValidationErrors(
            [{"field": ".", "message": "request body must be a JSON object"}]
        )
    for key in payload:
        if key not in _ALLOWED_TOP_LEVEL:
            _err(fields, key, "unknown field")
    lengths = _validate_lengths(payload, fields)
    n = len(lengths) if lengths is not None else None
    intervals = _validate_length_intervals(payload, n, fields)
    bounds = _validate_strain_bounds(payload, fields)
    windows = _validate_windows(payload, n, fields)
    if fields:
        raise ValidationErrors(fields)
    assert lengths is not None and bounds is not None and windows is not None
    return lengths, intervals, bounds[0], bounds[1], windows


# --------------------------------------------------------------------------- #
# 差分约束最长路闭包
# --------------------------------------------------------------------------- #
def _longest_path_closure(
    node_count: int, edges: list[tuple[int, int, int]]
) -> list[list[int | None]] | None:
    """约束 P_v >= P_u + w 的最长路闭包；存在正环返回 None。

    D[i][j] = P_j - P_i 的最强（最大）下界；None 表示尚无路径（-∞）。
    D[i][i] > 0 即正环，矛盾。以 None 为哨兵可兼容任意精度输入。
    """
    d: list[list[int | None]] = [[None] * node_count for _ in range(node_count)]
    for i in range(node_count):
        d[i][i] = 0
    for u, v, w in edges:
        if d[u][v] is None or w > d[u][v]:
            d[u][v] = w
    for k in range(node_count):
        dk = d[k]
        for i in range(node_count):
            dik = d[i][k]
            if dik is None:
                continue
            di = d[i]
            for j in range(node_count):
                if dk[j] is None:
                    continue
                cand = dik + dk[j]
                if di[j] is None or cand > di[j]:
                    di[j] = cand
    for i in range(node_count):
        if d[i][i] is not None and d[i][i] > 0:
            return None
    return d


def _ceil_div(a: int, b: int) -> int:
    """数学上的 ceil(a / b)（b>0），纯整数。"""
    return -((-a) // b)


def _prefix_closure(
    n: int, lengths: list[int], strain_min: int, strain_max: int,
    windows: list[Window],
) -> list[list[int]] | None:
    """前缀和 P 上的差分约束闭包（窗 + 每段应变界）。"""
    edges: list[tuple[int, int, int]] = []
    for i, length in enumerate(lengths):  # P_{i+1} - P_i ∈ [L·smin, L·smax]
        edges.append((i, i + 1, length * strain_min))
        edges.append((i + 1, i, -length * strain_max))
    for win in windows:  # P_{e+1} - P_s ∈ [lo, hi]
        edges.append((win.start, win.end + 1, win.lo))
        edges.append((win.end + 1, win.start, -win.hi))
    return _longest_path_closure(n + 1, edges)


def _x_bounds_from_prefix(
    n: int,
    lengths: list[int],
    strain_min: int,
    strain_max: int,
    p_closure: list[list[int]],
) -> tuple[list[int], list[int]]:
    """由 P 的最强界导出每段 x_i 的紧整数域。"""
    los: list[int] = []
    his: list[int] = []
    for i, length in enumerate(lengths):
        forward = p_closure[i][i + 1]
        backward = p_closure[i + 1][i]
        # i -> i+1 与 i+1 -> i 均为直接边，闭包后必非 None。
        assert forward is not None and backward is not None
        lo = max(strain_min, _ceil_div(forward, length))
        hi = min(strain_max, (-backward) // length)
        los.append(lo)
        his.append(hi)
    return los, his


def _x_diff_closure(
    n: int, los: list[int], his: list[int], m: int
) -> tuple[list[int], list[int]] | None:
    """并入一元域与 |x_k-x_{k-1}|<=M 后的 x 最长路闭包（锚点节点 n）。"""
    anchor = n
    edges: list[tuple[int, int, int]] = []
    for i in range(n):  # x_i >= anchor + lo_i；anchor >= x_i - hi_i
        edges.append((anchor, i, los[i]))
        edges.append((i, anchor, -his[i]))
    for k in range(1, n):
        edges.append((k - 1, k, -m))  # x_k >= x_{k-1} - M
        edges.append((k, k - 1, -m))  # x_{k-1} >= x_k - M
    closure = _longest_path_closure(n + 1, edges)
    if closure is None:
        return None
    out_lo = [closure[anchor][i] if closure[anchor][i] is not None else los[i] for i in range(n)]
    out_hi = [-closure[i][anchor] if closure[i][anchor] is not None else his[i] for i in range(n)]
    for i in range(n):
        if out_lo[i] > out_hi[i]:
            return None
    return out_lo, out_hi


# --------------------------------------------------------------------------- #
# 通用整数界传播 + 回溯（x 空间）
# --------------------------------------------------------------------------- #
# 约束：sum_j w_j * var_j ∈ [lo, hi]，端点可为 None。
Constraint = tuple[tuple[tuple[int, int], ...], int | None, int | None]


def _propagate(
    los: list[int], his: list[int], constraints: list[Constraint]
) -> tuple[list[int], list[int]] | None:
    """对所有线性区间约束反复做一元界传播；矛盾返回 None。"""
    los = list(los)
    his = list(his)
    while True:
        changed = False
        for terms, a, b in constraints:
            for j, w in terms:
                # 即使域已为单点也不能跳过：后加入的约束可能与其冲突。
                others_min = 0
                others_max = 0
                for j2, w2 in terms:
                    if j2 == j:
                        continue
                    if w2 > 0:
                        others_min += w2 * los[j2]
                        others_max += w2 * his[j2]
                    else:
                        others_min += w2 * his[j2]
                        others_max += w2 * los[j2]
                if b is not None:  # w*x_j <= b - others_min
                    c = b - others_min
                    if w > 0:
                        v = c // w
                        if v < his[j]:
                            his[j] = v
                            changed = True
                    else:
                        v = _ceil_div(c, w)
                        if v > los[j]:
                            los[j] = v
                            changed = True
                if a is not None:  # w*x_j >= a - others_max
                    c = a - others_max
                    if w > 0:
                        v = _ceil_div(c, w)
                        if v > los[j]:
                            los[j] = v
                            changed = True
                    else:
                        v = c // w
                        if v < his[j]:
                            his[j] = v
                            changed = True
                if los[j] > his[j]:
                    return None
        if not changed:
            return los, his


def _search(
    los: list[int], his: list[int], constraints: list[Constraint]
) -> tuple[int, ...] | None:
    """找一个可行赋值（MRV 变量序，域折半分支）；无可行解返回 None。"""
    narrowed = _propagate(los, his, constraints)
    if narrowed is None:
        return None
    los, his = narrowed
    best = -1
    best_size = 0
    for j in range(len(los)):
        if los[j] != his[j]:
            size = his[j] - los[j]
            if best == -1 or size < best_size:
                best = j
                best_size = size
    if best == -1:
        return tuple(los)
    j = best
    mid = (los[j] + his[j]) // 2
    lo2, hi2 = list(los), list(his)
    hi2[j] = mid
    result = _search(lo2, hi2, constraints)
    if result is not None:
        return result
    lo3, hi3 = list(los), list(his)
    lo3[j] = mid + 1
    return _search(lo3, hi3, constraints)


# --------------------------------------------------------------------------- #
# 反演
# --------------------------------------------------------------------------- #
def _window_constraints(n: int, lengths: list[int], windows: list[Window]) -> list[Constraint]:
    cons: list[Constraint] = []
    for win in windows:
        terms = tuple(
            (i, lengths[i]) for i in range(win.start, win.end + 1)
        )
        cons.append((terms, win.lo, win.hi))
    return cons


def _diff_constraints(n: int, m: int) -> list[Constraint]:
    """|x_k - x_{k-1}| <= M（亦供通用传播，与差分闭包一致）。"""
    cons: list[Constraint] = []
    for k in range(1, n):
        cons.append((((k, 1), (k - 1, -1)), -m, m))
    return cons


def _abs_sum_constraints(n: int, m: int, s_lo: int | None, s_hi: int | None) -> list[Constraint]:
    """e_k >= x_k-x_{k-1}、e_k >= x_{k-1}-x_k、0<=e_k<=M、Σe_k∈[s_lo,s_hi]。"""
    cons: list[Constraint] = []
    for k in range(1, n):
        e_var = n + k - 1
        cons.append((((e_var, 1), (k, -1), (k - 1, 1)), 0, None))
        cons.append((((e_var, 1), (k, 1), (k - 1, -1)), 0, None))
        cons.append((((e_var, 1),), 0, m))
    cons.append((tuple((n + k - 1, 1) for k in range(1, n)), s_lo, s_hi))
    return cons


def invert_payload(payload: object) -> dict:
    """完整反演，返回可直接复核的结果字典；校验失败/不可行抛对应异常。

    未提交 ``length_intervals`` 时严格保持原有固定长度语义与响应结构；
    提交后切换为稳健模式：每个观测窗对长度盒内任意独立取值都必须成立。
    """
    lengths, intervals, strain_min, strain_max, windows = validate(payload)
    if intervals is None:
        strains, diffs, best_m, best_s = _solve_fixed(
            lengths, strain_min, strain_max, windows
        )
        return _build_fixed_result(lengths, windows, strains, diffs, best_m, best_s)
    strains, diffs, best_m, best_s = _solve_robust(
        lengths, intervals, strain_min, strain_max, windows
    )
    return _build_robust_result(
        lengths, intervals, windows, strains, diffs, best_m, best_s
    )


def _solve_fixed(
    lengths: list[int], strain_min: int, strain_max: int, windows: list[Window]
) -> tuple[list[int], list[int], int, int]:
    """固定名义长度下的三级字典序最优反演（原逻辑）。"""
    n = len(lengths)
    window_cons = _window_constraints(n, lengths, windows)

    # 前缀和差分闭包：与 M 无关，只算一次。
    p_closure = _prefix_closure(n, lengths, strain_min, strain_max, windows)
    if p_closure is None:
        raise InfeasibleError("observation windows are mutually inconsistent")
    base_x_lo, base_x_hi = _x_bounds_from_prefix(
        n, lengths, strain_min, strain_max, p_closure
    )
    full_range = strain_max - strain_min

    def feasible(m: int, s_hi: int | None = None, pin: tuple[int, int] | None = None) -> bool:
        """|Δ|<=m（若给 s_hi 则 Σ|Δ|<=s_hi），可附加 x_var<=t 的钉压。"""
        bounds = _x_diff_closure(n, base_x_lo, base_x_hi, m)
        if bounds is None:
            return False
        x_lo, x_hi = bounds
        constraints = list(window_cons)
        constraints += _diff_constraints(n, m)
        if s_hi is not None:
            x_lo = x_lo + [0] * (n - 1)
            x_hi = x_hi + [m] * (n - 1)
            constraints += _abs_sum_constraints(n, m, None, s_hi)
        if pin is not None:
            var, t = pin
            constraints.append((((var, 1),), None, t))
        return _search(list(x_lo), list(x_hi), constraints) is not None

    # 阶段 0：无平滑约束（M = 全量程）下的整数可行性。
    if not feasible(full_range):
        raise InfeasibleError("observation windows are mutually inconsistent")

    # 阶段 1：二分最小可行 M。
    m_lo, m_hi = 0, full_range
    while m_lo < m_hi:
        mid = (m_lo + m_hi) // 2
        if feasible(mid):
            m_hi = mid
        else:
            m_lo = mid + 1
    best_m = m_lo

    # 阶段 2：二分最小可行 S = Σ|Δ_k|。
    s_lo, s_hi = 0, (n - 1) * best_m
    while s_lo < s_hi:
        mid = (s_lo + s_hi) // 2
        if feasible(best_m, s_hi=mid):
            s_hi = mid
        else:
            s_lo = mid + 1
    best_s = s_lo

    # 阶段 3：逐段钉死字典序最小值（x_0, x_1, ... 顺序；可行性关于阈值单调）。
    bounds = _x_diff_closure(n, base_x_lo, base_x_hi, best_m)
    assert bounds is not None
    x_lo, x_hi = bounds
    x_lo = x_lo + [0] * (n - 1)
    x_hi = x_hi + [best_m] * (n - 1)
    constraints = list(window_cons)
    constraints += _diff_constraints(n, best_m)
    constraints += _abs_sum_constraints(n, best_m, best_s, best_s)

    strains: list[int] = []
    for var in range(n):
        t_lo, t_hi = x_lo[var], x_hi[var]
        while t_lo < t_hi:
            mid = (t_lo + t_hi) // 2
            trial = constraints + [(((var, 1),), None, mid)]
            if _search(list(x_lo), list(x_hi), trial) is not None:
                t_hi = mid
            else:
                t_lo = mid + 1
        strains.append(t_lo)
        x_lo[var] = x_hi[var] = t_lo
        narrowed = _propagate(x_lo, x_hi, constraints)
        assert narrowed is not None  # 已选最优值必然可行
        x_lo, x_hi = narrowed

    diffs = [strains[i] - strains[i - 1] for i in range(1, n)]
    return strains, diffs, best_m, best_s
# 稳健模式：符号分支穷举
# --------------------------------------------------------------------------- #
# 给定 x 时窗 w 的稳健极值回算和是分段线性（精确整数）：
#   F_min(x)=Σ_{x_i≥0}a_i x_i+Σ_{x_i<0}b_i x_i ≥ lo
#   F_max(x)=Σ_{x_i≥0}b_i x_i+Σ_{x_i<0}a_i x_i ≤ hi
# 线性函数在整数盒上极值必在顶点取得，故对“各段长度在各自区间内任意独立
# 取值”充分必要。按每段应变正负把可行集切成至多 2^n 个符号分支（n≤12，
# ≤4096 个，完整覆盖而非抽查长度组合）；每分支内 F_min/F_max 均为普通线性
# 整数区间约束，复用固定长度求解器同一套传播/回溯。
def _sign_window_constraints(
    n: int,
    intervals: list[tuple[int, int]],
    windows: list[Window],
    sign: tuple[bool, ...],
) -> list[Constraint]:
    """固定符号（True:x_i≥0 / False:x_i≤-1）下的稳健窗线性约束。"""
    cons: list[Constraint] = []
    for win in windows:
        min_terms = tuple(
            (i, intervals[i][0] if sign[i] else intervals[i][1])
            for i in range(win.start, win.end + 1)
        )
        max_terms = tuple(
            (i, intervals[i][1] if sign[i] else intervals[i][0])
            for i in range(win.start, win.end + 1)
        )
        cons.append((min_terms, win.lo, None))
        cons.append((max_terms, None, win.hi))
    return cons


def _solve_robust(
    nominal: list[int],
    intervals: list[tuple[int, int]],
    strain_min: int,
    strain_max: int,
    windows: list[Window],
) -> tuple[list[int], list[int], int, int]:
    """长度区间稳健模式下的三级字典序最优反演。"""
    n = len(nominal)
    full_range = strain_max - strain_min
    patterns = list(itertools.product((True, False), repeat=n))

    # 与 M、S 无关的稳健窗可行性先全符号分支筛一次。每分支有两组线性系数
    # （下界窗系数 cmin、上界窗系数 cmax），分别在“双侧窗区间”上做前缀差分
    # 闭包并取交集，得到该符号分支内既保留 Σcmin·x≥lo 强下界、又保留
    # Σcmax·x≤hi 强上界的精确一元域（与固定长度求解器的前缀闭包同强度）。
    # 结果与 M/S 无关，只算一次；后续所有判定只枚举幸存符号分支。
    sign_base: dict[tuple[bool, ...], tuple[list[int], list[int]]] = {}

    def one_sided_x_bounds(
        coeff: list[int], use_lower: bool
    ) -> tuple[list[int], list[int]] | None:
        """仅并入下界窗（Σcoeff·x≥lo）或上界窗（Σcoeff·x≤hi）的前缀闭包。

        直接构造差分边，避免用极大哨兵区间；保持全程任意精度整数。
        """
        edges: list[tuple[int, int, int]] = []
        for i, length in enumerate(coeff):
            edges.append((i, i + 1, length * strain_min))
            edges.append((i + 1, i, -length * strain_max))
        for win in windows:
            if use_lower:
                edges.append((win.start, win.end + 1, win.lo))
            else:
                edges.append((win.end + 1, win.start, -win.hi))
        closure = _longest_path_closure(n + 1, edges)
        if closure is None:
            return None
        return _x_bounds_from_prefix(n, coeff, strain_min, strain_max, closure)

    for sign in patterns:
        cmin = [intervals[i][0] if sign[i] else intervals[i][1] for i in range(n)]
        cmax = [intervals[i][1] if sign[i] else intervals[i][0] for i in range(n)]
        # cmin 只承载下界窗 Σcmin·x≥lo；cmax 只承载上界窗 Σcmax·x≤hi。
        bmin = one_sided_x_bounds(cmin, True)
        bmax = one_sided_x_bounds(cmax, False)
        if bmin is None or bmax is None:
            continue
        x_lo: list[int] = []
        x_hi: list[int] = []
        ok = True
        for i in range(n):
            lo = max(bmin[0][i], bmax[0][i], 0 if sign[i] else strain_min)
            hi = min(bmin[1][i], bmax[1][i], strain_max if sign[i] else -1)
            if lo > hi:
                ok = False
                break
            x_lo.append(lo)
            x_hi.append(hi)
        if ok:
            # 再跑一次含全部窗约束（下界与上界同系数）的通用线性传播，补上
            # 两个前缀闭包无法单独表达的“联合剪枝”；两者取强。
            seed_lo = [0 if sign[i] else strain_min for i in range(n)]
            seed_hi = [strain_max if sign[i] else -1 for i in range(n)]
            joint = _propagate(
                seed_lo, seed_hi, _sign_window_constraints(n, intervals, windows, sign)
            )
            if joint is not None:
                x_lo = [max(x_lo[i], joint[0][i]) for i in range(n)]
                x_hi = [min(x_hi[i], joint[1][i]) for i in range(n)]
                if all(x_lo[i] <= x_hi[i] for i in range(n)):
                    sign_base[sign] = (x_lo, x_hi)
    if not sign_base:
        raise InfeasibleError(
            "no strain sequence is robustly feasible for the submitted length intervals"
        )

    def pattern_setup(
        sign: tuple[bool, ...],
        m: int,
        s_lo: int | None,
        s_hi: int | None,
        fixed_eq: list[Constraint],
        upper: tuple[int, int] | None,
    ) -> tuple[list[int], list[int], list[Constraint], int] | None:
        """构建并纯传播一个符号分支；返回收紧后域、约束与域总宽。"""
        base_lo, base_hi = sign_base[sign]
        # 关键：在“已含稳健窗”的一元域上跑 |Δ|≤M 差分闭包，得到窗与平滑
        # 联合的传递收紧（等价于固定长度求解器先做窗口前缀闭包再并入 M）。
        pattern_closure = _x_diff_closure(n, base_lo, base_hi, m)
        if pattern_closure is None:
            return None
        c_lo, c_hi = pattern_closure
        x_lo: list[int] = []
        x_hi: list[int] = []
        for i in range(n):
            lo = c_lo[i]
            hi = c_hi[i]
            if lo > hi:
                return None
            x_lo.append(lo)
            x_hi.append(hi)
        cons = _sign_window_constraints(n, intervals, windows, sign)
        cons += _diff_constraints(n, m)
        use_s = s_lo is not None or s_hi is not None
        if use_s:
            x_lo = x_lo + [0] * (n - 1)
            x_hi = x_hi + [m] * (n - 1)
            cons += _abs_sum_constraints(n, m, s_lo, s_hi)
        cons += fixed_eq
        if upper is not None:
            cons += [(((upper[0], 1),), None, upper[1])]
        narrowed = _propagate(list(x_lo), list(x_hi), cons)
        if narrowed is None:
            return None
        nlo, nhi = narrowed
        width = sum(nhi[i] - nlo[i] for i in range(n))
        return nlo, nhi, cons, width

    def feasible(
        m: int,
        s_lo: int | None = None,
        s_hi: int | None = None,
        fixed_eq: list[Constraint] | None = None,
        upper: tuple[int, int] | None = None,
    ) -> tuple[int, ...] | None:
        """跨符号分支：先纯传播筛掉不可行分支，幸存分支按收紧域总宽排序
        （最紧、最易成功的先搜），再依次精确回溯，首个可行即返回。"""
        fixed_eq = fixed_eq or []
        # 已钉死前缀（及上界 t<0）直接确定部分段符号，先据此剪符号分支。
        forced_sign: dict[int, bool] = {}
        for terms, lo, hi in fixed_eq:
            if lo is not None and lo == hi and len(terms) == 1:
                var = terms[0][0]
                forced_sign[var] = lo >= 0
        if upper is not None and upper[1] < 0:
            forced_sign[upper[0]] = False
        survivors: list[tuple[int, list[int], list[int], list[Constraint]]] = []
        for sign, base in sign_base.items():
            if any(sign[i] != want for i, want in forced_sign.items()):
                continue
            setup = pattern_setup(sign, m, s_lo, s_hi, fixed_eq, upper)
            if setup is not None:
                x_lo, x_hi, cons, width = setup
                survivors.append((width, x_lo, x_hi, cons))
        survivors.sort(key=lambda z: z[0])
        for _width, x_lo, x_hi, cons in survivors:
            found = _search(list(x_lo), list(x_hi), cons)
            if found is not None:
                return found
        return None

    # 阶段 0：无平滑约束下的稳健整数可行性。
    if feasible(full_range) is None:
        raise InfeasibleError(
            "no strain sequence is robustly feasible for the submitted length intervals"
        )

    # 阶段 1：二分最小可行 M（跨所有符号分支，关于 M 单调）。
    m_lo, m_hi = 0, full_range
    while m_lo < m_hi:
        mid = (m_lo + m_hi) // 2
        if feasible(mid) is not None:
            m_hi = mid
        else:
            m_lo = mid + 1
    best_m = m_lo

    # 阶段 2：二分最小可行 S = Σ|Δ_k|。先取一个 M* 可行见证，以其实际 S
    # 作为上界（远紧于 (n-1)·M*），减少二分次数与每次的回溯规模。
    m_witness = feasible(best_m)
    assert m_witness is not None
    mw = list(m_witness[:n])
    s_upper = sum(abs(mw[k] - mw[k - 1]) for k in range(1, n))
    s_lo, s_hi = 0, s_upper
    while s_lo < s_hi:
        mid = (s_lo + s_hi) // 2
        if feasible(best_m, s_hi=mid) is not None:
            s_hi = mid
        else:
            s_lo = mid + 1
    best_s = s_lo

    # 阶段 3：逐段钉死全局字典序最小值。x_var≤t 的可行性关于 t 单调；
    # 已固定前缀以等式约束随每步一起下发，跨所有符号分支取见证。先用阶段 2
    # 的一个可行见证收紧每段上界（其逐段值即字典序上界），减少二分跨度。
    witness = feasible(best_m, best_s, best_s)
    assert witness is not None
    incumbent = list(witness[:n])
    fixed_eq: list[Constraint] = []
    strains: list[int] = []
    any_closure = _x_diff_closure(n, [strain_min] * n, [strain_max] * n, best_m)
    assert any_closure is not None
    var_lo_all, var_hi_all = any_closure
    for var in range(n):
        t_lo = var_lo_all[var]
        t_hi = min(var_hi_all[var], incumbent[var])
        while t_lo < t_hi:
            mid = (t_lo + t_hi) // 2
            found = feasible(
                best_m, best_s, best_s, fixed_eq, upper=(var, mid)
            )
            if found is not None:
                t_hi = mid
                incumbent = list(found[:n])
            else:
                t_lo = mid + 1
        strains.append(t_lo)
        fixed_eq.append((((var, 1),), t_lo, t_lo))

    diffs = [strains[i] - strains[i - 1] for i in range(1, n)]
    return strains, diffs, best_m, best_s




def _build_fixed_result(
    lengths: list[int],
    windows: list[Window],
    strains: list[int],
    diffs: list[int],
    best_m: int,
    best_s: int,
) -> dict:
    prefix = [0]
    for length in lengths:
        prefix.append(prefix[-1] + length)
    window_checks = []
    for idx, win in enumerate(windows):
        weighted_sum = 0
        for i in range(win.start, win.end + 1):
            weighted_sum += lengths[i] * strains[i]
        total_length = prefix[win.end + 1] - prefix[win.start]
        window_checks.append(
            {
                "index": idx,
                "start_segment": win.start + 1,
                "end_segment": win.end + 1,
                "total_length": total_length,
                "min_elongation": win.lo,
                "max_elongation": win.hi,
                "weighted_strain_sum": weighted_sum,
                "satisfied": win.lo <= weighted_sum <= win.hi,
            }
        )
    # 两级平滑指标均由相邻差直接复核（重算以自证，不直接采用搜索内部值）。
    recomputed_m = max(abs(d) for d in diffs)
    recomputed_s = sum(abs(d) for d in diffs)
    assert recomputed_m == best_m and recomputed_s == best_s
    return {
        "segment_count": len(lengths),
        "strains": strains,
        "adjacent_diffs": diffs,
        "objectives": {
            "max_adjacent_diff": recomputed_m,
            "sum_adjacent_abs_diff": recomputed_s,
        },
        "window_checks": window_checks,
        "criteria_order": [
            "max_adjacent_diff",
            "sum_adjacent_abs_diff",
            "lexicographic",
        ],
    }


def _build_robust_result(
    nominal: list[int],
    intervals: list[tuple[int, int]],
    windows: list[Window],
    strains: list[int],
    diffs: list[int],
    best_m: int,
    best_s: int,
) -> dict:
    """稳健模式结果：每窗给出精确整数极值回算和与取得极值的长度端点见证。"""
    n = len(nominal)
    window_checks: list[dict] = []
    for idx, win in enumerate(windows):
        f_min = 0
        f_max = 0
        total_lo = 0
        total_hi = 0
        min_witness: list[int] = []
        max_witness: list[int] = []
        for i in range(win.start, win.end + 1):
            a, b = intervals[i]
            x = strains[i]
            total_lo += a
            total_hi += b
            if x >= 0:  # 非负应变：长度越大和越大
                f_min += a * x
                f_max += b * x
                min_witness.append(a)
                max_witness.append(b)
            else:  # 负应变：长度越大和越小
                f_min += b * x
                f_max += a * x
                min_witness.append(b)
                max_witness.append(a)
        satisfied = win.lo <= f_min and f_max <= win.hi
        assert satisfied, "solver returned a non-robust strain sequence"
        window_checks.append(
            {
                "index": idx,
                "start_segment": win.start + 1,
                "end_segment": win.end + 1,
                "total_length_interval": [total_lo, total_hi],
                "min_elongation": win.lo,
                "max_elongation": win.hi,
                "robust_min_weighted_sum": f_min,
                "robust_max_weighted_sum": f_max,
                # 按窗内段序（start_segment..end_segment）给出取得对应极值时
                # 各段所用的长度端点，可与应变逐段相乘直接复核极值。
                "min_extremum_lengths": min_witness,
                "max_extremum_lengths": max_witness,
                "robustly_satisfied": satisfied,
            }
        )
    recomputed_m = max(abs(d) for d in diffs)
    recomputed_s = sum(abs(d) for d in diffs)
    assert recomputed_m == best_m and recomputed_s == best_s
    return {
        "mode": "robust_interval",
        "segment_count": n,
        "length_intervals": [[a, b] for a, b in intervals],
        "strains": strains,
        "adjacent_diffs": diffs,
        "objectives": {
            "max_adjacent_diff": recomputed_m,
            "sum_adjacent_abs_diff": recomputed_s,
        },
        "window_checks": window_checks,
        "criteria_order": [
            "max_adjacent_diff",
            "sum_adjacent_abs_diff",
            "lexicographic",
        ],
    }
