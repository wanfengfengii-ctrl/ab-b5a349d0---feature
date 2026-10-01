"""海缆连续缆段整数微应变联合反演核心（仅使用 Python 标准库）。

输入
----
- 6..12 段按顺序排列的缆段长度（正整数）
- 统一应变闭区间 [strain_min, strain_max]（整数微应变）
- 8..20 个观测窗，每窗给出连续起止段（1 基，含端点）与累计伸长量闭区间

模型（所有比较与运算均为 Python 任意精度精确整数）
--------------------------------------------------
待求逐段应变为 x_0..x_{n-1}（整数微应变）。

观测窗 [s,e] 约束：Σ_{i=s..e} L_i·x_i ∈ [lo, hi]。
每段还须满足 strain_min ≤ x_i ≤ strain_max。

可选长度闭区间（稳健模式）
--------------------------
现场标距长度存在整数测量误差时，可为每段 i 额外提交正整数闭区间
[lo_i, hi_i]（必须包含名义长度 L_i）。此时每个观测窗必须对**各段长度
在各自区间内的任意独立整数取值**都满足累计伸长闭区间。长度加权和
关于各 L_i 线性，极值必在端点取得（x_i>0 时最小和取 lo_i、最大和取
hi_i；x_i<0 时反之），故稳健条件等价于

    Σ_i f^-_i(x_i) >= 窗下端点   且   Σ_i f^+_i(x_i) <= 窗上端点
    f^-_i(x) = lo_i·x (x>=0) / hi_i·x (x<0) —— 关于 x 单调递增
    f^+_i(x) = hi_i·x (x>=0) / lo_i·x (x<0) —— 关于 x 单调递增

f^± 分段线性且单调，可按段做精确整数界传播，直接嵌入通用传播/回溯
骨架；名义长度前缀闭包仍是稳健可行集的安全超集（区间必含名义值），
缩域逻辑原样复用。未提交区间时，请求、三级裁决与响应与原版完全一致。

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


def _validate_length_intervals(
    payload: dict, lengths: list[int] | None, fields: list[dict[str, str]]
) -> list[tuple[int, int]] | None:
    """可选的各段正整数长度闭区间；提交后启用稳健模式。

    每个区间须为正整数闭区间（min<=max，均 >=1）且包含对应名义长度；
    任何非法都定位到 segment_length_intervals[i] 的具体字段。
    """
    raw = payload.get("segment_length_intervals")
    if raw is None:
        return None
    if not isinstance(raw, list):
        _err(
            fields,
            "segment_length_intervals",
            'must be an array of {"min": int, "max": int}, one per segment',
        )
        return None
    if lengths is not None and len(raw) != len(lengths):
        _err(
            fields,
            "segment_length_intervals",
            f"must contain exactly {len(lengths)} entries, one per segment"
            f" (got {len(raw)})",
        )
        return None
    intervals: list[tuple[int, int]] = []
    all_ok = True
    for i, item in enumerate(raw):
        prefix = f"segment_length_intervals[{i}]"
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
        for key in item:
            if key not in ("min", "max"):
                _err(fields, f"{prefix}.{key}", "unknown field")
                ok = False
        if not ok:
            all_ok = False
            continue
        lo, hi = item["min"], item["max"]
        if lo < 1:
            _err(fields, f"{prefix}.min", "must be a positive integer")
            ok = False
        if hi < 1:
            _err(fields, f"{prefix}.max", "must be a positive integer")
            ok = False
        if ok and lo > hi:
            _err(fields, f"{prefix}.min", "must be less than or equal to max")
            ok = False
        if (
            ok
            and lengths is not None
            and i < len(lengths)
            and not (lo <= lengths[i] <= hi)
        ):
            _err(
                fields,
                prefix,
                f"must contain the nominal length segment_lengths[{i}]"
                f" ({lengths[i]})",
            )
            ok = False
        if ok:
            intervals.append((lo, hi))
        else:
            all_ok = False
    return intervals if all_ok else None


_ALLOWED_TOP_LEVEL = {
    "segment_lengths",
    "segment_length_intervals",
    "strain_bounds",
    "windows",
}


def validate(payload: object) -> tuple[
    list[int], int, int, list[Window], list[tuple[int, int]] | None
]:
    """校验并归一化输入；非法时抛 ValidationErrors。

    返回 (段长, 应变下限, 应变上限, 观测窗, 长度闭区间|None)。
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
    bounds = _validate_strain_bounds(payload, fields)
    n = len(lengths) if lengths is not None else None
    windows = _validate_windows(payload, n, fields)
    intervals = _validate_length_intervals(payload, lengths, fields)
    if fields:
        raise ValidationErrors(fields)
    assert lengths is not None and bounds is not None and windows is not None
    return lengths, bounds[0], bounds[1], windows, intervals


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

# 稳健窗约束：(段索引元组, 各段长度下界, 各段长度上界, 窗下端点|None, 窗上端点|None)。
# 语义：各段长度在 [lo_len, hi_len] 内任取独立整数时，窗内长度加权应变和
# 都必须落在 [窗下端点, 窗上端点]。等价于
#   Σ f^-_i(x_i) >= 窗下端点 且  Σ f^+_i(x_i) <= 窗上端点，
# 其中 f^-_i(x) = lo_len_i·x (x>=0) / hi_len_i·x (x<0)，
#       f^+_i(x) = hi_len_i·x (x>=0) / lo_len_i·x (x<0)，
# 两者都是单调递增的分段线性函数（斜率均为正，0 处连续且为 0）。
RobustConstraint = tuple[
    tuple[int, ...], tuple[int, ...], tuple[int, ...], int | None, int | None
]


def _propagate_robust(
    los: list[int], his: list[int], robust: list[RobustConstraint]
) -> bool | None:
    """稳健窗约束的精确整数界传播（就地收紧）；矛盾返回 None，否则返回是否有收紧。"""
    changed = False
    for idx, lo_len, hi_len, a, b in robust:
        if a is not None:
            # f^- 单调递增 => 各变量域上的最大值取在 x = his[i]。
            best = []
            for t, i in enumerate(idx):
                h = his[i]
                best.append(lo_len[t] * h if h >= 0 else hi_len[t] * h)
            total = sum(best)
            if total < a:
                return None
            for t, i in enumerate(idx):
                # 需 f^-_i(x) >= a - Σ_{j≠i} max f^-_j（其余项取当前域最大值）。
                r = a - (total - best[t])
                # f^-(x) >= r 的整数解集：r<=0 时 x >= ceil(r/hi_len)（负支），
                # r>0 时 x >= ceil(r/lo_len)（正支）。
                bound = _ceil_div(r, hi_len[t]) if r <= 0 else _ceil_div(r, lo_len[t])
                if bound > los[i]:
                    los[i] = bound
                    changed = True
                    if los[i] > his[i]:
                        return None
        if b is not None:
            # f^+ 单调递增 => 各变量域上的最小值取在 x = los[i]。
            best = []
            for t, i in enumerate(idx):
                l = los[i]
                best.append(hi_len[t] * l if l >= 0 else lo_len[t] * l)
            total = sum(best)
            if total > b:
                return None
            for t, i in enumerate(idx):
                # 需 f^+_i(x) <= b - Σ_{j≠i} min f^+_j（其余项取当前域最小值）。
                r = b - (total - best[t])
                # f^+(x) <= r 的整数解集：r>=0 时 x <= floor(r/hi_len)（正支），
                # r<0 时 x <= floor(r/lo_len)（负支）。
                bound = r // hi_len[t] if r >= 0 else r // lo_len[t]
                if bound < his[i]:
                    his[i] = bound
                    changed = True
                    if los[i] > his[i]:
                        return None
    return changed


def _propagate(
    los: list[int],
    his: list[int],
    constraints: list[Constraint],
    robust: list[RobustConstraint] | tuple[RobustConstraint, ...] = (),
) -> tuple[list[int], list[int]] | None:
    """对所有线性区间约束与稳健窗约束反复做一元界传播；矛盾返回 None。"""
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
        robust_changed = _propagate_robust(los, his, robust)
        if robust_changed is None:
            return None
        changed = changed or robust_changed
        if not changed:
            return los, his


def _search(
    los: list[int],
    his: list[int],
    constraints: list[Constraint],
    robust: list[RobustConstraint] | tuple[RobustConstraint, ...] = (),
) -> tuple[int, ...] | None:
    """找一个可行赋值（MRV 变量序，域折半分支）；无可行解返回 None。"""
    narrowed = _propagate(los, his, constraints, robust)
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
    result = _search(lo2, hi2, constraints, robust)
    if result is not None:
        return result
    lo3, hi3 = list(los), list(his)
    lo3[j] = mid + 1
    return _search(lo3, hi3, constraints, robust)


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


def _robust_constraints(
    windows: list[Window], intervals: list[tuple[int, int]]
) -> list[RobustConstraint]:
    """稳健窗约束：窗内各段长度在提交闭区间内任意独立取值都须满足窗区间。"""
    cons: list[RobustConstraint] = []
    for win in windows:
        idx = tuple(range(win.start, win.end + 1))
        lo_len = tuple(intervals[i][0] for i in idx)
        hi_len = tuple(intervals[i][1] for i in idx)
        cons.append((idx, lo_len, hi_len, win.lo, win.hi))
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
    """完整反演，返回可直接复核的结果字典；校验失败/不可行抛对应异常。"""
    lengths, strain_min, strain_max, windows, intervals = validate(payload)
    n = len(lengths)
    window_cons = _window_constraints(n, lengths, windows)
    # 稳健模式：另加「任意长度取值」窗约束；名义窗约束保留作冗余传播加强
    # （稳健可行 ⟹ 名义可行，因为区间必含名义长度）。
    robust_cons: list[RobustConstraint] = (
        _robust_constraints(windows, intervals) if intervals is not None else []
    )

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
        return _search(list(x_lo), list(x_hi), constraints, robust_cons) is not None

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
            if _search(list(x_lo), list(x_hi), trial, robust_cons) is not None:
                t_hi = mid
            else:
                t_lo = mid + 1
        strains.append(t_lo)
        x_lo[var] = x_hi[var] = t_lo
        narrowed = _propagate(x_lo, x_hi, constraints, robust_cons)
        assert narrowed is not None  # 已选最优值必然可行
        x_lo, x_hi = narrowed

    diffs = [strains[i] - strains[i - 1] for i in range(1, n)]
    return _build_result(lengths, windows, strains, diffs, best_m, best_s, intervals)


def _build_result(
    lengths: list[int],
    windows: list[Window],
    strains: list[int],
    diffs: list[int],
    best_m: int,
    best_s: int,
    intervals: list[tuple[int, int]] | None = None,
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
        check = {
            "index": idx,
            "start_segment": win.start + 1,
            "end_segment": win.end + 1,
            "total_length": total_length,
            "min_elongation": win.lo,
            "max_elongation": win.hi,
            "weighted_strain_sum": weighted_sum,
            "satisfied": win.lo <= weighted_sum <= win.hi,
        }
        if intervals is not None:
            # 按应变正负取长度端点，推导该窗最小/最大可能回算和（精确整数），
            # 并记录取得两端极值时各段所用的长度端点作为见证。
            w_min = 0
            w_max = 0
            lengths_at_min: list[int] = []
            lengths_at_max: list[int] = []
            for i in range(win.start, win.end + 1):
                x = strains[i]
                lo_i, hi_i = intervals[i]
                if x >= 0:
                    lengths_at_min.append(lo_i)
                    lengths_at_max.append(hi_i)
                    w_min += lo_i * x
                    w_max += hi_i * x
                else:
                    lengths_at_min.append(hi_i)
                    lengths_at_max.append(lo_i)
                    w_min += hi_i * x
                    w_max += lo_i * x
            check["min_possible_weighted_sum"] = w_min
            check["max_possible_weighted_sum"] = w_max
            check["lengths_at_min"] = lengths_at_min
            check["lengths_at_max"] = lengths_at_max
            # 稳健满足：任意长度取值下的回算和都落在提交闭区间内。
            check["satisfied"] = win.lo <= w_min and w_max <= win.hi
        window_checks.append(check)
    # 两级平滑指标均由相邻差直接复核（重算以自证，不直接采用搜索内部值）。
    recomputed_m = max(abs(d) for d in diffs)
    recomputed_s = sum(abs(d) for d in diffs)
    assert recomputed_m == best_m and recomputed_s == best_s
    result = {
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
    if intervals is not None:
        result["length_mode"] = "interval"
        result["segment_length_intervals"] = [
            {"min": lo, "max": hi} for lo, hi in intervals
        ]
    return result
