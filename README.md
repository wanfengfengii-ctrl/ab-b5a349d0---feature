# 海缆连续缆段整数微应变联合反演服务

海缆检修组只能取得多个**重叠标距**的累计伸长量。本服务把 6–12 段缆段的
整数微应变作为未知量，对 8–20 个由「连续起止段 + 累计伸长量闭区间」组成的
观测窗做**精确整数联合反演**，避免逐窗均摊导致彼此矛盾的局部结论。

## 数学模型与优选准则

设第 i 段名义长度为 L_i、待求整数微应变为 x_i。每个观测窗 [s,e] 要求

    Σ_{i=s..e} L_i · x_i ∈ [min_elongation, max_elongation]

且每段都满足统一应变闭区间 `strain_bounds.min ≤ x_i ≤ strain_bounds.max`。
所有运算与比较均为 Python 任意精度**整数**，无任何浮点参与。

### 可选：逐段稳健长度区间 `length_intervals`

可在请求中为每根缆段可选提交一个正整数长度闭区间
`length_intervals[i] = {"min": a_i, "max": b_i}`（与 `segment_lengths` 等长）。
- **不提交**：按 `segment_lengths` 固定长度求解，请求与响应完全保持原样。
- **提交**：进入稳健模式，每个观测窗必须对各段长度在各自区间内的**任意
  独立取值**都满足伸长闭区间（不是只在名义长度处成立，也不做事后抽查）。

对给定应变 x，记窗内段集为 w。线性函数在整数长度盒上的极值必在顶点取得，
故窗的最小/最大可能回算和（精确整数）为：

    F_min(x) = Σ_{i∈w, x_i≥0} a_i·x_i + Σ_{i∈w, x_i<0} b_i·x_i
    F_max(x) = Σ_{i∈w, x_i≥0} b_i·x_i + Σ_{i∈w, x_i<0} a_i·x_i

稳健要求即 `F_min ≥ min_elongation` 且 `F_max ≤ max_elongation`。响应逐窗
返回这两个极值及取得极值时各段所用的长度端点见证
`min_extremum_lengths` / `max_extremum_lengths`（按窗内段序），可由请求
数据与返回应变直接复核；合法但不存在稳健解时返回 `409 INFEASIBLE`。
稳健求解按每段应变正负穷举至多 2^n 个符号分支（n≤12，完整覆盖长度盒），
分支内复用与固定长度相同的差分闭包缩域、整数界传播与 MRV/折半回溯。

### 三级优选（两种模式共用）

可行解依次最小化（三级字典序，前一级最优后才比较下一级）：

1. `max_adjacent_diff` = max_k |x_k − x_{k−1}|
2. `sum_adjacent_abs_diff` = Σ_k |x_k − x_{k−1}|
3. 应变序列 (x_0, …, x_{n−1}) 的字典序

实现方法（`app/solver.py`，仅标准库）：

- 引入前缀和 P_i = Σ_{j<i} L_j·x_j，观测窗与应变界全部化为 P 上的
  **差分约束**，以 Floyd–Warshall 最长路闭包检测正环（冲突）并导出
  每段 x_i 的紧整数域；
- |x_k−x_{k−1}|≤M 同样是差分约束，并入闭包缩域；
- 对线性窗约束做整数界传播 + MRV/折半回溯判定可行性；
- 最小 M、最小 S（辅助变量 e_k≥|Δ_k| 精确松弛）与字典序最优序列
  均利用「可行性关于阈值单调」二分钉死。

## 运行（Docker Compose）

```bash
# 宿主机端口可配置（默认 8000）
HOST_PORT=9000 docker compose up --build -d api
curl -s http://localhost:9000/health
```

### 一次性验证服务 verify

`verify` 服务会等待 `api` 健康检查通过，然后依次运行：

1. 单元测试（pytest，含固定长度与稳健长度区间两种模式对全枚举暴力解的
   三级最优性对照、稳健极值/见证复核、字段校验）
2. 构建检查（`compileall`）与 `pip check`
3. 一组反演 API 冒烟：固定长度兼容、稳健成功（极值与端点见证复核）、
   稳健不可行（`INFEASIBLE`）、非法长度区间（`INVALID_INPUT`）

完成后**自行退出**，退出码即结论（0 为全部通过）：

```bash
docker compose build
docker compose up --abort-on-container-exit --exit-code-from verify verify
echo "verify exit code: $?"
```

## API

### `GET /health`

返回 `{"status": "ok"}`，供容器与 Compose 健康检查使用。

### `GET /api/v1/schema`

返回请求/响应/错误码的字段约定。

### `POST /api/v1/invert`

请求体：

```json
{
  "segment_lengths": [10, 12, 11, 13, 10, 14],
  "length_intervals": [
    {"min": 9, "max": 11}, {"min": 11, "max": 13}, {"min": 10, "max": 12},
    {"min": 12, "max": 14}, {"min": 9, "max": 11}, {"min": 13, "max": 15}
  ],
  "strain_bounds": {"min": -100, "max": 100},
  "windows": [
    {"start_segment": 1, "end_segment": 6,
     "min_elongation": 100, "max_elongation": 400}
  ]
}
```

- `segment_lengths`：6–12 个正整数，按顺序排列的名义长度。
- `length_intervals`：**可选**；与 `segment_lengths` 等长的正整数闭区间，
  每项 `{"min": a, "max": b}` 且 `1 ≤ a ≤ b`。未提交/为 null 时按固定
  名义长度求解；提交后进入稳健模式。非法项定位到 `length_intervals[i].min/max`。
- `strain_bounds`：统一应变闭区间（整数微应变）。
- `windows`：8–20 个观测窗；段号 1 基且含端点，`start ≤ end`，
  `min_elongation ≤ max_elongation`。

成功（200，固定长度模式，结构与未启用区间时完全一致）：

```json
{
  "code": "OK",
  "result": {
    "segment_count": 6,
    "strains": [3, 5, 4, 2, 6, 1],
    "adjacent_diffs": [2, -1, -2, 4, -5],
    "objectives": {
      "max_adjacent_diff": 5,
      "sum_adjacent_abs_diff": 14
    },
    "window_checks": [
      {
        "index": 0,
        "start_segment": 1,
        "end_segment": 6,
        "total_length": 70,
        "min_elongation": 100,
        "max_elongation": 400,
        "weighted_strain_sum": 234,
        "satisfied": true
      }
    ],
    "criteria_order": ["max_adjacent_diff",
                       "sum_adjacent_abs_diff", "lexicographic"]
  }
}
```

每个 `weighted_strain_sum` 都能用提交的 `segment_lengths` 与返回的
`strains` 直接复核并确认落在提交闭区间内；两级平滑指标可用
`adjacent_diffs` 直接复核。

成功（200，稳健模式，提交了 `length_intervals`）：

```json
{
  "code": "OK",
  "result": {
    "mode": "robust_interval",
    "segment_count": 6,
    "length_intervals": [[9, 11], [11, 13], [10, 12], [12, 14], [9, 11], [13, 15]],
    "strains": [3, 3, 3, 3, 3, 3],
    "adjacent_diffs": [0, 0, 0, 0, 0],
    "objectives": {"max_adjacent_diff": 0, "sum_adjacent_abs_diff": 0},
    "window_checks": [
      {
        "index": 0,
        "start_segment": 1,
        "end_segment": 6,
        "total_length_interval": [64, 76],
        "min_elongation": 100,
        "max_elongation": 400,
        "robust_min_weighted_sum": 192,
        "robust_max_weighted_sum": 228,
        "min_extremum_lengths": [9, 11, 10, 12, 9, 13],
        "max_extremum_lengths": [11, 13, 12, 14, 11, 15],
        "robustly_satisfied": true
      }
    ],
    "criteria_order": ["max_adjacent_diff",
                       "sum_adjacent_abs_diff", "lexicographic"]
  }
}
```

稳健模式下每个窗给出：

- `robust_min_weighted_sum` / `robust_max_weighted_sum`：对各段长度在区间内
  任意独立取值时，该窗回算和的精确最小/最大值（按返回应变的正负取端点）；
- `min_extremum_lengths` / `max_extremum_lengths`：取得相应极值时，窗内各段
  （按 start→end 顺序）实际使用的长度端点见证，全部是提交的 `min` 或 `max`；
- `total_length_interval`：窗内总长度闭区间 `[Σmin, Σmax]`；
- 恒有 `min_elongation ≤ robust_min_weighted_sum` 且
  `robust_max_weighted_sum ≤ max_elongation`，可用请求数据与 `strains`
  逐窗直接复核，无需信任服务。

错误：

- `400`：请求体不是合法 JSON。
- `422 INVALID_INPUT`：字段错误，`fields[]` 逐条给出 `field` 与 `message`
  （非法长度区间定位到 `length_intervals[i].min/max`）。
- `409 INFEASIBLE`：输入合法但不存在满足全部闭区间的整数应变序列；稳健
  模式下指不存在对长度盒内任意独立取值都成立的序列。

## 本地开发（无 Docker）

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
API_BASE_URL=http://127.0.0.1:8000 .venv/bin/python scripts/smoke.py
```
