# 海缆连续缆段整数微应变联合反演服务

海缆检修组只能取得多个**重叠标距**的累计伸长量。本服务把 6–12 段缆段的
整数微应变作为未知量，对 8–20 个由「连续起止段 + 累计伸长量闭区间」组成的
观测窗做**精确整数联合反演**，避免逐窗均摊导致彼此矛盾的局部结论。

## 数学模型与优选准则

设第 i 段长度为 L_i、待求整数微应变为 x_i。每个观测窗 [s,e] 要求

    Σ_{i=s..e} L_i · x_i ∈ [min_elongation, max_elongation]

且每段都满足统一应变闭区间 `strain_bounds.min ≤ x_i ≤ strain_bounds.max`。
所有运算与比较均为 Python 任意精度**整数**，无任何浮点参与。

### 可选：标距长度闭区间（稳健模式）

现场复测发现标距长度本身存在整数测量误差时，可为每段额外提交
`segment_length_intervals[i] = {"min": lo_i, "max": hi_i}`（正整数闭区间，
须包含名义长度 L_i）。此时每个观测窗必须对**各段长度在各自区间内的任意
独立整数取值**都满足累计伸长闭区间——不是只按名义长度求解，也不是抽查
若干组合。

长度加权和关于各 L_i 线性，极值必在区间端点取得：x_i>0 时最小和取 lo_i、
最大和取 hi_i；x_i<0 时反之。因此稳健条件等价于

    Σ_i f⁻ᵢ(x_i) ≥ min_elongation  且  Σ_i f⁺ᵢ(x_i) ≤ max_elongation
    f⁻ᵢ(x) = lo_i·x (x≥0) / hi_i·x (x<0)（关于 x 单调递增）
    f⁺ᵢ(x) = hi_i·x (x≥0) / lo_i·x (x<0)（关于 x 单调递增）

服务以精确整数按应变正负推导每个窗口的最小/最大可能回算和，并在响应中
返回取得两端极值时各段所用的长度端点（见证），可由请求数据直接复核。
未提交 `segment_length_intervals` 时，请求、三级裁决与响应与固定长度
原版完全一致。

可行解依次最小化（三级字典序，前一级最优后才比较下一级）：

1. `max_adjacent_diff` = max_k |x_k − x_{k−1}|
2. `sum_adjacent_abs_diff` = Σ_k |x_k − x_{k−1}|
3. 应变序列 (x_0, …, x_{n−1}) 的字典序

实现方法（`app/solver.py`，仅标准库）：

- 引入前缀和 P_i = Σ_{j<i} L_j·x_j，观测窗与应变界全部化为 P 上的
  **差分约束**，以 Floyd–Warshall 最长路闭包检测正环（冲突）并导出
  每段 x_i 的紧整数域；
- |x_k−x_{k−1}|≤M 同样是差分约束，并入闭包缩域；
- 稳健模式下，每窗的 Σ f⁻ᵢ(x_i) ≥ 下端、Σ f⁺ᵢ(x_i) ≤ 上端是单调分段
  线性约束，按段做**精确整数界传播**（区间一致性），与线性窗约束一起
  嵌入同一传播/回溯骨架；名义长度前缀闭包仍是稳健可行集的安全超集
  （区间必含名义值），缩域逻辑原样复用；
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

1. 单元测试（pytest，含与全枚举暴力解的三级最优性对照，固定长度与
   稳健模式均覆盖）
2. 构建检查（`compileall`）与 `pip check`
3. 一组反演 API 冒烟：固定长度兼容（成功回算、`INFEASIBLE`、
   `INVALID_INPUT`）、稳健模式成功（窗口极值与端点见证复核）、
   稳健模式不可行（名义可行但无法稳健满足 -> 409）、非法长度区间
   （422 并定位缆段字段）

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
  "strain_bounds": {"min": -100, "max": 100},
  "windows": [
    {"start_segment": 1, "end_segment": 6,
     "min_elongation": 100, "max_elongation": 400}
  ]
}
```

- `segment_lengths`：6–12 个正整数，按顺序排列。
- `segment_length_intervals`（可选）：与 `segment_lengths` 等长，每项
  `{"min": 正整数, "max": 正整数}` 且 `min ≤ max`、包含对应名义长度。
  提交后启用稳健模式；未提交时行为与固定长度原版一致。
- `strain_bounds`：统一应变闭区间（整数微应变）。
- `windows`：8–20 个观测窗；段号 1 基且含端点，`start ≤ end`，
  `min_elongation ≤ max_elongation`。

成功（200）：

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

稳健模式下，`window_checks` 每项额外返回（且响应含
`"length_mode": "interval"` 与回显的 `segment_length_intervals`）：

```json
{
  "min_possible_weighted_sum": 218,
  "max_possible_weighted_sum": 250,
  "lengths_at_min": [9, 11, 10, 12, 9, 13],
  "lengths_at_max": [11, 13, 12, 14, 11, 15],
  "satisfied": true
}
```

- `min/max_possible_weighted_sum`：按应变正负取长度端点推导的该窗
  最小/最大可能回算和（精确整数）；`satisfied` 要求
  `min_elongation ≤ min_possible 且 max_possible ≤ max_elongation`，
  即任意长度取值都满足。
- `lengths_at_min` / `lengths_at_max`：取得两端极值时窗内各段所用的
  长度端点（与窗内段一一对应，均取自提交区间的端点），其加权和即
  对应极值，可由请求数据直接复核。

错误：

- `400`：请求体不是合法 JSON。
- `422 INVALID_INPUT`：字段错误，`fields[]` 逐条给出 `field` 与
  `message`；长度区间非法时定位到 `segment_length_intervals[i]` 的
  具体字段。
- `409 INFEASIBLE`：输入合法但观测窗彼此冲突（稳健模式下：无法对所有
  长度取值同时满足），不存在满足全部闭区间的整数应变序列。

## 本地开发（无 Docker）

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
API_BASE_URL=http://127.0.0.1:8000 .venv/bin/python scripts/smoke.py
```
