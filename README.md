# 流式 CTC 解码工作台

面向语音算法工程师的本地网页工具：上传词表与声学模型逐帧概率，按束宽做 CTC
前缀束搜索，分块推进，随时确认 / 撤销字幕前缀，并查看每条候选的对数概率；
另提供**双片段对读**，用于核对两段录音是否为重复采集。

- 后端：FastAPI + NumPy（CTC 前缀束搜索与对读算法均为纯 Python/NumPy 实现，无需下载模型）
- 前端：单个静态页面（原生 JS，无构建步骤）
- 部署：`docker compose up` 即可访问 <http://localhost:8000>

## 快速开始

### Docker Compose（推荐）

```bash
docker compose up --build
# 浏览器打开 http://localhost:8000
```

### 本机运行

```bash
python3 -m pip install -r requirements.txt
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

需要 Python 3.10+。页面会列出 `samples/` 下的内置样例，点击即可载入，无需任何模型文件。

## 使用流程

1. **加载数据**：上传两个 JSON 文件，或直接点击内置样例。
   - 词表：`["<blank>", "啊", "阿", "波", "播"]`（第 0 项为 blank，最多再加 8 个单字）
   - 帧概率：`{"frames": [[0.9, 0.06, 0.04], ...]}`，最多 2000 帧，每帧长度等于词表，
     取值 `0～1`（`0` 表示 `log 0 = -∞`）
2. **设置束宽**（1～16）与分块大小，点击「推进分块」或「连续推进」。
3. **查看候选**：表格按分数排序列出每个候选的 `log P`、概率、末字的 blank / 非 blank
   路径概率。把「查看帧」滑到当前帧之前的任意帧，可以回放当时已推进到的候选（只读）。
4. **确认前缀**：点「确认当前候选」，或在输入框中输入更短的前缀后确认。确认后只保留
   兼容候选，后续推进不会再把不兼容路径带回束中。
5. **撤销确认**：恢复「未做该次确认时」在当前帧应有的束与分数（不是简单解除文字限制），
   已推进的帧不会回退。
6. **双片段对读**：在第 5 面板从**已推进帧**中选择连续 8～64 帧作为 A 段，上传同词表的
   另一段 8～64 帧概率作为 B 段（或点击内置对读样例），核对两段是否为重复采集。

## 双片段对读的判定语义

对读只采信**发音事件**，blank 帧永远不会成为「重复」的证据。完整流水线见
`app/compare.py`：

1. **逐帧归一化**：每帧除以自身之和。出现全零帧时无法归一化，接口返回 400 并提示
   「已保留上次对读结果」，前端不改动已有结论。
2. **事件抽取**：归一化后某个非 blank 类概率 **≥ 0.8** 的帧才记为发音帧；连续同字帧合并为
   一个事件，只有中间出现 **blank 占优帧**（argmax 为 blank）时，同字才算两次发音；
   没有任何类别过阈值的帧只算「模糊帧」，不会切分事件。
3. **单调对齐**：在帧网格上做只允许 `(1,1)/(1,0)/(0,1)` 的 DTW，且禁止两个连续的同轴
   移动（局部 Sakoe-Chiba 斜率 2 约束），任一侧局部速度都不超过 2 倍。事件帧与不同字
   对齐代价最高，因此高置信事件无法靠 blank 路径被悄悄跳过。
4. **事件对应与冲突**：把对齐路径投影到事件上，要求一一且同字对应。事件被跳过、一对多、
   异字对应或事件帧跨度比超过 2 倍都列为冲突；长度比超过 2 倍、不存在合法对齐路径时直接
   判为冲突。
5. **相似度（0～1）**：仅由事件证据组成——字标一致占 0.6，时序占 0.3（局部速度与事件中心
   在归一化时间轴上的位置各半），事件平均置信度占 0.1。任一侧没有发音事件时返回
   「证据不足」且相似度为 `null`，大量 blank 不会抬高相似度。

### 过期与迟到结果

- 每次对读请求带 `request_id`，前端只接受最新一次请求的响应，迟到响应直接丢弃，不会覆盖
  新结果。
- 结果记录计算时的会话 `revision` 与 A 段帧范围；推进帧、切换会话或改动区间后，旧结果
  保留在页面上并显示「结果已过期」横幅，标明原帧范围。
- 对读失败（全零帧、坏 JSON、区间越界等）只显示错误，旧结果原样保留；对读接口不修改
  解码会话。

## 关键语义与实现

### CTC 规则

- 空白记作第 0 类；同一文本的所有路径在**剪枝前**按 `log-sum-exp` 合并，
  因此同一字符串不会占用两个束位。
- 相同学分按文本的 Unicode 码点升序排列（`sorted(key=(-logp, text))`）。
- 连续重复字符只有在中间有 blank 时才会展开成两个字符；无 blank 的连续同字
  坍缩为一个字符。
- 每束保留 `logp_blank` / `logp_nonblank`，即末字以 blank 结束或非 blank 结束的
  路径概率质量。

### 任意分块一致性

`CTCBeamDecoder.advance` 只是对 `step` 的循环，确认限制也在 `step` 剪枝前应用，
所以「一次推进 N 帧」与「逐帧推进 N 次」得到完全相同的束与分数。测试
`test_chunked_equals_per_frame` 用随机 23 帧数据逐位比较。

### 确认与撤销

- 确认时记录当前束的副本（只含更早的确认限制）、确认帧号、确认文本及其精确 CTC
  前向概率（blank / 非 blank 拆分）。分数从不重新归一化。
- 撤销时从记录的副本出发，用记录的前缀限制重放确认之后已消费的帧，得到「若从未做过
  这次确认」在当前帧应有的束。这是撤销正确性的关键：仅移除文字过滤会留下被限制后
  剪枝丢失的候选。
- 撤销后若再次确认，会生成新的记录，历史确认列表也随之更新。

### 并发与迟到结果

- 每个会话有一个单线程 executor，所有推进 / 确认 / 撤销按提交顺序串行执行，
  读取（`GET /api/sessions/{id}`）也在会话锁内取快照，不会看到半更新状态。
  因此重算（撤销后的重放）期间仍可点击推进或再次撤销：这些请求排队在重算之后，
  作用于重算结果，重算的迟到响应不会覆盖它们。
- 每次状态变更使 `revision` 递增；响应携带该值。前端丢弃 `revision` 更小的响应
  （以及不属于当前会话的响应），并在切换会话时递增 `generation`。
- 操作失败（非法前缀、帧越界、会话过期等）返回 4xx，且**不修改**会话；
  前端保留原字幕与进度，仅显示错误信息。

## API 摘要

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 健康检查 |
| POST | `/api/sessions` | 用 JSON 创建会话（`vocab`、`beam_width`、`frames`） |
| POST | `/api/sessions/upload` | 上传词表与帧文件（multipart） |
| GET | `/api/sessions/{id}` | 读取当前状态 |
| DELETE | `/api/sessions/{id}` | 删除会话 |
| POST | `/api/advance` | `{session_id, to_frame}` 推进到帧边界 |
| POST | `/api/peek` | `{session_id, frame}` 回放已推进帧的候选 |
| POST | `/api/confirm` | `{session_id, prefix}` 延长已确认前缀 |
| POST | `/api/undo` | 撤销最近一次确认 |
| POST | `/api/compare` | 双片段对读：`{session_id, start_frame, end_frame, frames_b, request_id?}` |
| POST | `/api/compare/upload` | multipart 上传 B 段文件的对读入口 |
| GET | `/api/samples`、`/api/samples/{name}` | 内置样例 |
| GET | `/api/compare-samples`、`/api/compare-samples/{name}` | 内置对读样例（`samples/pairs`） |

状态中的 `beams[].logp_blank` / `logp_nonblank` 为 `null` 表示 `-∞`（JSON 无 `-Infinity`）。
`prefix_scores` 是确认时记录的末字 blank / 非 blank 路径概率（以及完整前缀分数
`logp` 与该前缀之前的分数 `previous_logp`），之后不再重新归一化或重算。

## 测试

```bash
python3 -m pytest -q tests            # 单元 + HTTP + 并发 + 对读算法/接口测试（75 项）
python3 scripts/make_samples.py       # 重新生成内置解码样例
python3 scripts/make_pair_samples.py  # 重新生成内置对读样例
```

可选的浏览器级冒烟测试（需要 jsdom，在服务已启动时执行）：

```bash
mkdir -p /tmp/uitest && npm --prefix /tmp/uitest install jsdom
JSDOM_DIR=/tmp/uitest node tests/ui_smoke.mjs http://127.0.0.1:8000
```

该脚本使用 jsdom 驱动真实页面，覆盖：样例载入、分块推进、确认前缀、撤销后进度保留、
非法确认的错误提示、历史帧只读回放、连续推进到结尾、零概率样例不出现 NaN，以及双片段
对读的高相似度结论、冲突事件、对齐路径展开、区间变更后的过期标记、全零帧失败保留旧结果。

## 内置样例

| 文件 | 覆盖点 |
| --- | --- |
| `samples/sample_repeat.json` | 连续同字、blank 分隔后的重复字符 |
| `samples/sample_blank_heavy.json` | 大量 blank 帧中间夹短音 |
| `samples/sample_sparse_zero.json` | 含 `0` 概率（`-∞`），界面不出现 NaN |
| `samples/pairs/pair_duplicate.json` | 对读：事件一一对应、节奏略变，应判高相似度无冲突 |
| `samples/pairs/pair_conflict.json` | 对读：同位置异字与事件跳过，必须报冲突而非判重复 |

## 已知限制

- 会话保存在单进程内存中，重启即丢失；多副本部署需要外部存储。
- 束搜索为纯 Python 循环，2000 帧 × 束宽 16 约需 1.5 秒，期间通过单线程队列串行处理；
  未做多进程并行。
- 页面只接受 UTF-8 编码的 JSON/文本文件，单文件大小受反向代理默认限制。
- 对读固定在 8～64 帧的小片段上，DTW 为 `O(n·m)` 的纯 Python 网格；超出该长度的片段
  需要先切分再逐段核对。
