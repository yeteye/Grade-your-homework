# A-DEF-001 超大评分数值导致接口返回 500

- 被测基线：`2da8824`
- 发现人：曹浩（M202677234）
- 关联用例：A-019
- 严重程度：中；优先级：中
- 复现环境：Windows x64、Python 3.12.14、Flask 3.1.3；隔离临时 SQLite 数据库
- 状态：已修复并回归通过
- 证据：`reports/module1/a-defects-baseline.json`，`test_a019_giant_numeric_input_is_validation_error`

## 复现

向 `POST /compare_texts` 发送正常作业和参考答案，将 `maxScore` 设置为 JSON 整数 `10**400`。脚本在 Python 中构造该整数并通过 Flask test client 发送，不依赖外部服务。发送前后均查询 `/api/stats`。

预期：满分超出 1～1000，应返回 400 和参数错误说明，不保存记录。

实际：返回 500，服务端 `validation.number` 的 `math.isfinite(value)` 在将超大整数转换为浮点数时抛出 `OverflowError`。数据库没有新增记录。

## 影响与修复验证

不可信输入造成内部错误响应，掩盖了可解释的参数校验结果。修复提交 `358a085` 先按有效区间校验，再检查有限性，避免超大整数触发隐式浮点转换。`reports/module1/a-final-full.json` 中 A-019 已通过，返回 400 且未保存记录。原有 A-004、A-006 边界和类型用例也通过。
