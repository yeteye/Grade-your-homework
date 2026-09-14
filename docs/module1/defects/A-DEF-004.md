# A-DEF-004 超大云端评分响应未触发本地回退

- 被测基线：`2da8824`
- 发现人：曹浩（M202677234）
- 关联用例：A-022
- 严重程度：中；优先级：中
- 修复前状态：已复现，待修复
- 证据：`reports/module1/a-defects-baseline.json`，`test_a022_giant_ai_response_keeps_local_score`

## 复现

将网络响应替换为本地构造的 JSON，响应中 `choices[0].message.content` 为 `{"score": 10**400}` 对应的 JSON 文本。请求 `POST /compare_texts`，使用作答 `A`、参考答案 `AB`，开启 AI、权重 70%。没有发生真实网络请求，也未使用真实密钥。

预期：云端分数无效，保留文本基线得分 66.67，返回 200、非空警告，并保存这条基线成绩。

实际：`ai_grader.parse_score` 调用 `float(value)` 抛出未捕获的 `OverflowError`，接口返回 500，回退未执行，也未保存记录。

## 影响与修复验证

异常云端响应破坏已定义的本地回退流程。修复后须确认极大整数、超范围值、NaN 和正常 0～1 分数的解析及回退行为；填写修复提交与复测证据后关闭。
