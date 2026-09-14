# Model assets

`transformer/model_best.pdparams` 是原项目训练得到的 Paddle 权重，`transformer/vocab.txt` 是对应的 BERT 中文词表。二者均从原位置迁移，内容未改。

模型代码位于 `homework/models/transformer.py`，仅在选择 Transformer 时加载。无需加载训练集、创建优化器或导入 HanLP/PaddleNLP/Matplotlib。

该权重可以运行，但不能直接当作可靠的作业评分模型。本次在 LCQMC 测试集前 20 条上的小规模检查：准确率 0.55，F1 0.40；这只是启动后的诊断样本，不是完整性能评估。对一组相同的短文本也出现低匹配概率。默认演示使用文本基线，Transformer 在页面中标注为实验引擎。

```powershell
.\.venv\Scripts\python.exe tools/evaluate_model.py --engine transformer --limit 100
```

权重文件约 97 MB；托管平台如有限制，应使用 Git LFS 或可靠下载地址，不要删除词表或将权重路径写死到其他人的电脑上。仅加载可信来源的模型文件。
