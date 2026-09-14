# Research archive

本目录保存有复用价值的研究代码和历史结果，不参与 Web 服务启动。

| Directory | Purpose |
| --- | --- |
| `text_matching/` | 原训练循环、数据集加载、批处理填充；模型架构已迁入 `homework/models/transformer.py` |
| `image_processing/` | 文字像素提取实验；已改为显式输入输出参数，不再硬编码用户路径或自动弹窗 |
| `super_resolution/esrgan/` | ESRGAN RRDB 架构、权重转换与插值、上游说明和许可证、历史图像结果 |
| `notebooks/` | ESRGAN / SRCNN 研究笔记；历史路径可能需要按当前结构调整 |
| `notes/` | HanLP 模型分类笔记 |
| `results/` | 保留的历史 OCR / HanLP 结果，不计入工作台统计 |
| `ui/` | 原简易 OCR 页面原型，仅供参考；不作为新服务页面 |

## Reused tools

```powershell
# 文字像素提取：connected 方法无需 Tesseract
.\.venv\Scripts\python.exe research/image_processing/text_extraction_experiment.py examples/demo-work.png instance/extracted_text.png --method connected

# ESRGAN：需要额外安装 PyTorch 并提供可信的模型权重
.\.venv\Scripts\python.exe tools/super_resolve.py examples/demo-work.png instance/upscaled.png --weights E:/models/RRDB_ESRGAN_x4.pth --device cpu

# 权重转换 / 插值：输出文件必须不存在
.\.venv\Scripts\python.exe research/super_resolution/esrgan/convert_rrdb_weights.py old.pth converted.pth
.\.venv\Scripts\python.exe research/super_resolution/esrgan/interpolate_weights.py psnr.pth esrgan.pth blended.pth --alpha 0.5
```

主流程已经复用了旧二值化思路和文本评估逻辑。ESRGAN 没有现成权重，且需要额外的 PyTorch 环境，因此保留为可选工具，没有伪装成已可用的网页功能。原上游 README 中的历史运行命令以本目录的新命名和上面的入口为准；上游许可证完整保留。
