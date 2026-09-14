# Structure and migration notes

## Runtime boundaries

运行入口统一为 `server.py`，浏览器启动包装为 `launch.py`。Web 模块集中于 `homework/`；离线工具位于 `tools/`，它们复用同一份图像预处理、文本基线和模型推理代码。

## Renamed and consolidated

| Previous path | Current path / replacement |
| --- | --- |
| `grade.py` | `homework/ai_grader.py` |
| `reference.py` | 模型架构 / 推理迁入 `homework/models/transformer.py`；训练数据类迁入 `research/text_matching/datasets.py` |
| `nndl.py` | `research/text_matching/training.py` |
| `utilss/data.py` | `research/text_matching/datasets.py` |
| `utilss/bert-base-chinese-vocab.txt` | `models/transformer/vocab.txt` |
| `checkpoint/model_best.pdparams` | `models/transformer/model_best.pdparams` |
| `lcqmc/*.csv` | `datasets/lcqmc/*.tsv`（原文件实际是 TSV） |
| `transferIntoBlack.py` | `homework/preprocessing.py` 与 `tools/preprocess_image.py`，并接入 OCR 页面 |
| `test.py` | `research/image_processing/text_extraction_experiment.py`，显式 CLI 参数与输出保护 |
| `reference_func.py`、`txt_compare/text_comparer.py` | `tools/compare_texts.py` |
| `txt_compare/texts_comparer.py` | `tools/evaluate_model.py`，修正 FPR / FNR 分母，并输出 precision / recall / F1 |
| `ESRGAN/enhance_func.py`、`ESRGAN/test.py` | `tools/super_resolve.py`，复用保留的 RRDB 架构 |
| `ESRGAN/RRDBNet_arch.py` | `research/super_resolution/esrgan/rrdb_architecture.py` |
| `ESRGAN/transer_RRDB_models.py` | `research/super_resolution/esrgan/convert_rrdb_weights.py` |
| `ESRGAN/net_interp.py` | `research/super_resolution/esrgan/interpolate_weights.py` |
| `enhance/*.ipynb` | `research/notebooks/` |
| `templates/test.html` | `research/ui/ocr_prototype.html` |
| `others/model_sorted.md` | `research/notes/hanlp_models.md` |
| `output/HANLP/`、`output/OCR/` | `research/results/` |
| 根目录中文图片、`image/`、`uploads/` 唯一图片 | `examples/legacy/`，英文命名；重复内容经过哈希核对去重 |

## Removed

- `env/` 中三个绑定旧用户 Python 路径的失效虚拟环境；新 `.venv/` 保留。
- 根目录旧 `pyvenv.cfg`、空 `Include/`、Python 字节码缓存和已被追踪的旧 `.pyc`。
- `paddle-transformer/` 重复代码副本；其中一份推理示例会覆盖实际输入，已由参数化工具替代。
- 根目录 `debug_*`、`output*.jpg`、`enhanced_extracted_text.jpg`、`homework_bw2.jpg` 等临时处理产物。
- 空 `test.docx`、空 `test114514.txt`、无用途的 `testgit.txt`、原始模型列表控制台转储。
- 重复的 `server2.py` 服务入口，以及已经由共享模块替代的脚本。
- 迁移后为空或仅剩重复内容的旧目录。

唯一权重、训练数据、原始图片、实验笔记、历史评估结果和第三方许可证保留。`paddletransformer.zip` 在此次工作开始前已经是删除状态，本次未恢复或修改。旧的 `transferIntoBlack.py` 本地改动所实现的二值化流程已经融入共享模块；原本未追踪的 `test.py`、页面原型和唯一上传样例均已迁移保留。

新开发文件采用英文名称和 Python `snake_case`。第三方的说明、论文图片名及网络结构内的参数名保留兼容性，避免破坏权重匹配或文档引用。
