# 阅知 · Homework Studio

本地运行的作业批改工作台。支持图片识别、文本与批量批改、评分要点、答案模板、历史检索、统计和报告导出。

## 快速启动

本次已在项目 `.venv` 中重建 Python 3.12 环境。当前电脑可直接双击 **`run.bat`**，打开 <http://127.0.0.1:5000>。关闭运行窗口或按 Ctrl+C 可停止服务。

也可在项目目录运行：

```powershell
.\.venv\Scripts\python.exe server.py
```

首次在其他电脑使用：安装 Python 3.12（64 位，含 Python Launcher），再依次运行 `setup.bat`、`setup-ai.bat`、`run.bat`。如只需要文本基线和记录管理，可以跳过 `setup-ai.bat`。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements/ocr.txt -r requirements/model.txt
.\.venv\Scripts\python.exe tools\check_environment.py
.\.venv\Scripts\python.exe server.py
```

已验证的完整版本记录在 `requirements/lock-windows-py312.txt`，可用 `python -m pip install -r requirements/lock-windows-py312.txt` 复现。网络缓慢时可在安装命令末尾添加 `--index-url https://pypi.tuna.tsinghua.edu.cn/simple`。

当前 `.venv` 的基础解释器来自本机 Codex 工作区运行时。虚拟环境不可复制到其他电脑；若基础解释器被升级或移除，使用独立安装的 Python 3.12 重建环境。原先绑定其他用户路径的失效虚拟环境已清理。

如果端口被占用，可复用已经运行的工作台，或指定另一端口：

```powershell
$env:PORT = '5001'
.\.venv\Scripts\python.exe server.py
```

## 演示流程

1. 点击“载入演示内容”，再点击“开始批改”。默认采用本地文本基线，不会调用付费 API。
2. 在右侧查看真实得分、关键词覆盖和文本差异；前往“批改记录”复核，导出 CSV / JSON 或打开可打印报告。打印对话框可选择“另存为 PDF”。
3. 在“答案模板”新建参考答案和要点；点击“使用模板”回填工作台。
4. 切换“批量批改”，每行输入 `姓名 | 作答`。每次最多 20 份，共用参考答案、要点和评分规则。全部成功才保存记录；任一输入失败时不写入部分记录。
5. 图片流程：切换“图片识别”，选择 RapidOCR，上传学生作业与参考答案，或点击“使用演示图片”，识别并校正文本，再批改。可选择原图、灰度、Otsu 二值化或对比度增强。安装后 RapidOCR 的识别模型随包提供，不需单独下载，也不需 GPU。`examples/` 提供演示图片。
6. 选择 Transformer 可调用仓库原有本地语义模型。查看“评分与环境”了解依赖和权重是否存在。首次推理可能比后续稍慢。

页面中由开发验收生成的记录和模板均使用“示例 / 演示”名称；这些是实际运行结果，不是预填统计。记录保存在本机，刷新或重启不会丢失。

## 模块与评分规则

| 模块 | 功能及边界 |
| --- | --- |
| 文本批改 | 姓名、作业名称、答案校验；单段最多 5000 字；文本差异、原文恢复 |
| 图片识别 | 原图 / 灰度 / Otsu 二值化 / 增强；每张最多 8 MB、2000 万像素；检查真实图片内容；独立临时目录；识别完成或失败均清理 |
| 批量批改 | 1～20 份，事务保存；本地引擎；不使用云端 AI |
| 评分配置 | 满分 1～1000；达标线 0～100%；AI 权重 0～100%；全局默认规则持久化 |
| 评分要点 | 最多 20 项，关键词不重复，权重 0.1～100；每条可显示是否命中 |
| 答案模板 | 新建、编辑、使用、删除；保存答案、要点和规则；不改变历史记录 |
| 批改记录 | 姓名 / 作业检索，达标状态过滤，分页，详情，单条删除 |
| 报告导出 | 当前筛选结果 CSV；单条 CSV / JSON；可打印 HTML；CSV 公式注入防护 |
| 学习概览 | 实际记录总数、平均得分率、达标比例、得分率分布；不同满分归一化 |
| 环境状态 | 按需检查 OCR、Paddle、权重、云端密钥是否配置；不会自动请求云端 |

**文本基线是透明、可重复的字符匹配算法，不等于语义理解。** 先做 NFKC 规范化、忽略大小写与非文字数字字符，再计算字符多重集合 Dice：`2 × 重合字符数量 / 两段字符总数`。文字顺序变化、否定句等可能产生高分，需人工复核。

Transformer 使用原项目的二分类匹配概率；它也不是经教学标定的正确率。合并作业和参考答案后最多 509 字（另含 3 个特殊标记），超限会明确拒绝，不会静默截掉学生内容。

若有评分要点，基础分为 `40% × 引擎相似度 + 60% × 加权关键词覆盖率`。要点使用规范化后的字面包含判断，不保证语义正确。若启用 DeepSeek 且权重大于 0，最终分按所选权重混合 AI 评分；AI 未配置、超时或返回无效分数时，保留完整基础分并显示原因，不将失败伪装为 0.5 分。

最终得分保留两位小数；达标判断使用已显示得分与对应两位小数的达标分数。历史记录保存当时的完整规则快照。每次批改生成独立记录，重复批改也会计入统计。

## 可选组件与密钥

### 本地模型

默认权重位置为 `models/transformer/model_best.pdparams`（原项目已有，约 97 MB）。只使用可信来源的 Paddle 权重文件。推理不再读取 LCQMC 训练数据，也不再要求 HanLP / PaddleNLP / Matplotlib。

可覆盖权重路径：

```powershell
$env:HOMEWORK_MODEL_PATH = 'E:\models\model_best.pdparams'
```

### DeepSeek

原源码曾包含硬编码密钥，已移除。**仓库历史中的旧密钥仍应由账号持有人撤销或轮换。** 不要把真实密钥提交到 Git 或发到聊天里。

在 PowerShell 中设置环境变量，再从同一终端启动：

```powershell
$env:DEEPSEEK_API_KEY = '替换为你自己的新密钥'
$env:DEEPSEEK_MODEL = 'deepseek-chat'
.\.venv\Scripts\python.exe server.py
```

`.env.example` 仅为配置说明，不会自动加载。实际发送内容为参考答案与学生作答，不含姓名；调用可能产生账号费用。适配器超时为 30 秒，不自动重试，不记录答案或密钥到应用日志。模型名可通过环境变量调整。实现参照 [DeepSeek Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/) 与 [JSON Output](https://api-docs.deepseek.com/guides/json_mode/)。

### OCR 引擎

在“图片识别”中选择引擎、识别语言，上传作业与答案图片，点击“识别两张图片”。可先用“使用演示图片”检查流程。识别完成后校正文字，再执行批改；首次加载 PaddleOCR 模型会稍慢。

| 引擎 | 本地配置 | 重新安装 |
| --- | --- | --- |
| RapidOCR | 1.4.4，模型随 Python 包提供 | `setup-ai.bat` |
| PaddleOCR | 3.2.0 / PaddleX 3.2.1 / PaddlePaddle 3.2.2；PP-OCRv5 mobile 检测和识别模型，中英文共用，CPU 推理 | `setup-paddleocr.bat` |
| Tesseract | `runtime/tesseract/tesseract.exe`；`tessdata/chi_sim.traineddata` 与 `eng.traineddata` | 安装引擎后运行 `tools/setup_tesseract_languages.py` |

PaddleOCR 模型在 `models/ocr/PP-OCRv5_mobile_det` 和 `models/ocr/PP-OCRv5_mobile_rec`，约 21 MB；安装脚本从 Paddle 官方地址下载并校验 SHA-256。推理使用显式本地目录，不在网页识别时下载模型。CPU 配置关闭文档旋转、去扭曲和文字方向辅助模型，上传前请将图片旋正。接口按 [PaddleOCR 官方使用文档](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html) 接入 `predict()`，固定使用上述已配套版本，不再使用旧版 `ocr(..., cls=True)`。

本机已验证三种引擎的中文、英文印刷体图片与后续批改流程。PaddleOCR 首次识别两张图片约 15 秒，后续示例约 4 秒，实际耗时随图片变化。PaddleX 3.2 首次导入会探测模型站点，断网时可能显示 `No model hoster is available`；已有本地模型仍可使用。`No ccache found` 不影响本项目的预编译 CPU 推理。无需因此重新下载模型或安装编译工具。

Tesseract 的 Windows 安装程序与真正的 `tesseract.exe` 不同，单独复制安装程序不能用于识别。本机已将提供的安装程序安装到 `runtime/tesseract`。程序会自动查找该目录、项目根目录、`Tesseract-OCR/`、系统 PATH 和常规安装目录；无需修改系统 PATH。中文语言数据来自 [Tesseract 官方 tessdata_fast](https://github.com/tesseract-ocr/tessdata_fast)。

在另一台电脑上，先安装 Tesseract 到 `runtime/tesseract`，再运行：

```powershell
.\.venv\Scripts\python.exe tools/setup_tesseract_languages.py
.\.venv\Scripts\python.exe tools/check_environment.py
```

也可以设置 `TESSERACT_CMD` 指向其他安装位置，`TESSDATA_PREFIX` 指向语言数据目录，`HOMEWORK_OCR_MODEL_DIR` 指向包含两份 PaddleOCR 模型的父目录。这些是启动前的环境变量；`.env.example` 不会自动加载。

命令行识别（不会创建批改记录）：

```powershell
.\.venv\Scripts\python.exe tools/recognize_image.py examples/demo-work.png --engine PaddleOCR
.\.venv\Scripts\python.exe tools/recognize_image.py examples/demo-work.png --engine Tesseract
# For English images: add --language en
```

环境页只检查依赖、模型和语言文件是否存在，不能替代实际识别或精度评估。三种引擎均在本地运行。模型目录与引擎安装目录不提交到 Git，换电脑需复制相应资源或重新运行安装步骤。

## 项目结构

```text
server.py                     唯一 Web 服务入口
launch.py                     等待服务就绪后打开浏览器
homework/                     业务模块：校验、评分、OCR、预处理、存储、AI 适配
homework/models/transformer.py 原模型结构与按需推理
models/transformer/            模型权重和词表
models/ocr/                    PaddleOCR 本地推理模型（不提交）
runtime/tesseract/             Tesseract 引擎和语言包（不提交）
static/                       页面 CSS / JavaScript
templates/                   工作台与可打印报告 HTML
tools/                       环境检查、预处理、比较、模型评估、可选超分辨率
checks/                      开发回归检查
examples/                    演示图片与去重后的历史样例
datasets/lcqmc/              原有训练、验证、测试 TSV 数据
research/                    研究实验、笔记、历史结果及第三方许可证
requirements/                OCR / 模型依赖、版本约束和已验证环境锁定文件
docs/                        结构迁移说明
instance/                    本机数据库、临时文件与工具输出（不提交）
```

详细迁移与删除记录见 [结构整理说明](docs/structure_changes.md)，历史实验及可选依赖见 [Research archive](research/README.md)。旧的第二服务入口和重复脚本已合并，运行主流程不再导入训练工具或实验脚本。

## Reusable tools

```powershell
.\.venv\Scripts\python.exe tools/preprocess_image.py examples/demo-work.png instance/binary_demo.png --mode binary
.\.venv\Scripts\python.exe tools/compare_texts.py "软件测试发现缺陷" "软件测试发现问题" --engine lexical
.\.venv\Scripts\python.exe tools/evaluate_model.py --engine transformer --limit 100 --output instance/evaluation.json
```

以上工具不创建批改记录。预处理和超分辨率工具保护已有输出文件，避免覆盖原图。模型评估针对二分类文本匹配，不能直接解释为教学评分准确率。实际对原有模型在测试集前 20 条进行诊断，准确率为 0.55、F1 为 0.40，仅为小样本运行检查；页面因此将其标为实验引擎。

## 开发验证

本次实际检查结果和未验证范围见 [开发验证记录](docs/development_verification.md)。

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s checks -v
.\.venv\Scripts\python.exe tools\check_environment.py
```

`checks/` 验证本次改造的关键行为与安全回归，在独立临时数据库中运行。外部 OCR / AI 异常使用替身模拟；这些通过不代表真实模型精度或云端连接已验证，也不替代课程用例清单、缺陷报告或测试报告。

## 课程模块一测试

本阶段的被测基线为 `2da8824`。成员 A 的正式用例与四个缺陷闭环说明见 [模块一测试协作](docs/module1/README.md)。安装基础依赖后，在项目根目录一键运行：

```powershell
.\run-module1.bat --suite all --label my-run-1
```

每次使用新的 `--label`，运行输出保存在 `reports/module1/`。`checks/` 的旧开发回归测试与 `tests/module1/` 的课程正式用例分别计数；完整小组交付需合并队友的用例、报告与演示材料。

本版本默认只监听 `127.0.0.1`，供本机演示和使用，没有多用户身份认证。数据库路径可通过 `HOMEWORK_DATA_DIR` 指定；备份时先停止服务，再复制整个 `instance/` 目录。CSV 导出不是完整数据库备份。上传图片识别后不保留，识别文字与批改记录会保留到主动删除。

## HTTP 接口

| 方法与路径 | 用途 |
| --- | --- |
| `POST /ocr` | multipart：`file1`、`file2`、`model`、`language`、可选 `preprocessing` |
| `POST /compare_texts` | 单份批改并保存 |
| `POST /api/batch` | 多份作答，共用答案和规则，事务保存 |
| `GET /api/records?q=&status=all&page=1&pageSize=10` | 记录检索；状态 `all/passed/review` |
| `GET/DELETE /api/records/<id>` | 详情 / 删除 |
| `GET /api/export` | 按记录筛选条件导出 CSV |
| `GET /api/records/<id>/export?format=json` | 单份 JSON / CSV |
| `GET /records/<id>/print` | 可打印报告 |
| `GET/POST /api/templates` | 模板列表 / 创建 |
| `PUT/DELETE /api/templates/<id>` | 模板更新 / 删除 |
| `GET/PUT /api/settings` | 默认评分规则 |
| `GET /api/stats` | 记录统计 |
| `GET /api/health` | 可选依赖状态 |

单份请求示例：

```json
{"studentName":"演示同学","title":"测试基础","workContent":"软件测试发现缺陷","answerContent":"软件测试发现缺陷","engine":"lexical","maxScore":100,"passPercent":60,"useDeepseek":false,"aiWeight":70,"rubric":[]}
```

批量请求将作答放入 `items: [{"studentName":"甲", "workContent":"答案"}]`，其余字段与单份请求一致；显式设置 `useDeepseek:false`。模板请求包含 `title`、`answerContent`、`rubric`、`options`。
