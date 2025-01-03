# 试卷题目打分器

## 环境配置

推荐使用python版本3.9,由于hanlp和flask、paddleOCR之间存在版本冲突，需要用不同环境隔离

### 建立hanlp环境并安装hanlp
```
python3.9 -m venv env\MyHanlp

source env/MyHanlp/bin/activate   # Windows: env\MyHanlp\Scripts\activate

# 安装hanlp
pip install hanlp[full]
```
### 建立图像识别环境
```
python3.9 -m venv env\TxtCapturer

source env/TxtCapturer/bin/activate   # Windows: env\TxtCapturer\Scripts\activate

# 安装paddlepaddle
# cpu运行版本
python -m pip install paddlepaddle==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# gpu运行版本
python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# 安装PaddleOCR
pip install "paddleocr>=2.0.1"      # 推荐使用2.0.1+版本

```
## 训练数据来源

### DIV2K图片数据集
https://data.vision.ee.ethz.ch/cvl/DIV2K/


## 使用说明

### 运行run.bat

运行前注意修改必要路径（server.py）
```
python_path = './env/MyHanlp/Scripts/python3.9.exe' 
pytesseract_path = r'D:/Program Files/Tesseract-OCR/tesseract.exe'
txt_compare_path = './txt_compare/text_comparer.py'
```