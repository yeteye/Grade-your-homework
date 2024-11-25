# 试卷题目打分器

## 环境配置

推荐使用python版本3.9

### 建立hanlp环境并安装hanlp
```
python3.9 -m venv MyHanlp
source MyHanlp/bin/activate   # Windows: MyHanlp\Scripts\activate
pip install hanlp[full]
```
### 建立图像识别环境
```
python3.9 -m venv TxtCapturer
source TxtCapturer/bin/activate   # Windows: TxtCapturer\Scripts\activate
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