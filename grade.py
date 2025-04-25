from openai import OpenAI
import re
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)

# 初始化OpenAI客户端
client = OpenAI(api_key="sk-979c647fff274ae7a6c5e0394ccc559a", base_url="https://api.deepseek.com")

def get_points(reference, query):
    """
    使用deepseek API评估学生答案与参考答案的相似度
    
    参数:
        reference (str): 参考答案文本
        query (str): 学生答案文本
    
    返回:
        float: 0.00-1.00之间的分数，表示相似度
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": """你是一个专业评分老师，按以下规则精确打分：
    1. 完全符合参考答案语义给1.00分
    2. 部分正确按比例评分（如覆盖3个要点中的2个给0.67）
    3. 语义相同但表达不同仍给满分
    4. 必须输出0.00-1.00之间的两位小数

    示例：
    参考：水的沸点是100℃
    答案：水烧开需要100度 → 1.00
    答案：开水温度约一百度 → 0.95
    答案：水会沸腾 → 0.30"""
                },
                {
                    "role": "user",
                    "content": f"""
    [参考答案]
    {reference}

    [学生答案]
    {query}

    请严格输出0.00-1.00的数字评分："""
                }
            ],
            temperature=0.1,
            stream=False
        )
        
        # 结果处理
        score_str = response.choices[0].message.content
        logging.info(f"Deepseek原始输出: {score_str}")
        
        try:
            return float(score_str)
        except:
            # 使用正则提取数字
            match = re.search(r"\d?\.\d{1,2}", score_str)
            if match:
                score = float(match.group())
                logging.info(f"从文本中提取的评分: {score}")
                return score
            else:
                logging.warning("无法从Deepseek响应中提取评分")
                return 0.5  # 默认返回中等分数
                
    except Exception as e:
        logging.error(f"Deepseek API调用失败: {str(e)}")
        return 0.5  # 出错时返回中等分数