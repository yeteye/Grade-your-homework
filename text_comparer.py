import hanlp

# 加载可用的中文语义相似度模型
similarity_model = hanlp.load('STS_ELECTRA_BASE_ZH')

# 示例文本
text1 = "I hate programming because it's challenging.？"
text2 = "Programming is fun and challenging, which is why I love it."
text3 = "今天天气很好，我想出去玩。"

# 计算相似度
similarity_1_2 = similarity_model((text1, text2))
# similarity_1_3 = similarity_model((text1, text3))

# 输出结果
print(f"'{text1}' 和 '{text2}' 的相似度: {similarity_1_2:.4f}")
# print(f"'{text1}' 和 '{text3}' 的相似度: {similarity_1_3:.4f}")