import sys
import hanlp

# 加载hanlp模型
similarity_model = hanlp.load('STS_ELECTRA_BASE_ZH')

# 获取输入文本
text1 = sys.argv[1]
text2 = sys.argv[2]

# 计算文本相似度
similarity_score = similarity_model((text1, text2))

# 输出相似度（返回值是一个浮动的值，直接输出它）
print(similarity_score)
