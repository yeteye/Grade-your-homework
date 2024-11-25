import hanlp
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 加载 HanLP 模型
text2vec = hanlp.load('LARGE_ALBERT_BASE')
segmenter = hanlp.load('PKU_NAME_MERGED_SIX_MONTHS_CONVSEG')

# 输入两个句子
sentence1 = '我喜欢学习自然语言处理'
sentence2 = '自然语言处理是一个有趣的领域'

# 使用分词器分词
seg1 = " ".join(segmenter(sentence1))  # 将分词结果转换为空格拼接的字符串
seg2 = " ".join(segmenter(sentence2))

print(f"分词结果1: {seg1}")
print(f"分词结果2: {seg2}")

# 获取句子向量
embedding1 = text2vec(seg1)
embedding2 = text2vec(seg2)

# 如果 text2vec 返回的是嵌套结构，提取真正的数值向量
if isinstance(embedding1, dict) and 'vector' in embedding1:
    embedding1 = embedding1['vector']
if isinstance(embedding2, dict) and 'vector' in embedding2:
    embedding2 = embedding2['vector']

# 将嵌入向量转换为 numpy 数组
embedding1 = np.array(embedding1, dtype=np.float32)
embedding2 = np.array(embedding2, dtype=np.float32)

# 检查向量是否为一维并长度相同
print(f'Embedding1 shape: {embedding1.shape}')
print(f'Embedding2 shape: {embedding2.shape}')

if embedding1.shape != embedding2.shape:
    raise ValueError("两个句子的向量长度不一致，无法计算相似度")

# 确保向量为二维
embedding1 = embedding1.reshape(1, -1)
embedding2 = embedding2.reshape(1, -1)

# 计算余弦相似度
similarity = cosine_similarity(embedding1, embedding2)
print(f'语义相似度: {similarity[0][0]}')
