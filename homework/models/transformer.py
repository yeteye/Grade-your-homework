"""Original local Transformer architecture, with lazy CPU-only inference loading."""
from functools import lru_cache
from pathlib import Path
import os
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]


class WordEmbedding(nn.Layer):
    def __init__(self, vocab_size, emb_size, padding_idx=0):
        super(WordEmbedding, self).__init__()
        # Embedding的维度
        self.emb_size = emb_size
        # 使用随机正态（高斯）分布初始化 embedding
        self.word_embedding = nn.Embedding(vocab_size, emb_size,
            padding_idx=padding_idx, weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Normal(0.0, emb_size ** -0.5) ), )

    def forward(self, word):
        word_emb = self.emb_size ** 0.5 * self.word_embedding(word)
        return word_emb

class SegmentEmbedding(nn.Layer):
    def __init__(self, vocab_size, emb_size):
        super(SegmentEmbedding, self).__init__()
        # Embedding的维度
        self.emb_size = emb_size
        # 分段编码
        self.seg_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=emb_size
        )

    def forward(self, word):
        seg_embedding = self.seg_embedding(word)
        return seg_embedding

def get_sinusoid_encoding(position_size, hidden_size):
    """位置编码 """

    def cal_angle(pos, hidden_idx):
        # 公式里的 i = hid_idx // 2
        return pos / np.power(10000, 2 * (hidden_idx // 2) / hidden_size)

    def get_posi_angle_vec(pos):
        return [cal_angle(pos, hidden_j) for hidden_j in range(hidden_size)]

    sinusoid = np.array([get_posi_angle_vec(pos_i) for pos_i in range(position_size)])
    # dim 2i  偶数正弦
    # 从0开始，每隔2间隔求正弦值
    sinusoid[:, 0::2] = np.sin(sinusoid[:, 0::2])
    # dim 2i 1  奇数余弦
    # 从1开始，每隔2间隔取余弦
    sinusoid[:, 1::2] = np.cos(sinusoid[:, 1::2])
    # position_size × hidden_size  得到每一个词的位置向量
    return sinusoid.astype("float32")

class PositionalEmbedding(nn.Layer):
    def __init__(self, max_length,emb_size):
        super(PositionalEmbedding, self).__init__()
        self.emb_size = emb_size
        # 使用三角函数初始化Embedding
        self.pos_encoder = nn.Embedding(
            num_embeddings=max_length,
            embedding_dim=self.emb_size,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Assign(
                    get_sinusoid_encoding(max_length, self.emb_size))))

    def forward(self, pos):
        pos_emb = self.pos_encoder(pos)
        # 关闭位置编码的梯度更新
        pos_emb.stop_gradient = True
        return pos_emb

class TransformerEmbeddings(nn.Layer):
    """
    包括输入编码，分段编码，位置编码
    """
    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        hidden_dropout_prob=0.1,
        position_size=512,
        segment_size=2,
    ):
        super(TransformerEmbeddings, self).__init__()
        # 输入编码向量
        self.word_embeddings = WordEmbedding(vocab_size, hidden_size)
        # 位置编码向量
        self.position_embeddings = PositionalEmbedding(position_size, hidden_size)
        # 分段编码
        self.segment_embeddings = SegmentEmbedding(segment_size, hidden_size)
        # 层规范化
        self.layer_norm = nn.LayerNorm(hidden_size)
        # Dropout操作
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, segment_ids = None, position_ids = None):
        if position_ids is None:
            # 初始化全1的向量，比如[1,1,1,1]
            ones = paddle.ones_like(input_ids, dtype="int64")
            # 累加输入,求出序列前K个的长度,比如[1,2,3,4]
            seq_length = paddle.cumsum(ones, axis=-1)
            # position id的形式： 比如[0,1,2,3]
            position_ids = seq_length - ones
            position_ids.stop_gradient = True
        # 输入编码
        input_embedings = self.word_embeddings(input_ids)
        # 分段编码
        segment_embeddings = self.segment_embeddings(segment_ids)
        # 位置编码
        position_embeddings = self.position_embeddings(position_ids)
        # 输入张量, 分段张量，位置张量进行叠加
        embeddings = input_embedings + segment_embeddings + position_embeddings
        # 层规范化
        embeddings = self.layer_norm(embeddings)
        # Dropout
        embeddings = self.dropout(embeddings)
        return embeddings

class AddNorm(nn.Layer):
    """加与规范化"""
    def __init__(self, size, dropout_rate):
        super(AddNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X, H):
        """
            X：表示被包裹的非线性层的输入
            H：表示被包裹的非线性层的输出
        """
        H = X + self.dropout(H)
        return self.layer_norm(H)

class PositionwiseFFN(nn.Layer):
    """逐位前馈层"""
    def __init__(self, input_size, mid_size, dropout=0.1):
        super(PositionwiseFFN, self).__init__()
        self.W_1 = nn.Linear(input_size, mid_size)
        self.W_2 = nn.Linear(mid_size, input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        return self.W_2(self.dropout(F.relu(self.W_1(X))))

class TransformerBlock(nn.Layer):
    def __init__(
        self,
        input_size,
        head_num,
        ffn_size,
        dropout=0.1,
        attn_dropout=None,
        act_dropout=None,
    ):
        super(TransformerBlock, self).__init__()
        # 输入数据的维度
        self.input_size = input_size
        # 多头自注意力多头的个数
        self.head_num = head_num
        # 逐位前馈层的大小
        self.ffn_size = ffn_size
        # 加与规范化里面 Dropout的参数
        self.dropout = dropout
        # 多头注意力里面的 Dropout参数
        self.attn_dropout = dropout if attn_dropout is None else attn_dropout
        # 逐位前馈层里面的 Dropout参数
        self.act_dropout = dropout if act_dropout is None else act_dropout
        # 多头自注意力机制
        self.multi_head_attention = nn.MultiHeadAttention(
            self.input_size,
            self.head_num,
            dropout=self.attn_dropout,
            need_weights=True,
        )
        # 逐位前馈层
        self.ffn = PositionwiseFFN(self.input_size, self.ffn_size, self.act_dropout)
        # 加与规范化
        self.addnorm = AddNorm(self.input_size, self.dropout)

    def forward(self, X, src_mask=None):
        # 多头注意力
        X_atten, atten_weights = self.multi_head_attention(X, attn_mask=src_mask)
        # 加与规范化
        X = self.addnorm(X, X_atten)
        # 前馈层
        X_ffn = self.ffn(X)
        # 加与规范化
        X = self.addnorm(X, X_ffn)
        return X, atten_weights

class Model_Transformer(nn.Layer):
    def __init__(
        self,
        vocab_size,
        n_block=2,
        hidden_size=768,
        heads_num=12,
        intermediate_size=3072,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        act_dropout=0,
        position_size=512,
        num_classes=2,
        padding_idx=0,
    ):
        super(Model_Transformer, self).__init__()
        # 词表大小
        self.vocab_size = vocab_size
        # Transformer的编码器的数目
        self.n_block = n_block
        # 每个词映射成稠密向量的维度
        self.hidden_size = hidden_size
        # 多头注意力的个数
        self.heads_num = heads_num
        # 逐位前馈层的的维度
        self.intermediate_size = intermediate_size
        # Embedding层的 Dropout
        self.hidden_dropout = hidden_dropout
        # 多头注意力的dropout的 dropout参数
        self.attention_dropout = attention_dropout
        # 位置编码的大小 position_size
        self.position_size = position_size
        # 类别数
        self.num_classes = num_classes
        # 逐位前馈层的dropout
        self.act_dropout = act_dropout
        # [PAD]字符的ID
        self.padding_idx = padding_idx
        # 实例化输入编码，分段编码和位置编码
        self.embeddings = TransformerEmbeddings(
            self.vocab_size, self.hidden_size, self.hidden_dropout, self.position_size )
        # 实例化Transformer的编码器
        self.layers = nn.LayerList([])
        for i in range(n_block):
            encoder_layer = TransformerBlock(
                hidden_size,
                heads_num,
                intermediate_size,
                dropout=hidden_dropout,
                attn_dropout=attention_dropout,
                act_dropout=act_dropout,
            )
            self.layers.append(encoder_layer)
        # 全连接层
        self.dense = nn.Linear(hidden_size, hidden_size)
        # 双曲正切激活函数
        self.activation = nn.Tanh()
        # 最后一层分类器
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs, position_ids=None, attention_mask=None):
        input_ids, segment_ids = inputs
        # 构建Mask矩阵，把Pad的位置即input_ids中为0的位置设置为True,非0的位置设置为False
        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.padding_idx).astype("float32") * -1e9, axis=[1, 2] )
        # 抽取特征向量
        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, segment_ids=segment_ids )
        sequence_output = embedding_output
        self._attention_weights = []
        # Transformer的输出和注意力权重的输出
        for i, encoder_layer in enumerate(self.layers):
            sequence_output, atten_weights = encoder_layer(
                sequence_output, src_mask=attention_mask )
            self._attention_weights.append(atten_weights)
        # 选择第0个位置的向量作为句向量
        first_token_tensor = sequence_output[:, 0]
        # 输出层
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        # 句子级别的输出经过分类器
        logits = self.classifier(pooled_output)
        return logits

    @property
    def attention_weights(self):
        return self._attention_weights

@lru_cache(maxsize=1)
def load_model():
    vocab_path = ROOT / 'models' / 'transformer' / 'vocab.txt'
    vocabulary = {line.strip(): i for i, line in enumerate(vocab_path.read_text(encoding='utf-8').splitlines())}
    model = Model_Transformer(vocab_size=len(vocabulary), n_block=1, num_classes=2,
                              heads_num=4, padding_idx=vocabulary['[PAD]'])
    path = Path(os.environ.get('HOMEWORK_MODEL_PATH', ROOT / 'models' / 'transformer' / 'model_best.pdparams'))
    model.set_state_dict(paddle.load(str(path)))
    model.eval()
    return model, vocabulary


def calculate_similarity(text_a, text_b):
    if not isinstance(text_a, str) or not isinstance(text_b, str) or not text_a.strip() or not text_b.strip():
        raise ValueError('Both texts must be nonempty strings')
    if len(text_a) + len(text_b) + 3 > 512:
        raise ValueError('Combined text exceeds the 512-token model capacity')
    model, vocabulary = load_model()
    a = [vocabulary.get(c, vocabulary['[UNK]']) for c in text_a]
    b = [vocabulary.get(c, vocabulary['[UNK]']) for c in text_b]
    ids = [vocabulary['[CLS]']] + a + [vocabulary['[SEP]']] + b + [vocabulary['[SEP]']]
    segments = [0] * (len(a) + 2) + [1] * (len(b) + 1)
    with paddle.no_grad():
        logits = model([paddle.to_tensor([ids], dtype='int64'), paddle.to_tensor([segments], dtype='int64')])
        return float(F.softmax(logits, axis=-1).numpy()[0][1])
