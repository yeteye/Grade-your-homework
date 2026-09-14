import os
import numpy as np
from pathlib import Path

# 读取词汇表
def load_vocab():
    word_dict={}
    with open(Path(__file__).resolve().parents[2] / 'models/transformer/vocab.txt', encoding='utf-8') as f:
        for idx,item in enumerate(f.readlines()):
            word_dict[item.strip()]=idx

    return word_dict

def load_dataset(data_path, is_test):
        examples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if is_test:
                    text_a,text_b = line.strip().split('\t')
                    examples.append((text_a,text_b,))
                else:
                    text_a,text_b,label = line.strip().split('\t')
                    examples.append((text_a,text_b,label))
        return examples



def convert_example(example, is_test=False):
    if is_test:
        text = example
    else:
        text, label = example
    input_ids = text
    valid_length = np.array(len(input_ids), dtype='int64')
    input_ids = np.array(input_ids, dtype='int64')
    if not is_test:
        label = np.array(label, dtype="int64")
        return input_ids, label
    else:
        return input_ids

def load_lcqmc_data(path):
    train_path=os.path.join(path,'train.tsv')
    dev_path=os.path.join(path,'dev.tsv')
    test_path=os.path.join(path,'test.tsv')

    train_data = load_dataset(train_path, False)
    dev_data = load_dataset(dev_path, False)
    test_data = load_dataset(test_path, False)
    return train_data,dev_data,test_data

def load_thucnews_data(path):
    train_path=os.path.join(path,'train.txt')
    dev_path=os.path.join(path,'val.txt')
    test_path=os.path.join(path,'test.txt')

    train_data = load_dataset(train_path, False)
    dev_data = load_dataset(dev_path, False)
    test_data = load_dataset(test_path, False)
    return train_data,dev_data,test_data


import paddle
from paddle.io import Dataset

class LCQMCDataset(Dataset):
    def __init__(self, data, word2id_dict):
        # 词表
        self.word2id_dict = word2id_dict
        # 数据
        self.examples = data
        # ['CLS']的id，占位符
        self.cls_id = self.word2id_dict['[CLS]']
        # ['SEP']的id，句子的分隔
        self.sep_id = self.word2id_dict['[SEP]']

    def __getitem__(self, idx):
        # 返回单条样本
        example = self.examples[idx]
        text, segment, label = self.words_to_id(example)
        return text, segment, label

    def __len__(self):
        # 返回样本的个数
        return len(self.examples)

    def words_to_id(self, example):
        text_a, text_b, label = example
        # text_a 转换成id的形式
        input_ids_a = [self.word2id_dict[item] if item in self.word2id_dict else self.word2id_dict['[UNK]'] for item in text_a]
        # text_b 转换成id的形式
        input_ids_b = [self.word2id_dict[item] if item in self.word2id_dict else self.word2id_dict['[UNK]'] for item in text_b]
        # 加入[CLS],[SEP]
        input_ids = [self.cls_id]+ input_ids_a + [self.sep_id] + input_ids_b + [self.sep_id]
        # 对句子text_a,text_b做id的区分，进行的分隔
        segment_ids = [0]*(len(input_ids_a)+2)+[1]*(len(input_ids_b)+1)
        return input_ids, segment_ids, int(label)

    @property
    def label_list(self):
        # 0表示不相似，1表示相似
        return ['0', '1']

def collate_fn(batch_data, pad_val=0, max_seq_len=512):
    input_ids, segment_ids, labels = [], [], []
    max_len = 0
    # print(batch_data)
    for example in batch_data:
        input_id, segment_id, label = example
        # 对数据序列进行截断
        input_ids.append(input_id[:max_seq_len])
        segment_ids.append(segment_id[:max_seq_len])
        labels.append(label)
        # 保存序列最大长度
        max_len = max(max_len, min(max_seq_len, len(input_id)))
    # 对数据序列进行填充至最大长度
    for i in range(len(labels)):
        input_ids[i] = input_ids[i]+[pad_val] * (max_len - len(input_ids[i]))
        segment_ids[i] = segment_ids[i]+[pad_val] * (max_len - len(segment_ids[i]))
    return (
        paddle.to_tensor(input_ids),
        paddle.to_tensor(segment_ids),
    ), paddle.to_tensor(labels)
