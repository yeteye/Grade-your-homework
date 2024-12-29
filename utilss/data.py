import os
import numpy as np
from paddlenlp.transformers import BertTokenizer

# 读取词汇表
def load_vocab():
    word_dict={}
    with open('utilss/bert-base-chinese-vocab.txt', encoding='utf-8') as f:
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
    dev_path=os.path.join(path,'dev.csv')
    test_path=os.path.join(path,'test.csv')

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