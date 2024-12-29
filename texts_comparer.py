import hanlp
import json
import csv

class SemanticSimilarityCalculator:
    def __init__(self, input_file, output_file, num_entries=1000, similarity_threshold=0.7):
        """
        初始化语义相似度计算器。

        :param input_file: 输入的 JSON 文件路径，包含句子对及标签。
        :param output_file: 输出的 CSV 文件路径，用于保存相似度和匹配结果。
        :param num_entries: 处理的数据条目数，默认为1000条。
        :param similarity_threshold: 判定是否匹配的相似度阈值，默认为0.7。
        """
        self.input_file = input_file
        self.output_file = output_file
        self.num_entries = num_entries
        self.similarity_threshold = similarity_threshold
        self.similarity_model = hanlp.load('STS_ELECTRA_BASE_ZH')

    def load_data(self):
        """
        从输入的 JSON 文件加载数据，并返回句子对。
        """
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]
        # 只处理前 num_entries 条数据
        return data[:self.num_entries]

    def calculate_similarity(self, sentence1, sentence2):
        """
        计算两个句子的语义相似度。
        """
        return self.similarity_model((sentence1, sentence2))

    def is_match(self, similarity):
        """
        判断两个句子是否匹配，基于相似度阈值。
        """
        return 1 if similarity >= self.similarity_threshold else 0

    def save_to_csv(self, data):
        """
        将句子对和计算结果保存到 CSV 文件。
        """
        with open(self.output_file, mode='w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['sentence1', 'sentence2', '匹配度', '是否匹配', '真实标签']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # 写入表头
            writer.writeheader()

            # 遍历数据并写入
            for entry in data:
                sentence1 = entry["sentence1"]
                sentence2 = entry["sentence2"]
                true_label = entry["label"]  # 真实标签

                # 计算语义相似度
                similarity = self.calculate_similarity(sentence1, sentence2)

                # 判定是否匹配
                predicted_label = self.is_match(similarity)

                # 写入到 CSV 文件
                writer.writerow({
                    'sentence1': sentence1,
                    'sentence2': sentence2,
                    '匹配度': round(similarity, 4),  # 保留4位小数
                    '是否匹配': predicted_label,
                    '真实标签': true_label
                })

    def calculate_accuracy(self):
        """
        计算模型的准确率，并输出假阳性和假阴性的比例。
        """
        # 读取 CSV 文件中保存的数据
        correct_predictions = 0
        total_predictions = 0
        false_positive = 0
        false_negative = 0

        with open(self.output_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                true_label = int(row['真实标签'])
                predicted_label = int(row['是否匹配'])

                # 判断预测结果与真实标签是否一致
                if predicted_label == true_label:
                    correct_predictions += 1
                else:
                    if predicted_label == 1 and true_label == 0:
                        false_positive += 1  # 错误的判断为正确（假阳性）
                    elif predicted_label == 0 and true_label == 1:
                        false_negative += 1  # 正确的判断为错误（假阴性）
                
                total_predictions += 1

        # 计算准确率
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        false_positive_rate = false_positive / total_predictions if total_predictions > 0 else 0
        false_negative_rate = false_negative / total_predictions if total_predictions > 0 else 0

        return accuracy, false_positive_rate, false_negative_rate

    def calculate_similarity_and_save(self):
        """
        计算句子对的语义相似度并将结果保存到 CSV 文件。
        """
        # 加载数据
        data = self.load_data()

        # 保存结果到 CSV 文件
        self.save_to_csv(data)

        print(f"结果已保存到 {self.output_file}")


# 使用类来处理
input_file = 'data/TXT_data/simclue_public/train_pair.json'
output_file = 'semantic_similarity_results.csv'
num_entries = 1000

# 创建实例并计算相似度
calculator = SemanticSimilarityCalculator(input_file, output_file, num_entries=num_entries, similarity_threshold=0.4)
calculator.calculate_similarity_and_save()

# 计算并打印准确率及错误比例
accuracy, false_positive_rate, false_negative_rate = calculator.calculate_accuracy()
print(f"模型的准确率: {accuracy * 100:.2f}%")
print(f"假阳性比例 (错误的判断为正确): {false_positive_rate * 100:.2f}%")
print(f"假阴性比例 (正确的判断为错误): {false_negative_rate * 100:.2f}%")
