import os
import numpy as np

# 文件路径
split_file_path = r"D:\py worksapce\DNN_attention_FPHA\dataset\data_split_action_recognition.txt"
base_data_path = r"D:\py worksapce\DNN_attention_FPHA\dataset\hand_skbone"

# 读取分割文件内容
with open(split_file_path, 'r') as file:
    lines = file.readlines()

# 初始化训练集和测试集
train_data = []
test_data = []

# 解析文件内容
is_train = True
for line in lines:
    line = line.strip()
    if line.startswith("Training"):
        is_train = True
        continue
    elif line.startswith("Test"):
        is_train = False
        continue

    # 获取数据路径和标签
    parts = line.split()
    data_path = parts[0]
    label = parts[1]

    # 构建完整的文件路径
    full_path = os.path.join(base_data_path, data_path, 'skeleton.txt')

    # 读取skeleton.txt文件内容
    if os.path.exists(full_path):
        data = np.genfromtxt(full_path, delimiter=' ')
        # 只取前30帧数据
        if data.shape[0] >= 30:
            data = data[:30]
        else:
            # 如果帧数不足30，进行补齐
            padding = np.zeros((30 - data.shape[0], data.shape[1]))
            data = np.vstack((data, padding))

        # 将数据和标签组合
        labeled_data = (data, label)

        if is_train:
            train_data.append(labeled_data)
        else:
            test_data.append(labeled_data)
    else:
        print(f"文件未找到: {full_path}")

# 将训练集和测试集写入新的文本文件
train_file_path = r"D:\py worksapce\DNN_attention_FPHA\train_data.txt"
test_file_path = r"D:\py worksapce\DNN_attention_FPHA\test_data.txt"


def save_to_txt(file_path, data):
    with open(file_path, 'w') as file:
        for frames, label in data:
            for frame in frames:
                file.write(' '.join(map(str, frame)) + ' ' + label + '\n')


save_to_txt(train_file_path, train_data)
save_to_txt(test_file_path, test_data)

print("数据集分割完成！")