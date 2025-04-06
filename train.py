import cantools
import can 
import pandas as pd
import os
import random
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

def calculate_entropy(data):
    """
    计算给定数据的熵值。

    参数:
    data (pd.Series): 输入数据列。

    返回:
    float: 计算得到的熵值。
    """
    probability = data.value_counts(normalize=True)  # 计算每个值的频率
    entropy = -np.sum(probability * np.log2(probability + 1e-10))  # 加上一个小常数以避免对数为零
    return entropy

def process_directory(input_directory, label):
    """
    读取指定目录中的 CSV 文件，计算信息熵，并生成表现 DataFrame。

    参数:
    input_directory (str): 输入 CSV 文件的目录路径。
    label (str): 为每个记录指定的标签。

    返回:
    pd.DataFrame: 包含文件名、信息熵和标签的 DataFrame。
    """
    results = []  # 用于存储结果的列表

    # 遍历文件夹中的所有 CSV 文件
    for filename in os.listdir(input_directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_directory, filename)
            df = pd.read_csv(file_path)

            # 计算信息熵
            entropy_value = calculate_entropy(df['arbitration_id'])

            # 添加结果
            results.append({
                # 'filename': filename,
                'entropy': entropy_value,
                'label': label
            })

    # 创建结果 DataFrame
    result_df = pd.DataFrame(results)
    return result_df



# 使用示例
input_directory = r"./can0_final/can500/normal"  # 输入文件夹路径
label = 0  # 指定标签
train_ndf = process_directory(input_directory, label)
input3_directory = r"./can0_final/can500/ddos"  # 输入文件夹路径
label3 = 1  # 指定标签
train_ddf = process_directory(input3_directory, label3)
train_df = pd.concat([train_ndf, train_ddf], ignore_index=True)
# 输出结果 DataFrame
print(train_df)


# 假设 train_df 是您的数据框
X = train_df['entropy'].values  # 特征数组
y = train_df['label'].values     # 标签数组

# Step 1: 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: 在正常样本上拟合高斯模型
normal_data = X_train[y_train == 0]
mu, sigma = np.mean(normal_data), np.std(normal_data)

# Step 3: 计算训练集的异常分数
cdf_values = norm.cdf(X_train, mu, sigma) 
tail_probs = np.where(X_train < mu, cdf_values, 1 - cdf_values)
two_tailed_p_values = 2 * tail_probs
anomaly_scores_train = 1 - two_tailed_p_values

# Step 4: 计算测试集的异常分数
cdf_values_test = norm.cdf(X_test, mu, sigma)
tail_probs_test = np.where(X_test < mu, cdf_values_test, 1 - cdf_values_test)
two_tailed_p_values_test = 2 * tail_probs_test
anomaly_scores_test = 1 - two_tailed_p_values_test

# Step 5: 计算 ROC 曲线并选择最佳阈值
fpr, tpr, thresholds = roc_curve(y_train, anomaly_scores_train)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# Step 6: 在测试集上进行预测
y_pred = (anomaly_scores_test > optimal_threshold).astype(int)  # 预测标签

# 输出结果
print("测试集的预测标签:", y_pred)
print(mu)
print(sigma)
print(optimal_threshold)



# 假设 y_pred 是您的预测结果，y 是原始标签
# y_pred = [0, 1, 0, 1, ...]
# y = [0, 1, 0, 0, ...]

# 计算准确度
accuracy = accuracy_score(y_test, y_pred)
print(f"准确度: {accuracy:.2f}")

# 计算混淆矩阵
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# 计算误检率（假阳性率）
false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
print(f"误检率: {false_positive_rate:.2f}")