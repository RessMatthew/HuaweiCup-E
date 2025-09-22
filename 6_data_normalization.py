import pandas as pd
import numpy as np

# Read the data
df = pd.read_csv('/Users/matthew/Workspace/HuaweiCup-E/data/data_特征提取汇总.csv')

# Separate target_label and features
target_label = df['target_label']
features = df.drop('target_label', axis=1)

# Apply integer encoding to target_label using pandas
unique_labels = target_label.unique()
label_mapping = {label: i for i, label in enumerate(unique_labels)}
encoded_labels = target_label.map(label_mapping)

# Apply z-score normalization to all feature columns
# z-score = (x - mean) / std
normalized_features = features.copy()
for col in features.columns:
    mean_val = features[col].mean()
    std_val = features[col].std()
    if std_val != 0:  # Avoid division by zero
        normalized_features[col] = (features[col] - mean_val) / std_val
    else:
        normalized_features[col] = 0

# Combine the results
processed_df = pd.DataFrame({'target_label': encoded_labels})
processed_df = pd.concat([processed_df, normalized_features], axis=1)

# Save the processed data
processed_df.to_csv('/Users/matthew/Workspace/HuaweiCup-E/data/data_特征提取汇总_标准化.csv', index=False)

print("数据标准化处理完成!")
print(f"原始target_label唯一值: {unique_labels}")
print(f"编码后target_label范围: {encoded_labels.min()} 到 {encoded_labels.max()}")
print(f"特征列使用z-score标准化")
print(f"处理后的数据保存到: data/data_特征提取汇总_标准化.csv")