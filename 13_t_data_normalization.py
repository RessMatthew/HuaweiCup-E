import pandas as pd
import numpy as np

# Read the data
df = pd.read_csv('/Users/matthew/Workspace/HuaweiCup-E/data/t_data_特征提取汇总.csv')

# Apply z-score normalization to all feature columns
# z-score = (x - mean) / std
normalized_features = df.copy()
for col in df.columns:
    mean_val = df[col].mean()
    std_val = df[col].std()
    if std_val != 0:  # Avoid division by zero
        normalized_features[col] = (df[col] - mean_val) / std_val
    else:
        normalized_features[col] = 0

# Save the processed data
normalized_features.to_csv('/Users/matthew/Workspace/HuaweiCup-E/data/t_data_特征提取汇总_标准化.csv', index=False)

print("数据标准化处理完成!")
print(f"特征列使用z-score标准化")
print(f"处理后的数据保存到: data/t_data_特征提取汇总_标准化.csv")