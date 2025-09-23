import pandas as pd

# Read the original CSV file
df = pd.read_csv('data/data_特征提取汇总_标准化.csv')

# Define the target labels subset
target_labels = [
    'target_label', 'COR_BPFO', 'COR_BPFI', 'COR_BSF', 'high_freq_energy_ratio',
    'var', 'rms', 'peak_to_peak', 'low_freq_energy_ratio', 'peak',
    'mid_freq_energy_ratio', 'peak_frequency'
]

# Extract the subset of columns
subset_df = df[target_labels]

# Save the subset to a new CSV file
subset_df.to_csv('data/data_特征提取汇总_标准化_标签子集.csv', index=False)

print("标签子集已提取并保存到: data/data_out.csv")
print(f"提取的列数: {len(target_labels)}")
print(f"数据行数: {len(subset_df)}")