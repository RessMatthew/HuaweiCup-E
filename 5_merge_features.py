import pandas as pd

# 文件路径
cor_file = './data/data_COR特征结果.csv'
freq_file = './data/data_频域特征结果.csv'
time_file = './data/data_时域特征结果.csv'
output_file = './data/data_特征提取汇总.csv'

# 读取三个 CSV 文件
cor_df = pd.read_csv(cor_file)
freq_df = pd.read_csv(freq_file)
time_df = pd.read_csv(time_file)

# 选择需要的特征列
cor_features = ['target_label', 'COR_BPFO', 'COR_BPFI', 'COR_BSF', 'COR_FTF']
freq_features = ['spectral_centroid', 'spectral_variance', 
                 'peak_frequency', 'peak_amplitude', 'low_freq_energy_ratio', 
                 'mid_freq_energy_ratio', 'high_freq_energy_ratio']
time_features = ['mean', 'std', 'var', 'rms', 'peak', 'peak_to_peak', 'skewness', 
                 'kurtosis', 'crest_factor', 'impulse_factor', 'shape_factor', 'clearance_factor']

# 提取指定特征
cor_selected = cor_df[cor_features]
freq_selected = freq_df[freq_features]
time_selected = time_df[time_features]

# 直接按列合并
merged_df = pd.concat([cor_selected, freq_selected, time_selected], axis=1)

# 保存合并后的结果
merged_df.to_csv(output_file, index=False)

print(f"合并完成，保存为 {output_file}，总行数：{len(merged_df)}")