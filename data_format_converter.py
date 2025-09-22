import os
import scipy.io
import pandas as pd
import numpy as np
from pathlib import Path

def convert_mat_to_csv(input_dir, output_dir):
    """
    将指定目录下的所有.mat文件转换为.csv文件，保留目录结构
    
    Args:
        input_dir (str): 输入目录路径
        output_dir (str): 输出目录路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 递归遍历输入目录
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.mat'):
                # 构建完整的输入文件路径
                input_file_path = os.path.join(root, file)
                
                # 计算相对路径以保持目录结构
                relative_path = os.path.relpath(root, input_dir)
                
                # 构建输出文件路径
                if relative_path == '.':
                    output_subdir = output_dir
                else:
                    output_subdir = os.path.join(output_dir, relative_path)
                
                # 确保输出子目录存在
                os.makedirs(output_subdir, exist_ok=True)
                
                # 构建输出文件名（将.mat替换为.csv）
                output_file_name = os.path.splitext(file)[0] + '.csv'
                output_file_path = os.path.join(output_subdir, output_file_name)
                
                print(f"正在转换: {input_file_path} -> {output_file_path}")
                
                try:
                    # 读取.mat文件
                    mat_data = scipy.io.loadmat(input_file_path)
                    
                    # 处理.mat文件内容
                    df_list = []
                    
                    # 遍历.mat文件中的所有变量
                    for key, value in mat_data.items():
                        # 跳过MATLAB的元数据字段
                        if key.startswith('__'):
                            continue
                        
                        # 如果是数值数组，转换为DataFrame
                        if isinstance(value, np.ndarray):
                            # 处理不同维度的数组
                            if value.ndim == 1:
                                # 一维数组，创建单列DataFrame
                                df = pd.DataFrame(value, columns=[key])
                            elif value.ndim == 2:
                                # 二维数组，每列作为一个变量
                                rows, cols = value.shape
                                if cols == 1:
                                    # 单列数据
                                    df = pd.DataFrame(value, columns=[key])
                                else:
                                    # 多列数据，为每列创建变量名
                                    col_names = [f"{key}_{i}" for i in range(cols)]
                                    df = pd.DataFrame(value, columns=col_names)
                            else:
                                # 高维数组，展平处理
                                flattened = value.flatten()
                                df = pd.DataFrame(flattened, columns=[key])
                            
                            df_list.append(df)
                    
                    # 如果没有找到有效数据，跳过该文件
                    if not df_list:
                        print(f"警告: {input_file_path} 中没有找到有效数据")
                        continue
                    
                    # 合并所有DataFrame
                    if len(df_list) == 1:
                        final_df = df_list[0]
                    else:
                        final_df = pd.concat(df_list, axis=1)
                    
                    # 保存为CSV文件
                    final_df.to_csv(output_file_path, index=False)
                    print(f"成功转换: {output_file_path}")
                    
                except Exception as e:
                    print(f"转换失败: {input_file_path} - 错误: {str(e)}")
                    continue

def main():
    """主函数"""
    # 设置输入和输出目录
    input_directory = "数据集"
    output_directory = "数据集-csv"
    
    print(f"开始转换: {input_directory} -> {output_directory}")
    print("=" * 50)
    
    # 执行转换
    convert_mat_to_csv(input_directory, output_directory)
    
    print("=" * 50)
    print("转换完成!")

if __name__ == "__main__":
    main()