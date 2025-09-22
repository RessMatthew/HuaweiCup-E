# 时域图绘制工具
# 根据输入的CSV文件路径，自动检测表头中的DE、FE、BA字段，并绘制相应数量的时域图
# 时域信号可视化工具 - 绘制轴承传感器数据的时域波形图，支持批量处理和自定义样式

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
from math import pi
from matplotlib import rcParams
from matplotlib.patches import Rectangle
import os
from pathlib import Path

def setup_chinese_font():
    """
    设置matplotlib支持中文显示的字体
    """
    # 常见的中文字体列表
    chinese_fonts = [
        'SimHei', 'Microsoft YaHei', 'PingFang SC', 'Heiti TC', 
        'STHeiti', 'Arial Unicode MS', 'WenQuanYi Micro Hei',
        'Noto Sans CJK SC', 'Source Han Sans CN', 'DejaVu Sans'
    ]
    
    available_fonts = []
    
    # 检测可用的中文字体
    for font_name in chinese_fonts:
        try:
            font_path = fm.findfont(font_name, fallback_to_default=False)
            available_fonts.append(font_name)
            print(f"✅ 找到可用中文字体: {font_name}")
        except:
            continue
    
    if available_fonts:
        # 设置字体回退列表，确保中文字体能正常显示
        plt.rcParams['font.sans-serif'] = available_fonts
        plt.rcParams['axes.unicode_minus'] = False
        print(f"✅ 成功设置中文字体: {available_fonts}")
        return True
    else:
        print("⚠️ 警告：未找到可用的中文字体，图表中的中文可能无法正常显示")
        return False


def set_chart_style():
    """
    设置图表样式，保留中文字体设置
    """
    # 保存当前字体设置
    current_font = plt.rcParams.get('font.sans-serif', [])
    
    # 设置字体大小
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 12
    
    # 线条宽度
    plt.rcParams['lines.linewidth'] = 1.0
    plt.rcParams['axes.linewidth'] = 0.5
    
    # 图表尺寸
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['figure.dpi'] = 100
    
    # 设置主颜色为 #578DAA
    main_color = '#578DAA'
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[main_color])
    plt.rcParams['lines.color'] = main_color
    plt.rcParams['patch.edgecolor'] = main_color
    
    # 设置背景为纯白色
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    # 其他样式设置
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.05
    plt.rcParams['savefig.dpi'] = 300
    
    # 恢复中文字体设置
    if current_font:
        plt.rcParams['font.sans-serif'] = current_font


# 设置图表样式
plt.style.use('seaborn-v0_8')

# 首先设置中文字体
setup_chinese_font()

# 然后设置其他图表样式
set_chart_style()


def plot_time_domain_graphs(file_path, max_points=10000):
    """
    根据CSV文件绘制时域图
    
    参数:
    file_path (str): CSV文件路径
    max_points (int): 最大显示数据点数，避免图表过于密集
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 获取列名
        columns = df.columns.tolist()
        
        print(f"文件: {file_path}")
        print(f"数据形状: {df.shape}")
        print(f"列名: {columns}")
        
        # 检测包含DE、FE、BA的列
        de_cols = [col for col in columns if 'DE_time' in col]
        fe_cols = [col for col in columns if 'FE_time' in col]
        ba_cols = [col for col in columns if 'BA_time' in col]
        
        # 获取RPM列
        rpm_cols = [col for col in columns if 'RPM' in col]
        
        print(f"\n检测到的传感器数据:")
        print(f"DE列: {de_cols}")
        print(f"FE列: {fe_cols}")
        print(f"BA列: {ba_cols}")
        print(f"RPM列: {rpm_cols}")
        
        # 合并所有要绘制的列
        plot_cols = de_cols + fe_cols + ba_cols
        
        if not plot_cols:
            print("\n未找到DE、FE、BA时间序列数据！")
            return
        
        # 创建图表
        n_plots = len(plot_cols)
        fig, axes = plt.subplots(n_plots, 1, figsize=(15, 4 * n_plots))
        
        # 如果只有一个图，将axes转换为列表
        if n_plots == 1:
            axes = [axes]
        
        # 绘制每个传感器数据
        for i, col in enumerate(plot_cols):
            # 获取数据，限制显示点数
            data = df[col].dropna()
            if len(data) > max_points:
                # 均匀采样
                indices = np.linspace(0, len(data)-1, max_points, dtype=int)
                data = data.iloc[indices]
            
            # 创建时间序列（使用索引作为时间）
            time_index = range(len(data))
            
            # 绘制图表
            axes[i].plot(time_index, data, linewidth=0.8)
            axes[i].set_title(f'{col} 时域信号', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('时间点', fontsize=12)
            axes[i].set_ylabel('幅值', fontsize=12)
            axes[i].grid(True, alpha=0.3)
            
            # 如果有RPM数据，显示在标题中
            if rpm_cols and len(df[rpm_cols[0]].dropna()) > 0:
                rpm_value = df[rpm_cols[0]].iloc[0]
                axes[i].set_title(f'{col} 时域信号 (RPM: {rpm_value:.1f})',
                                fontsize=14, fontweight='bold')
            
            # 添加黑长方体框
            # 获取当前坐标轴的范围
            xlim = axes[i].get_xlim()
            ylim = axes[i].get_ylim()
            
            # 创建黑长方体框
            rect = Rectangle((xlim[0], ylim[0]),
                           xlim[1] - xlim[0],
                           ylim[1] - ylim[0],
                           linewidth=2,
                           edgecolor='black',
                           facecolor='none',
                           zorder=10)
            axes[i].add_patch(rect)
            
            # 添加表上刻度
            # 设置x轴和y轴的主要刻度
            axes[i].xaxis.set_major_locator(plt.MaxNLocator(5))
            axes[i].yaxis.set_major_locator(plt.MaxNLocator(5))
            
            # 设置刻度标签
            axes[i].tick_params(axis='both', which='major', labelsize=10)
        
        plt.tight_layout()
        plt.show()
        
        # 显示数据统计信息
        print("\n数据统计信息:")
        for col in plot_cols:
            col_data = df[col].dropna()
            print(f"\n{col}:")
            print(f"  数据点数: {len(col_data)}")
            print(f"  最大值: {col_data.max():.6f}")
            print(f"  最小值: {col_data.min():.6f}")
            print(f"  均值: {col_data.mean():.6f}")
            print(f"  标准差: {col_data.std():.6f}")
            
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 不存在！")
    except Exception as e:
        print(f"处理文件时发生错误: {e}")
        import traceback
        traceback.print_exc()


def plot_and_save_time_domain_graphs(file_path, output_dir, max_points=10000):
    """
    绘制时域图并保存到指定目录
    
    参数:
    file_path (str): CSV文件路径
    output_dir (str): 输出目录路径
    max_points (int): 最大显示数据点数
    """
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 获取列名
        columns = df.columns.tolist()
        
        print(f"处理文件: {file_path}")
        print(f"输出目录: {output_dir}")
        print(f"数据形状: {df.shape}")
        
        # 检测包含DE、FE、BA的列
        de_cols = [col for col in columns if 'DE_time' in col]
        fe_cols = [col for col in columns if 'FE_time' in col]
        ba_cols = [col for col in columns if 'BA_time' in col]
        
        # 获取RPM列
        rpm_cols = [col for col in columns if 'RPM' in col]
        
        print(f"\n检测到的传感器数据:")
        print(f"DE列: {de_cols}")
        print(f"FE列: {fe_cols}")
        print(f"BA列: {ba_cols}")
        print(f"RPM列: {rpm_cols}")
        
        # 合并所有要绘制的列
        plot_cols = de_cols + fe_cols + ba_cols
        
        if not plot_cols:
            print("\n未找到DE、FE、BA时间序列数据！")
            return
        
        # 创建图表
        n_plots = len(plot_cols)
        fig, axes = plt.subplots(n_plots, 1, figsize=(15, 4 * n_plots))
        
        # 如果只有一个图，将axes转换为列表
        if n_plots == 1:
            axes = [axes]
        
        # 绘制每个传感器数据
        for i, col in enumerate(plot_cols):
            # 获取数据，限制显示点数
            data = df[col].dropna()
            if len(data) > max_points:
                # 均匀采样
                indices = np.linspace(0, len(data)-1, max_points, dtype=int)
                data = data.iloc[indices]
            
            # 创建时间序列（使用索引作为时间）
            time_index = range(len(data))
            
            # 绘制图表
            axes[i].plot(time_index, data, linewidth=0.8)
            axes[i].set_title(f'{col} 时域信号', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('时间点', fontsize=12)
            axes[i].set_ylabel('幅值', fontsize=12)
            axes[i].grid(True, alpha=0.3)
            
            # 如果有RPM数据，显示在标题中
            if rpm_cols and len(df[rpm_cols[0]].dropna()) > 0:
                rpm_value = df[rpm_cols[0]].iloc[0]
                axes[i].set_title(f'{col} 时域信号 (RPM: {rpm_value:.1f})',
                                fontsize=14, fontweight='bold')
            
            # 添加黑长方体框
            # 获取当前坐标轴的范围
            xlim = axes[i].get_xlim()
            ylim = axes[i].get_ylim()
            
            # 创建黑长方体框
            rect = Rectangle((xlim[0], ylim[0]),
                           xlim[1] - xlim[0],
                           ylim[1] - ylim[0],
                           linewidth=2,
                           edgecolor='black',
                           facecolor='none',
                           zorder=10)
            axes[i].add_patch(rect)
            
            # 添加表上刻度
            # 设置x轴和y轴的主要刻度
            axes[i].xaxis.set_major_locator(plt.MaxNLocator(5))
            axes[i].yaxis.set_major_locator(plt.MaxNLocator(5))
            
            # 设置刻度标签
            axes[i].tick_params(axis='both', which='major', labelsize=10)
        
        plt.tight_layout()
        
        # 保存图表
        output_file = os.path.join(output_dir, os.path.basename(file_path).replace('.csv', '.png'))
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 图表已保存: {output_file}")
        
        # 显示数据统计信息
        print("\n数据统计信息:")
        for col in plot_cols:
            col_data = df[col].dropna()
            print(f"\n{col}:")
            print(f"  数据点数: {len(col_data)}")
            print(f"  最大值: {col_data.max():.6f}")
            print(f"  最小值: {col_data.min():.6f}")
            print(f"  均值: {col_data.mean():.6f}")
            print(f"  标准差: {col_data.std():.6f}")
            
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 不存在！")
    except Exception as e:
        print(f"处理文件时发生错误: {e}")
        import traceback
        traceback.print_exc()


def batch_process_csv_files(data_dir, output_base_dir, max_points=10000):
    """
    批量处理目录下的所有CSV文件
    
    参数:
    data_dir (str): 数据目录路径
    output_base_dir (str): 输出基础目录路径
    max_points (int): 最大显示数据点数
    """
    print(f"开始批量处理CSV文件...")
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_base_dir}")
    
    # 确保输出基础目录存在
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 递归查找所有CSV文件
    csv_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    # 处理每个CSV文件
    for i, csv_file in enumerate(csv_files, 1):
        print(f"\n{'='*50}")
        print(f"处理文件 {i}/{len(csv_files)}: {csv_file}")
        
        # 计算相对路径和输出目录
        relative_path = os.path.relpath(csv_file, data_dir)
        output_dir = os.path.join(output_base_dir, os.path.dirname(relative_path))
        
        try:
            plot_and_save_time_domain_graphs(csv_file, output_dir, max_points)
            print(f"✅ 文件处理完成: {csv_file}")
        except Exception as e:
            print(f"❌ 处理文件失败: {csv_file}")
            print(f"错误信息: {e}")
    
    print(f"\n{'='*50}")
    print(f"批量处理完成！共处理 {len(csv_files)} 个文件")


# 使用示例
if __name__ == "__main__":
    # 单个文件处理示例
    # file_path = "data/源域数据集/12kHz_DE_data/B/0007/B007_0.csv"
    # plot_time_domain_graphs(file_path)
    
    # 批量处理示例
    data_directory = "data/源域数据集/"
    output_directory = "out/时域图/"
    batch_process_csv_files(data_directory, output_directory)