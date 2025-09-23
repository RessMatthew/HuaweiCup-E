import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys
import argparse

def reduce_dimensions(features, method='tsne', n_components=2, random_state=42):
    """
    使用指定的方法进行降维

    Args:
        features: 输入特征数据
        method: 降维方法 ('tsne')
        n_components: 降维后的维度数
        random_state: 随机种子

    Returns:
        降维后的数据
    """
    if method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=random_state, perplexity=30)
        reduced_data = reducer.fit_transform(features)
        print(f"{method.upper()} reduction completed")
    else:
        raise ValueError(f"Unknown method: {method}")

    return reduced_data

def plot_domain_alignment_comparison():
    """
    绘制域对齐前后的t-SNE对比图
    """
    # --- 1. 数据加载和准备 ---
    source_file = 'data/data_out.csv'
    target_before_file = 'data/t_data_out.csv'
    target_after_file = 'data/t_data_mmd_aligned.csv'

    try:
        source_data = pd.read_csv(source_file)
        target_before_data = pd.read_csv(target_before_file)
        target_after_data = pd.read_csv(target_after_file)
    except FileNotFoundError as e:
        print(f"错误：找不到文件 {e}")
        return

    # 提取源域特征和标签（第一列是标签）
    source_features = source_data.iloc[:, 1:]
    source_labels = source_data.iloc[:, 0]

    # 提取目标域特征（无标签）
    target_before_features = target_before_data
    target_after_features = target_after_data

    # 确保特征维度一致
    if not (source_features.shape[1] == target_before_features.shape[1] == target_after_features.shape[1]):
        print("错误：特征维度不一致")
        return

    print(f"源域数据形状: {source_features.shape}")
    print(f"目标域对齐前数据形状: {target_before_features.shape}")
    print(f"目标域对齐后数据形状: {target_after_features.shape}")

    # --- 2. 合并数据进行t-SNE分析 ---
    # 为不同域创建标签
    domain_labels = []
    domain_labels.extend(['Source'] * len(source_features))
    domain_labels.extend(['Target_Before'] * len(target_before_features))
    domain_labels.extend(['Target_After'] * len(target_after_features))

    # 合并所有特征数据
    all_features = pd.concat([source_features, target_before_features, target_after_features], ignore_index=True)

    # --- 3. 配色方案 ---
    # 域配色
    domain_colors = {
        'Source': '#237B9F',      # 蓝色
        'Target_Before': '#AD0B08',  # 红色
        'Target_After': '#71BFB2'    # 绿色
    }

    # --- 4. t-SNE降维 ---
    print("开始t-SNE降维...")
    features_2d = reduce_dimensions(all_features, 'tsne', n_components=2)

    # --- 5. 绘制对比图 ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # 获取每个域的数据索引范围
    source_end = len(source_features)
    target_before_end = source_end + len(target_before_features)
    target_after_end = target_before_end + len(target_after_features)

    # 图1：源域和目标域对齐前
    ax1.scatter(
        features_2d[:source_end, 0],
        features_2d[:source_end, 1],
        c=domain_colors['Source'],
        label='Source Domain',
        alpha=0.7,
        s=50
    )
    ax1.scatter(
        features_2d[source_end:target_before_end, 0],
        features_2d[source_end:target_before_end, 1],
        c=domain_colors['Target_Before'],
        label='Target Domain (Before)',
        alpha=0.7,
        s=50
    )
    ax1.set_title('Before Alignment', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 图2：源域和目标域对齐后
    ax2.scatter(
        features_2d[:source_end, 0],
        features_2d[:source_end, 1],
        c=domain_colors['Source'],
        label='Source Domain',
        alpha=0.7,
        s=50
    )
    ax2.scatter(
        features_2d[target_before_end:target_after_end, 0],
        features_2d[target_before_end:target_after_end, 1],
        c=domain_colors['Target_After'],
        label='Target Domain (After)',
        alpha=0.7,
        s=50
    )
    ax2.set_title('After Alignment', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 图3：所有域一起显示
    ax3.scatter(
        features_2d[:source_end, 0],
        features_2d[:source_end, 1],
        c=domain_colors['Source'],
        label='Source Domain',
        alpha=0.7,
        s=50
    )
    ax3.scatter(
        features_2d[source_end:target_before_end, 0],
        features_2d[source_end:target_before_end, 1],
        c=domain_colors['Target_Before'],
        label='Target Domain (Before)',
        alpha=0.7,
        s=50
    )
    ax3.scatter(
        features_2d[target_before_end:target_after_end, 0],
        features_2d[target_before_end:target_after_end, 1],
        c=domain_colors['Target_After'],
        label='Target Domain (After)',
        alpha=0.7,
        s=50
    )
    ax3.set_title('Combined View', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 设置总标题
    fig.suptitle('Domain Alignment Comparison (t-SNE 2D)', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # 保存图片
    plt.savefig('daan_alignment_comparison.png', dpi=300, bbox_inches='tight')
    print("对比图已保存为 'daan_alignment_comparison.png'")

def main():
    parser = argparse.ArgumentParser(description='域对齐前后t-SNE对比可视化')
    parser.add_argument('--random-state', type=int, default=42, help='随机种子 (默认: 42)')

    args = parser.parse_args()

    print("开始域对齐对比分析...")
    plot_domain_alignment_comparison()
    print("分析完成！")

if __name__ == '__main__':
    main()