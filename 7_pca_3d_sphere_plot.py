import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

def plot_3d_scatter_in_sphere():
    """
    读取特征数据，进行PCA降维，并将数据点绘制在一个三维球体内部。
    """
    # --- 1. 数据加载和准备 ---
    file_path = 'data/data_特征提取汇总_标准化.csv'
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'。请确保文件路径正确。")
        return

    # 分离特征和标签
    target_labels = data.iloc[:, 0]
    features = data.iloc[:, 1:]
    
    # 获取唯一的类别和对应的颜色
    unique_labels = target_labels.unique()
    
    # --- 2. 配色方案 ---
    # 从 '配色方案.md' 获取
    colors = ['#237B9F', '#71BFB2', '#AD0B08', '#EC817E', '#FEE066']
    # 如果类别多于颜色，循环使用颜色
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
    point_colors = target_labels.map(color_map)
    alpha = 0.85

    # --- 3. PCA 降维 ---
    # 将特征降到3维
    pca = PCA(n_components=3)
    features_3d = pca.fit_transform(features)

    # 打印解释方差比
    print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
    print(f"Total Explained Variance: {sum(pca.explained_variance_ratio_):.2f}")

    # --- 4. 将数据点映射到球体空间 ---
    # 计算每个点到原点的距离
    distances = np.linalg.norm(features_3d, axis=1)
    # 找到最大距离，用于归一化，确保所有点都在球内
    max_distance = np.max(distances)
    # 归一化，使所有点都位于半径为1的球体内或表面
    features_spherical = features_3d / max_distance

    # --- 5. 绘制3D散点图和球体 ---
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制透明线框球体
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.2, rstride=5, cstride=5)

    # 绘制散点
    # 为了图例正确显示，我们按类别循环绘制
    for label in unique_labels:
        mask = target_labels == label
        ax.scatter(
            features_spherical[mask, 0],
            features_spherical[mask, 1],
            features_spherical[mask, 2],
            c=color_map[label],
            label=label,
            alpha=alpha,
            s=50  # 点的大小
        )
    
    # 设置坐标轴范围，使球体看起来更规整
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    
    # 隐藏坐标轴背景网格
    ax.grid(False)
    
    # 隐藏坐标轴平面
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # 移除坐标轴
    ax.set_axis_off()

    # 设置视角
    ax.view_init(elev=20, azim=30)
    
    # 添加图例，并调整位置使其不偏移
    ax.legend(title='Target Labels', loc='upper right', bbox_to_anchor=(1.0, 1.0))

    # 保存图片
    plt.savefig('pca_3d_scatter_in_sphere.png', dpi=300, bbox_inches='tight')

    plt.show()

if __name__ == '__main__':
    plot_3d_scatter_in_sphere()