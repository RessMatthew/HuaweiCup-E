#!/usr/bin/env python3
"""
SHAP可解释性分析脚本
用于对四分类XGBoost模型进行可解释性分析

生成内容：
- 力图（解释单次预测）：四个类各五张
- 摘要图（从总体上看，哪些特征最重要）：四个类各一张
- 图片生成在 ./out/SHAP 目录下

使用方法:
python 20_t_shap_explainability.py --model xgboost --data data/t_data_daan_aligned.csv --segments_per_sample 124
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import logging
from pathlib import Path
import argparse

# 添加ml目录到路径
sys.path.append('ml')

from model_inference import ModelInference
from ml_config import MLConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SHAPExplainability:
    """SHAP可解释性分析类"""

    def __init__(self, config: MLConfig):
        self.config = config
        self.inference = ModelInference(config)
        self.output_dir = Path("./out/SHAP")
        self.shap_values = None
        self.explainer = None
        self.X_sample = None

        # 配色方案 (85%透明度)
        self.color_scheme = {
            'primary': '#237B9F',      # 主色
            'secondary': '#71BFB2',    # 次要色
            'accent1': '#AD0B08',      # 强调色1
            'accent2': '#EC817E',      # 强调色2
            'accent3': '#FEE066',      # 强调色3
            'alpha': 0.85
        }

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 设置matplotlib样式
        plt.style.use('default')
        sns.set_palette([
            self.color_scheme['primary'],
            self.color_scheme['secondary'],
            self.color_scheme['accent1'],
            self.color_scheme['accent2'],
            self.color_scheme['accent3']
        ])

    def load_data_and_model(self, model_name: str, csv_path: str, segments_per_sample: int = 124):
        """加载数据和模型"""
        logger.info("="*60)
        logger.info("加载数据和模型")
        logger.info("="*60)

        # 加载数据
        df = pd.read_csv(csv_path)
        logger.info(f"数据文件: {csv_path}")
        logger.info(f"数据形状: {df.shape}")

        # 使用所有特征列（假设没有目标列）
        feature_columns = df.columns.tolist()
        X = df[feature_columns].values

        logger.info(f"特征数量: {len(feature_columns)}")
        logger.info(f"特征名称: {feature_columns[:10]}...")  # 显示前10个特征

        # 加载模型
        self.inference.load_models(model_name)
        logger.info(f"✓ 模型加载完成: {model_name}")

        return X, feature_columns

    def prepare_sample_data(self, X: np.ndarray, segments_per_sample: int = 124):
        """准备样本数据用于SHAP分析"""
        logger.info("准备样本数据...")

        total_samples = len(X) // segments_per_sample
        logger.info(f"总样本数: {total_samples}")

        # 从每个样本中随机选择一个段作为代表
        sample_indices = []
        for i in range(total_samples):
            start_idx = i * segments_per_sample
            end_idx = start_idx + segments_per_sample
            # 随机选择该样本的一个段
            random_segment_idx = np.random.choice(range(start_idx, end_idx))
            sample_indices.append(random_segment_idx)

        self.X_sample = X[sample_indices]
        logger.info(f"样本数据形状: {self.X_sample.shape}")

        return sample_indices

    def create_explainer(self, X: np.ndarray, feature_names: list):
        """创建SHAP解释器"""
        logger.info("创建SHAP解释器...")

        # 使用第一个模型创建解释器
        model = self.inference.loaded_models[self.inference.model_type][0]

        # 创建TreeExplainer（适用于XGBoost）
        self.explainer = shap.TreeExplainer(model)

        # 计算SHAP值
        logger.info("计算SHAP值...")
        self.shap_values = self.explainer.shap_values(X)

        logger.info(f"SHAP值形状: {np.array(self.shap_values).shape}")

        return self.shap_values

    def plot_force_plots(self, class_labels: list, feature_names: list, num_samples_per_class: int = 5):
        """生成力图 - 每个类别5个样本 (自定义实现模拟力图)"""
        logger.info("="*60)
        logger.info("生成力图 (每个类别5个样本)")
        logger.info("="*60)

        if self.shap_values is None or self.X_sample is None:
            raise ValueError("请先计算SHAP值")

        # 获取预测结果
        predictions = self.inference.predict_ensemble(self.X_sample)

        for class_label in class_labels:
            logger.info(f"生成类别 {class_label} 的力图...")

            # 找到该类别对应的样本索引
            class_indices = np.where(predictions == class_label)[0]

            if len(class_indices) == 0:
                logger.warning(f"类别 {class_label} 没有找到样本")
                continue

            # 如果样本数量不足，选择所有可用样本
            selected_indices = class_indices[:min(num_samples_per_class, len(class_indices))]

            for i, sample_idx in enumerate(selected_indices):
                try:
                    # 生成自定义力图 - 模拟SHAP力图的视觉效果
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 6),
                                                  gridspec_kw={'height_ratios': [3, 1]})

                    # 对于多分类问题
                    if isinstance(self.shap_values, list):
                        # 获取该类别的SHAP值和基线值
                        base_value = self.explainer.expected_value[class_label]
                        shap_values_single = self.shap_values[class_label][sample_idx]
                        features_single = self.X_sample[sample_idx]

                        # 确保数据类型正确
                        shap_values_single = np.array(shap_values_single).astype(float)
                        features_single = np.array(features_single).astype(float)
                    else:
                        # 二分类问题
                        base_value = self.explainer.expected_value
                        shap_values_single = np.array(self.shap_values[sample_idx]).astype(float)
                        features_single = np.array(self.X_sample[sample_idx]).astype(float)

                    # 关键：SHAP值形状应该是 (特征数,) 而不是 (特征数, 类别数)
                    # 如果SHAP值是多维的，需要正确处理
                    if len(shap_values_single.shape) == 2 and shap_values_single.shape[1] == len(feature_names):
                        # 形状为 (类别数, 特征数)，需要选择对应类别的SHAP值
                        shap_values_single = shap_values_single[class_label] if shap_values_single.shape[0] > class_label else shap_values_single[0]
                    elif len(shap_values_single.shape) == 2 and shap_values_single.shape[0] == len(feature_names):
                        # 形状为 (特征数, 类别数)，直接使用
                        shap_values_single = shap_values_single.ravel()[:len(feature_names)]

                    # 计算预测值
                    if isinstance(base_value, (list, np.ndarray)):
                        prediction_value = float(base_value[0]) + np.sum(shap_values_single)
                    else:
                        prediction_value = float(base_value) + np.sum(shap_values_single)

                    # 确保SHAP值是1D数组且长度匹配
                    if len(shap_values_single) != len(feature_names):
                        logger.warning(f"SHAP值数量 ({len(shap_values_single)}) 与特征名称数量 ({len(feature_names)}) 不匹配，使用实际特征数量")
                        # 截断或填充到匹配的长度
                        if len(shap_values_single) > len(feature_names):
                            shap_values_single = shap_values_single[:len(feature_names)]
                        else:
                            # 如果SHAP值较少，只使用对应的特征
                            feature_names = feature_names[:len(shap_values_single)]

                    abs_shap = np.abs(shap_values_single)

                    # 只显示实际存在的特征
                    num_features = len(feature_names)

                    # 按重要性排序
                    top_indices = np.argsort(abs_shap)[-num_features:]

                    # 直接使用，因为已经确保索引有效
                    top_shap = shap_values_single[top_indices]
                    top_features = [feature_names[idx] for idx in top_indices]

                    # 上半部分：特征贡献条形图（模拟力图的主体）
                    colors = []
                    for x in np.array(top_shap).flat:
                        if x > 0:
                            colors.append(self.color_scheme['primary'])
                        else:
                            colors.append(self.color_scheme['accent1'])

                    bars = ax1.barh(range(len(top_shap)), top_shap,
                                   color=colors, alpha=self.color_scheme['alpha'])

                    ax1.set_yticks(range(len(top_features)))
                    ax1.set_yticklabels(top_features, fontsize=10)
                    ax1.set_xlabel('SHAP Value (Feature Contribution)', fontsize=12, color=self.color_scheme['primary'])
                    ax1.set_title(f'SHAP Force Plot - Class {class_label} Sample {i+1}',
                                 fontsize=14, color=self.color_scheme['primary'], pad=20)

                    # 添加垂直线表示0点
                    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=2)

                    # 添加网格
                    ax1.grid(axis='x', alpha=0.3, color=self.color_scheme['secondary'])

                    # 设置x轴范围以更好地显示
                    max_val = np.max(np.abs(top_shap))
                    ax1.set_xlim(-max_val * 1.1, max_val * 1.1)

                    # 下半部分：基线值和预测值（模拟力图的底部区域）
                    ax2.axis('off')

                    # 创建基线和预测的视觉效果
                    try:
                        base_val = float(base_value[0]) if isinstance(base_value, (list, np.ndarray)) else float(base_value)
                        base_text = f'Base Value: {base_val:.3f}'
                    except:
                        base_text = f'Base Value: {base_value}'
                    pred_text = f'Output Value: {prediction_value:.3f}'

                    # 绘制基线值
                    ax2.text(0.1, 0.7, base_text, fontsize=12, color=self.color_scheme['primary'],
                            transform=ax2.transAxes, ha='left', va='center',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor=self.color_scheme['secondary'], alpha=0.3))

                    # 绘制预测值
                    ax2.text(0.9, 0.7, pred_text, fontsize=12, color=self.color_scheme['accent1'],
                            transform=ax2.transAxes, ha='right', va='center',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor=self.color_scheme['accent2'], alpha=0.3))

                    # 添加箭头表示从基线到预测的转换
                    ax2.annotate('', xy=(0.85, 0.7), xytext=(0.15, 0.7),
                                arrowprops=dict(arrowstyle='->', lw=2, color=self.color_scheme['primary']))

                    # 添加图例
                    from matplotlib.patches import Patch
                    legend_elements = [
                        Patch(facecolor=self.color_scheme['primary'], alpha=self.color_scheme['alpha'],
                             label='Positive Impact (Increases Prediction)'),
                        Patch(facecolor=self.color_scheme['accent1'], alpha=self.color_scheme['alpha'],
                             label='Negative Impact (Decreases Prediction)')
                    ]
                    ax1.legend(handles=legend_elements, loc='lower right', fontsize=10)

                    # 调整布局
                    plt.tight_layout()

                    # 保存图片
                    output_path = self.output_dir / f'forceplot_class_{class_label}_sample_{i+1}.png'
                    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                               facecolor='white', edgecolor='none')
                    plt.close()

                    logger.info(f"✓ 力图已保存: {output_path}")

                except Exception as e:
                    logger.error(f"生成力图失败 (类别 {class_label}, 样本 {i+1}): {e}")
                    import traceback
                    traceback.print_exc()
                    plt.close()
                    continue

    def plot_summary_plots(self, feature_names: list, class_labels: list):
        """生成摘要图 - 每个类别一张"""
        logger.info("="*60)
        logger.info("生成摘要图 (每个类别一张)")
        logger.info("="*60)

        if self.shap_values is None:
            raise ValueError("请先计算SHAP值")

        for class_label in class_labels:
            logger.info(f"生成类别 {class_label} 的摘要图...")

            try:
                plt.figure(figsize=(12, 8))

                # 对于多分类问题
                if isinstance(self.shap_values, list):
                    # 使用该类别的SHAP值生成摘要图
                    shap.summary_plot(
                        self.shap_values[class_label],
                        self.X_sample,
                        feature_names=feature_names,
                        show=False,
                        plot_size=(12, 8),
                        color=self.color_scheme['primary']
                    )
                else:
                    # 二分类问题
                    shap.summary_plot(
                        self.shap_values,
                        self.X_sample,
                        feature_names=feature_names,
                        show=False,
                        plot_size=(12, 8),
                        color=self.color_scheme['primary']
                    )

                # 自定义样式 - 使用英文标题避免字体问题
                plt.title(f'SHAP Summary Plot - Class {class_label}',
                         fontsize=16,
                         color=self.color_scheme['primary'],
                         pad=20)

                # 设置颜色透明度
                for collection in plt.gca().collections:
                    if hasattr(collection, 'set_alpha'):
                        collection.set_alpha(self.color_scheme['alpha'])

                # 保存图片
                output_path = self.output_dir / f'summary_class_{class_label}.png'
                plt.tight_layout()
                plt.savefig(output_path, dpi=300, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                plt.close()

                logger.info(f"✓ 已保存: {output_path}")

            except Exception as e:
                logger.error(f"生成摘要图失败 (类别 {class_label}): {e}")
                plt.close()
                continue

    def plot_overall_summary(self, feature_names: list):
        """生成总体摘要图"""
        logger.info("生成总体摘要图...")

        try:
            plt.figure(figsize=(14, 10))

            # 计算平均绝对SHAP值
            if isinstance(self.shap_values, list):
                # 多分类：合并所有类别的SHAP值
                # 计算每个特征的平均绝对SHAP值（跨所有样本和类别）
                all_shap_abs = [np.abs(shap_val) for shap_val in self.shap_values]
                # 在类别维度上平均，然后在样本维度上平均
                feature_importance = np.mean([np.mean(class_shap, axis=0) for class_shap in all_shap_abs], axis=0)
            else:
                # 二分类
                feature_importance = np.mean(np.abs(self.shap_values), axis=0)

            # 确保feature_importance是一维数组并与特征数量匹配
            if feature_importance.ndim > 1:
                feature_importance = feature_importance.ravel()

            # 确保长度匹配
            if len(feature_importance) != len(feature_names):
                logger.warning(f"Feature importance length ({len(feature_importance)}) != feature names length ({len(feature_names)})")
                # 调整长度
                min_len = min(len(feature_importance), len(feature_names))
                feature_importance = feature_importance[:min_len]
                feature_names = feature_names[:min_len]

            # 创建条形图
            feature_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=True)

            # 绘制水平条形图
            plt.barh(range(len(feature_df)), feature_df['importance'],
                    color=self.color_scheme['primary'], alpha=self.color_scheme['alpha'])

            plt.yticks(range(len(feature_df)), feature_df['feature'])
            plt.xlabel('Mean Absolute SHAP Value', fontsize=12)
            plt.ylabel('Features', fontsize=12)
            plt.title('Feature Importance Ranking (Based on SHAP Values)', fontsize=16,
                     color=self.color_scheme['primary'], pad=20)

            # 添加网格
            plt.grid(axis='x', alpha=0.3, color=self.color_scheme['secondary'])

            # 保存图片
            output_path = self.output_dir / 'overall_feature_importance.png'
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()

            logger.info(f"✓ Overall summary plot saved: {output_path}")

            # 保存特征重要性数据
            importance_csv_path = self.output_dir / 'feature_importance.csv'
            feature_df.to_csv(importance_csv_path, index=False)
            logger.info(f"✓ Feature importance data saved: {importance_csv_path}")

        except Exception as e:
            logger.error(f"Failed to generate overall summary plot: {e}")
            import traceback
            traceback.print_exc()
            plt.close()

    def run_analysis(self, model_name: str, csv_path: str, segments_per_sample: int = 124):
        """运行完整的SHAP分析"""
        logger.info("="*60)
        logger.info("开始SHAP可解释性分析")
        logger.info("="*60)

        try:
            # 1. 加载数据和模型
            X, feature_names = self.load_data_and_model(model_name, csv_path, segments_per_sample)

            # 2. 准备样本数据
            sample_indices = self.prepare_sample_data(X, segments_per_sample)

            # 3. 创建解释器并计算SHAP值
            self.create_explainer(self.X_sample, feature_names)

            # 4. 获取类别信息（基于预测结果）
            predictions = self.inference.predict_ensemble(self.X_sample)
            unique_classes = np.unique(predictions)
            logger.info(f"检测到的类别: {unique_classes}")

            # 5. 生成力图
            self.plot_force_plots(unique_classes, feature_names, num_samples_per_class=5)

            # 6. 生成摘要图
            self.plot_summary_plots(feature_names, unique_classes)

            # 7. 生成总体摘要图
            self.plot_overall_summary(feature_names)

            logger.info("="*60)
            logger.info("✓ SHAP分析完成！")
            logger.info(f"✓ 结果保存在: {self.output_dir}")
            logger.info("="*60)

        except Exception as e:
            logger.error(f"SHAP分析失败: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='SHAP可解释性分析工具 - 四分类模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 对XGBoost模型进行SHAP分析
  python 16_t_shap_explainability.py --model xgboost --data data/t_data_daan_aligned.csv --segments_per_sample 124

  # 使用默认参数
  python 16_t_shap_explainability.py --model xgboost --data data/t_data_daan_aligned.csv
        """
    )

    parser.add_argument('--model', type=str, required=True,
                       choices=['xgboost', 'random_forest', 'knn', 'svm', 'decision_tree',
                               'adaboost', 'extra_trees', 'gradient_boosting', 'bagging_ensemble'],
                       help='要分析的模型类型')

    parser.add_argument('--data', type=str, required=True,
                       help='输入数据文件路径 (如: data/t_data_daan_aligned.csv)')

    parser.add_argument('--segments_per_sample', type=int, default=124,
                       help='每个样本的分段数量 (默认: 124)')

    parser.add_argument('--output_dir', type=str, default='./out/SHAP',
                       help='输出目录路径 (默认: ./out/SHAP)')

    args = parser.parse_args()

    # 设置默认输出文件名
    logger.info("="*60)
    logger.info("SHAP可解释性分析工具")
    logger.info("="*60)
    logger.info(f"模型类型: {args.model}")
    logger.info(f"输入数据: {args.data}")
    logger.info(f"分段数量: {args.segments_per_sample}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info("="*60)

    try:
        # 创建配置
        config = MLConfig()

        # 初始化SHAP分析器
        shap_analyzer = SHAPExplainability(config)

        # 修改输出目录（如果指定）
        if args.output_dir != './out/SHAP':
            shap_analyzer.output_dir = Path(args.output_dir)
            shap_analyzer.output_dir.mkdir(parents=True, exist_ok=True)

        # 运行分析
        shap_analyzer.run_analysis(
            model_name=args.model,
            csv_path=args.data,
            segments_per_sample=args.segments_per_sample
        )

        logger.info("\n🎉 分析完成！生成的文件包括:")
        logger.info("- 力图: forceplot_class_*_sample_*.png")
        logger.info("- 摘要图: summary_class_*.png")
        logger.info("- 总体特征重要性: overall_feature_importance.png")
        logger.info("- 特征重要性数据: feature_importance.csv")

    except FileNotFoundError as e:
        logger.error(f"文件未找到: {e}")
        logger.error("请确保模型已训练并保存在 ./models/ 目录下")
        sys.exit(1)
    except Exception as e:
        logger.error(f"分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()