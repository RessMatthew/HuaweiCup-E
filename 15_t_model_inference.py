#!/usr/bin/env python3
"""
模型推理脚本
用于加载已训练好的模型并对新数据进行预测

自动在 ./models/{model_name}/ 路径下找到最新的模型文件。
加载所有交叉验证折叠模型，使用投票进行最终预测。
支持分类和概率预测。

使用方法:
python 15_model_inference.py --model xgboost --data data/t_data_out.csv --segment_voting --segments_per_sample 124
"""

import sys
import os

# 添加ml目录到路径
sys.path.append('ml')

from model_inference import ModelInference
from ml_config import MLConfig
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """主函数 - 模型推理接口"""
    import argparse

    parser = argparse.ArgumentParser(
        description='模型推理工具 - 加载已训练的模型进行预测',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用XGBoost模型进行推理
  python 11_model_inference.py --model xgboost --data data/data_out.csv --output xgboost_predictions.csv

  # 使用随机森林模型
  python 11_model_inference.py --model random_forest --data data/data_out.csv

  # 指定特征列
  python 11_model_inference.py --model xgboost --data data/data_out.csv --features COR_BPFO COR_BPFI COR_BSF

  # 使用指定数量的模型（比如只用3折交叉验证中的模型）
  python 11_model_inference.py --model xgboost --data data/data_out.csv --num_folds 3

  # 不计算准确率（仅预测）
  python 11_model_inference.py --model xgboost --data data/data_out.csv --no_evaluate

  # 使用分段投票模式预测无标签数据（如t_data_out.csv）
  python 11_model_inference.py --model xgboost --data data/t_data_out.csv --segment_voting --show_summary

  # 指定每个样本的分段数量（默认124）
  python 11_model_inference.py --model xgboost --data data/t_data_out.csv --segment_voting --segments_per_sample 124
        """
    )

    parser.add_argument('--model', type=str, required=True,
                       choices=['xgboost', 'random_forest', 'knn', 'svm', 'decision_tree',
                               'adaboost', 'extra_trees', 'gradient_boosting', 'bagging_ensemble'],
                       help='要使用的模型类型')

    parser.add_argument('--data', type=str, default='data/data_out.csv',
                       help='输入数据文件路径 (默认: data/data_out.csv)')

    parser.add_argument('--output', type=str, default=None,
                       help='输出文件路径 (默认: {model}_predictions.csv)')

    parser.add_argument('--features', type=str, nargs='+', default=None,
                       help='要使用的特征列名 (如果不指定，将使用所有列)')

    parser.add_argument('--voting', type=str, default='majority',
                       choices=['majority', 'average'],
                       help='集成模型的投票策略 (默认: majority)')

    parser.add_argument('--num_folds', type=int, default=None,
                       help='要使用的模型折数 (默认: 使用所有可用的模型)')

    parser.add_argument('--show_summary', action='store_true',
                       help='显示预测结果汇总')

    parser.add_argument('--evaluate', action='store_true', default=True,
                       help='计算并显示预测准确率 (默认: True)')

    parser.add_argument('--no_evaluate', dest='evaluate', action='store_false',
                       help='不计算预测准确率')

    parser.add_argument('--segment_voting', action='store_true',
                       help='使用分段投票模式（用于t_data_out.csv等无标签数据）')

    parser.add_argument('--segments_per_sample', type=int, default=124,
                       help='每个样本的分段数量（默认: 124）')

    args = parser.parse_args()

    # 设置默认输出文件名
    if args.output is None:
        args.output = f"{args.model}_predictions.csv"

    logger.info("="*60)
    logger.info("模型推理工具")
    logger.info("="*60)
    logger.info(f"模型类型: {args.model}")
    logger.info(f"输入数据: {args.data}")
    logger.info(f"输出文件: {args.output}")
    if args.features:
        logger.info(f"使用特征: {len(args.features)} 个特征")
    if args.segment_voting:
        logger.info(f"分段投票模式: 每个样本 {args.segments_per_sample} 个分段")
    logger.info("="*60)

    try:
        # 创建配置
        config = MLConfig()

        # 初始化推理工具
        inference = ModelInference(config)

        # 加载模型
        logger.info(f"正在加载 {args.model} 模型...")
        inference.load_models(args.model, num_folds=args.num_folds)
        logger.info(f"✓ 模型加载完成")

        # 执行推理
        logger.info("开始推理...")

        if args.segment_voting:
            # 使用分段投票模式
            results = inference.inference_with_segment_voting(
                model_name=args.model,
                csv_path=args.data,
                feature_columns=args.features,
                voting=args.voting,
                segments_per_sample=args.segments_per_sample
            )
        else:
            # 使用普通推理模式
            results = inference.inference_from_csv(
                model_name=args.model,
                csv_path=args.data,
                feature_columns=args.features,
                voting=args.voting,
                evaluate=args.evaluate
            )

        # 保存结果
        results.to_csv(args.output, index=False)
        logger.info(f"✓ 推理结果已保存到: {args.output}")
        logger.info(f"结果数据形状: {results.shape}")

        # 显示汇总信息
        if args.show_summary and 'predicted_label' in results.columns:
            logger.info("\n预测结果汇总:")
            if args.segment_voting:
                # 分段投票模式的汇总
                pred_summary = results.groupby('sample_id')['predicted_label'].first().value_counts()
                for label, count in pred_summary.items():
                    percentage = (count / len(results['sample_id'].unique())) * 100
                    logger.info(f"  类别 {label}: {count} 个样本 ({percentage:.1f}%)")
            else:
                # 普通模式的汇总
                pred_summary = results['predicted_label'].value_counts()
                for label, count in pred_summary.items():
                    percentage = (count / len(results)) * 100
                    logger.info(f"  类别 {label}: {count} 个样本 ({percentage:.1f}%)")

        # 显示准确率信息（如果已计算）
        if args.evaluate and 'accuracy_score' in results.columns and not args.segment_voting:
            overall_accuracy = results['accuracy_score'].iloc[0]  # 所有行都有相同的准确率值
            logger.info(f"\n📊 整体准确率: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")

        # 显示前几条结果
        if len(results) > 0:
            logger.info("\n前5条预测结果:")
            if args.segment_voting:
                # 显示分段投票结果的前几个样本（每个样本显示一行汇总）
                sample_summary = results.groupby('sample_id').agg({
                    'predicted_label': 'first',
                    'total_segments': 'first'
                }).reset_index().head()
                print(sample_summary.to_string())

                # 显示每个样本的详细投票分布
                logger.info("\n详细投票分布（前3个样本）:")
                for sample_id in sorted(results['sample_id'].unique())[:3]:
                    sample_votes = results[results['sample_id'] == sample_id]
                    logger.info(f"样本 {sample_id}: 预测标签 {sample_votes['predicted_label'].iloc[0]}")
                    for _, row in sample_votes.iterrows():
                        logger.info(f"  类别 {row['class_label']}: {row['vote_count']} 票 ({row['vote_percentage']:.1f}%)")
            else:
                display_cols = ['predicted_label']
                if args.features and len(args.features) <= 3:
                    display_cols = args.features + display_cols
                print(results[display_cols].head().to_string())

        logger.info("="*60)
        logger.info("推理完成！")
        logger.info("="*60)

    except FileNotFoundError as e:
        logger.error(f"文件未找到: {e}")
        logger.error("请确保模型已训练并保存在 ./models/ 目录下")
        sys.exit(1)
    except Exception as e:
        logger.error(f"推理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()