# 轴承故障分析工具包

## 脚本功能说明

### `signal_visualization.py`
时域信号可视化工具 - 绘制轴承传感器数据的时域波形图，支持批量处理和自定义样式

### `data_format_converter.py`
数据格式转换器 - 将MATLAB的.mat文件批量转换为CSV格式，保持原有目录结构

### `bearing_fault_detector.py`
轴承故障检测器 - 基于包络分析和COR指数的特征提取，计算BPFO/BPFI/BSF/FTF等故障特征频率

### `cor_analysis_summary.py`
COR分析汇总工具 - 批量分析COR指数模式，统计故障特征并输出分析结果

## 使用说明
所有脚本均支持独立运行，可根据需要调用相应的功能模块。