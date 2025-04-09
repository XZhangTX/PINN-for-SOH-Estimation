import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from data_preprocessing import load_mat_data, preprocess_data, create_sequences

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def visualize_preprocessed_data(data_file='data/MITdata/2017-05-12.mat'):
    """可视化data_preprocessing.py处理后的数据"""
    # 加载和预处理数据
    data = load_mat_data(data_file)
    features, labels = preprocess_data(data)
    
    # 创建图形
    plt.figure(figsize=(15, 10))
    
    # 1. 绘制特征分布
    plt.subplot(2, 2, 1)
    feature_names = ['电压', '电流', '温度', '时间']
    for i in range(4):
        plt.hist(features[:, i], bins=30, alpha=0.5, label=feature_names[i])
    plt.title('特征分布')
    plt.xlabel('特征值')
    plt.ylabel('频数')
    plt.legend()
    
    # 2. 绘制SOH标签分布
    plt.subplot(2, 2, 2)
    plt.hist(labels, bins=30, alpha=0.7)
    plt.title('SOH标签分布')
    plt.xlabel('SOH')
    plt.ylabel('频数')
    
    # 3. 绘制特征随样本索引的变化
    plt.subplot(2, 2, 3)
    for i in range(4):
        plt.plot(range(len(features)), features[:, i], 
                label=feature_names[i], alpha=0.5)
    plt.title('特征随样本索引的变化')
    plt.xlabel('样本索引')
    plt.ylabel('特征值')
    plt.legend()
    
    # 4. 绘制特征相关性热力图
    plt.subplot(2, 2, 4)
    correlation = np.corrcoef(features.T)
    plt.imshow(correlation, cmap='coolwarm')
    plt.colorbar()
    plt.xticks(range(4), feature_names)
    plt.yticks(range(4), feature_names)
    plt.title('特征相关性')
    
    plt.tight_layout()
    plt.savefig('preprocessed_data_visualization.png')
    plt.show()
    
    # 打印数据统计信息
    print("\n数据统计信息:")
    print(f"样本总数: {len(features)}")
    print("\n特征统计:")
    for i, name in enumerate(feature_names):
        print(f"\n{name}:")
        print(f"  最小值: {features[:, i].min():.4f}")
        print(f"  最大值: {features[:, i].max():.4f}")
        print(f"  平均值: {features[:, i].mean():.4f}")
        print(f"  标准差: {features[:, i].std():.4f}")
    
    print("\nSOH标签统计:")
    print(f"  最小值: {labels.min():.4f}")
    print(f"  最大值: {labels.max():.4f}")
    print(f"  平均值: {labels.mean():.4f}")
    print(f"  标准差: {labels.std():.4f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='data/MITdata/2017-05-12.mat',
                        help='数据文件路径')
    args = parser.parse_args()
    
    # 可视化预处理后的数据
    visualize_preprocessed_data(args.data_file) 