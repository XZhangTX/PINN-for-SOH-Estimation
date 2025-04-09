import numpy as np
import h5py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import matplotlib.pyplot as plt
import argparse

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class BatteryDataset(Dataset):
    def __init__(self, X, y):
        """
        初始化数据集
        Args:
            X: 特征序列数据，形状为 (n_samples, sequence_length, n_features)
            y: SOH标签，形状为 (n_samples,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        返回一个完整的序列及其对应的SOH标签
        Returns:
            sequence: 当前序列的特征数据，形状为 (sequence_length, n_features)
            soh: 当前序列对应的SOH值
        """
        return self.X[idx], self.y[idx]


def load_mat_data(file_path):
    """加载MAT文件数据"""
    print(f"正在加载数据文件: {file_path}")
    bat_dict = {}

    with h5py.File(file_path, 'r') as f:
        batch = f.get("batch")
        num_cells = batch['summary'].shape[0]

        for i in range(num_cells):
            # 获取循环寿命
            cl = f[batch['cycle_life'][i, 0]][()]

            # 获取充电策略
            policy = f[batch['policy_readable'][i, 0]][()].tobytes()[::2].decode()

            # 获取summary数据
            summary_IR = np.hstack(f[batch['summary'][i, 0]]['IR'][0, :].tolist())
            summary_QC = np.hstack(f[batch['summary'][i, 0]]['QCharge'][0, :].tolist())
            summary_QD = np.hstack(f[batch['summary'][i, 0]]['QDischarge'][0, :].tolist())
            summary_TA = np.hstack(f[batch['summary'][i, 0]]['Tavg'][0, :].tolist())
            summary_TM = np.hstack(f[batch['summary'][i, 0]]['Tmin'][0, :].tolist())
            summary_TX = np.hstack(f[batch['summary'][i, 0]]['Tmax'][0, :].tolist())
            summary_CT = np.hstack(f[batch['summary'][i, 0]]['chargetime'][0, :].tolist())
            summary_CY = np.hstack(f[batch['summary'][i, 0]]['cycle'][0, :].tolist())

            summary = {
                'IR': summary_IR,
                'QC': summary_QC,
                'QD': summary_QD,
                'Tavg': summary_TA,
                'Tmin': summary_TM,
                'Tmax': summary_TX,
                'chargetime': summary_CT,
                'cycle': summary_CY
            }

            # 获取循环数据
            cycles = f[batch['cycles'][i, 0]]
            cycle_dict = {}

            for j in range(cycles['I'].shape[0]):
                I = np.hstack((f[cycles['I'][j, 0]][()]))
                Qc = np.hstack((f[cycles['Qc'][j, 0]][()]))
                Qd = np.hstack((f[cycles['Qd'][j, 0]][()]))
                Qdlin = np.hstack((f[cycles['Qdlin'][j, 0]][()]))
                T = np.hstack((f[cycles['T'][j, 0]][()]))
                Tdlin = np.hstack((f[cycles['Tdlin'][j, 0]][()]))
                V = np.hstack((f[cycles['V'][j, 0]][()]))
                dQdV = np.hstack((f[cycles['discharge_dQdV'][j, 0]][()]))
                t = np.hstack((f[cycles['t'][j, 0]][()]))

                cd = {
                    'I': I,
                    'Qc': Qc,
                    'Qd': Qd,
                    'Qdlin': Qdlin,
                    'T': T,
                    'Tdlin': Tdlin,
                    'V': V,
                    'dQdV': dQdV,
                    't': t
                }
                cycle_dict[str(j)] = cd

            cell_dict = {
                'cycle_life': cl,
                'charge_policy': policy,
                'summary': summary,
                'cycles': cycle_dict
            }

            key = f"b1c{i}"  # 使用固定的批次号1
            bat_dict[key] = cell_dict

    print("数据加载完成")
    return bat_dict


def preprocess_data(bat_dict, time_step=10):
    """
    预处理数据，提取特征和标签
    Args:
        bat_dict: 电池数据字典
        time_step: 采样间隔，每隔多少个时间点取一个样本
    """
    print("开始预处理数据...")

    features_list = []
    labels_list = []
    cycle_indices = []  # 记录循环索引

    for cell_key, cell_data in bat_dict.items():
        print(f"\n开始处理电池 {cell_key}...")
        # 获取summary数据中的放电容量
        QD = cell_data['summary']['QD']  # 放电容量
        cycles = cell_data['cycles']

        # 找到第一个非零容量作为初始容量
        initial_capacity = None
        for q in QD:
            if q > 0:
                initial_capacity = q
                break

        if initial_capacity is None or initial_capacity == 0:
            print(f"警告: 电池 {cell_key} 没有有效的初始容量，跳过")
            continue

        # 计算SOH
        soh = QD / initial_capacity  # SOH = 当前容量/初始容量

        # 获取每个循环的数据
        for cycle_key, cycle_data in cycles.items():
            cycle_idx = int(cycle_key)
            if cycle_idx >= len(soh):
                continue

            # 提取时间序列数据
            voltage = cycle_data['V']
            current = cycle_data['I']
            temperature = cycle_data['T']
            time = cycle_data['t']

            # 确保所有特征长度一致
            min_length = min(len(voltage), len(current), len(temperature), len(time))
            voltage = voltage[:min_length]
            current = current[:min_length]
            temperature = temperature[:min_length]
            time = time[:min_length]

            # 按时间步长采样
            for i in range(0, min_length, time_step):
                features = np.array([
                    voltage[i],
                    current[i],
                    temperature[i],
                    time[i],  # 循环内的相对时间
                    cycle_idx,  # 添加循环次数作为特征
                ])
                features_list.append(features)
                labels_list.append(soh[cycle_idx])
                cycle_indices.append(cycle_idx)

            # 打印每个循环的处理进度
            if cycle_idx % 50 == 0:
                print(f"  电池 {cell_key} 的第 {cycle_idx} 次循环，当前SOH: {soh[cycle_idx]:.4f}")

        print(f"电池 {cell_key} 处理完成，总循环数: {len(cycles)}")

    # 合并所有特征
    features = np.array(features_list)
    cycle_indices = np.array(cycle_indices)

    # 归一化处理，但保持循环次数的相对关系
    scaler = MinMaxScaler()
    # 分别对电压、电流、温度、时间进行归一化
    features[:, :4] = scaler.fit_transform(features[:, :4])
    # 对循环次数单独归一化
    features[:, 4] = features[:, 4] / features[:, 4].max()

    # 创建标签数组
    labels = np.array(labels_list)

    print(f"\n数据预处理完成:")
    print(f"特征形状: {features.shape}, 标签形状: {labels.shape}")
    print(f"特征维度: [电压, 电流, 温度, 时间, 循环次数]")
    print(f"总电池数: {len(bat_dict)}")
    print(f"总循环数: {len(np.unique(cycle_indices))}")
    print(f"每个循环的平均采样点数: {len(features) / len(np.unique(cycle_indices)):.2f}")
    
    # 保存统计信息
    stats = {
        'cycle_indices': cycle_indices,
        'unique_cycles': np.unique(cycle_indices),
        'samples_per_cycle': np.bincount(cycle_indices),
        'total_samples': len(features),
        'feature_names': ['电压', '电流', '温度', '时间', '循环次数'],
        'battery_ids': list(bat_dict.keys())
    }
    
    return features, labels, scaler, stats


def create_sequences(features, labels, sequence_length, cycle_indices):
    """
    创建时间序列数据，确保序列只在同一个循环内创建
    Args:
        features: 特征数据，形状为 (n_samples, n_features)
        labels: 标签数据，形状为 (n_samples,)
        sequence_length: 序列长度
        cycle_indices: 每个样本对应的循环索引
    """
    print(f"创建序列数据，序列长度: {sequence_length}")
    X, y = [], []

    # 获取唯一的循环索引
    unique_cycles = np.unique(cycle_indices)
    
    # 对每个循环单独处理
    for cycle in unique_cycles:
        # 获取当前循环的数据
        cycle_mask = cycle_indices == cycle
        cycle_features = features[cycle_mask]
        cycle_labels = labels[cycle_mask]
        
        # 如果当前循环的数据长度不足序列长度，跳过
        if len(cycle_features) <= sequence_length:
            continue
            
        # 在当前循环内创建序列
        for i in range(len(cycle_features) - sequence_length):
            # 获取序列数据
            seq_features = cycle_features[i:(i + sequence_length)]
            X.append(seq_features)
            y.append(cycle_labels[i + sequence_length])

        if cycle % 50 == 0:
            print(f"处理第 {cycle} 个循环，生成序列数: {len(X)}")

    X = np.array(X)
    y = np.array(y)

    print(f"\n序列数据统计:")
    print(f"输入形状 X: {X.shape} (样本数, 序列长度, 特征数)")
    print(f"输出形状 y: {y.shape} (样本数,)")
    print(f"特征维度: [电压, 电流, 温度, 时间, 循环次数]")
    
    return X, y


def visualize_data(features, labels, stats=None):
    """可视化预处理后的数据"""
    # 创建图形
    plt.figure(figsize=(15, 10))
    
    # 1. 绘制特征分布
    plt.subplot(2, 2, 1)
    feature_names = ['电压', '电流', '温度', '时间', '循环次数']
    for i in range(5):
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
    if stats is not None:
        # 只显示第一个循环的数据
        cycle_mask = stats['cycle_indices'] == 0
        cycle_features = features[cycle_mask]
        for i in range(5):
            plt.plot(range(len(cycle_features)), cycle_features[:, i], 
                    label=feature_names[i], alpha=0.5)
        plt.title('第一个循环的特征变化')
    else:
        for i in range(5):
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
    plt.xticks(range(5), feature_names)
    plt.yticks(range(5), feature_names)
    plt.title('特征相关性')
    
    plt.tight_layout()
    plt.savefig('preprocessed_data_visualization.png')
    plt.show()
    
    # 打印数据统计信息
    print("\n数据统计信息:")
    print(f"样本总数: {len(features)}")
    if stats is not None:
        print(f"循环总数: {len(stats['unique_cycles'])}")
        print(f"平均每个循环的样本数: {stats['total_samples']/len(stats['unique_cycles']):.2f}")
    
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


def main(data_path='data/MITdata/2017-05-12.mat', time_step=30, sequence_length=40):
    """
    主函数
    Args:
        data_path: 数据文件路径
        time_step: 采样间隔，每隔多少个时间点取一个样本
        sequence_length: 序列长度，每个序列包含多少个时间步
    """
    # 检查文件是否存在
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    # 加载数据
    bat_dict = load_mat_data(data_path)

    # 预处理数据
    print(f"\n使用采样间隔: {time_step}")
    features, labels, scaler, stats = preprocess_data(bat_dict, time_step=time_step)

    # 可视化预处理后的数据
    visualize_data(features, labels, stats)

    # 创建序列数据，确保序列在同一个循环内
    print(f"\n使用序列长度: {sequence_length}")
    X, y = create_sequences(features, labels, sequence_length, stats['cycle_indices'])

    # 首先划分训练集和测试集
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 从训练集中划分出验证集
    valid_size = int(0.2 * len(X_train))
    X_train, X_valid = X_train[:-valid_size], X_train[-valid_size:]
    y_train, y_valid = y_train[:-valid_size], y_train[-valid_size:]

    print(f"\n数据集划分:")
    print(f"训练集大小: {len(X_train)}")
    print(f"验证集大小: {len(X_valid)}")
    print(f"测试集大小: {len(X_test)}")
    print(f"序列形状: {X_train.shape} (样本数, 序列长度, 特征数)")
    print(f"特征维度: [电压, 电流, 温度, 时间, 循环次数]")

    # 创建数据集
    train_dataset = BatteryDataset(X_train, y_train)
    valid_dataset = BatteryDataset(X_valid, y_valid)
    test_dataset = BatteryDataset(X_test, y_test)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 保存一些额外的信息，可能对模型训练有用
    metadata = {
        'scaler': scaler,
        'stats': stats,
        'feature_names': ['电压', '电流', '温度', '时间', '循环次数'],
        'sequence_length': sequence_length,
        'time_step': time_step
    }

    return train_loader, valid_loader, test_loader, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='电池数据预处理')
    parser.add_argument('--data_path', type=str, default='data/MITdata/2017-05-12.mat',
                      help='数据文件路径')
    parser.add_argument('--time_step', type=int, default=20,
                      help='采样间隔，每隔多少个时间点取一个样本')
    parser.add_argument('--sequence_length', type=int, default=40,
                      help='序列长度，每个序列包含多少个时间步')
    
    args = parser.parse_args()
    
    train_loader, valid_loader, test_loader, metadata = main(
        data_path=args.data_path,
        time_step=args.time_step,
        sequence_length=args.sequence_length
    ) 