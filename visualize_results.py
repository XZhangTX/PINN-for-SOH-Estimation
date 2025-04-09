import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from battery_pinn import BatteryPINN, BatteryMLP
import argparse
from data_preprocessing import load_mat_data, preprocess_data, create_sequences

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def visualize_processed_data(data_file='data/MITdata/2017-05-12.mat'):
    """可视化预处理后的数据"""
    # 加载原始数据
    data = load_mat_data(data_file)
    
    # 创建图形
    plt.figure(figsize=(15, 10))
    
    # 1. 绘制电压、电流、温度随时间的变化
    plt.subplot(2, 2, 1)
    for cell_idx in range(min(3, len(data))):  # 只显示前3个电池的数据
        cell_data = data[cell_idx]
        cycles = cell_data['cycles']
        voltage = [cycle['V'] for cycle in cycles]
        time = list(range(len(voltage)))
        plt.plot(time, voltage, label=f'电池 {cell_idx+1}')
    plt.title('电压随时间变化')
    plt.xlabel('循环次数')
    plt.ylabel('电压 (V)')
    plt.legend()
    
    # 2. 绘制SOH随循环次数的变化
    plt.subplot(2, 2, 2)
    for cell_idx in range(min(3, len(data))):
        cell_data = data[cell_idx]
        cycles = cell_data['cycles']
        capacity = [cycle['QD'] for cycle in cycles]
        initial_capacity = capacity[0]
        soh = [cap/initial_capacity for cap in capacity]
        time = list(range(len(soh)))
        plt.plot(time, soh, label=f'电池 {cell_idx+1}')
    plt.title('SOH随循环次数变化')
    plt.xlabel('循环次数')
    plt.ylabel('SOH')
    plt.legend()
    
    # 3. 绘制容量随循环次数的变化
    plt.subplot(2, 2, 3)
    for cell_idx in range(min(3, len(data))):
        cell_data = data[cell_idx]
        cycles = cell_data['cycles']
        capacity = [cycle['QD'] for cycle in cycles]
        time = list(range(len(capacity)))
        plt.plot(time, capacity, label=f'电池 {cell_idx+1}')
    plt.title('容量随循环次数变化')
    plt.xlabel('循环次数')
    plt.ylabel('容量 (Ah)')
    plt.legend()
    
    # 4. 绘制温度随时间的变化
    plt.subplot(2, 2, 4)
    for cell_idx in range(min(3, len(data))):
        cell_data = data[cell_idx]
        cycles = cell_data['cycles']
        temp = [cycle['T'] for cycle in cycles]
        time = list(range(len(temp)))
        plt.plot(time, temp, label=f'电池 {cell_idx+1}')
    plt.title('温度随循环次数变化')
    plt.xlabel('循环次数')
    plt.ylabel('温度 (°C)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('processed_data_visualization.png')
    plt.show()

def load_model(args):
    # 加载模型
    model = BatteryPINN(args)
    model_path = os.path.join(args.save_folder, 'model.pth')
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.solution_u.load_state_dict(checkpoint['solution_u'])
        model.dynamical_F.load_state_dict(checkpoint['dynamical_F'])
        print("模型加载成功")
    else:
        print("未找到模型文件")
        return None
    return model

def visualize_predictions(args):
    # 加载真实标签和预测标签
    true_label_path = os.path.join(args.save_folder, 'true_label.npy')
    pred_label_path = os.path.join(args.save_folder, 'pred_label.npy')
    
    if not (os.path.exists(true_label_path) and os.path.exists(pred_label_path)):
        print("未找到标签文件")
        return
        
    true_label = np.load(true_label_path)
    pred_label = np.load(pred_label_path)
    
    # 创建图形
    plt.figure(figsize=(12, 6))
    
    # 绘制真实值和预测值
    plt.subplot(1, 2, 1)
    plt.scatter(range(len(true_label)), true_label, label='真实值', alpha=0.6)
    plt.scatter(range(len(pred_label)), pred_label, label='预测值', alpha=0.6)
    plt.title('SOH预测结果')
    plt.xlabel('样本索引')
    plt.ylabel('SOH')
    plt.legend()
    
    # 绘制误差分布
    plt.subplot(1, 2, 2)
    errors = np.abs(true_label - pred_label)
    plt.hist(errors, bins=30, alpha=0.7)
    plt.title('预测误差分布')
    plt.xlabel('绝对误差')
    plt.ylabel('频数')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_folder, 'prediction_results.png'))
    plt.show()

def visualize_training_log(args):
    # 读取训练日志
    log_path = os.path.join(args.save_folder, 'logging.txt')
    if not os.path.exists(log_path):
        print("未找到日志文件")
        return
        
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    # 提取训练损失
    train_losses = []
    valid_mses = []
    for line in lines:
        if '[Train]' in line:
            loss = float(line.split('total loss:')[-1].strip())
            train_losses.append(loss)
        elif '[Valid]' in line:
            mse = float(line.split('MSE:')[-1].strip())
            valid_mses.append(mse)
    
    # 创建图形
    plt.figure(figsize=(12, 6))
    
    # 绘制训练损失
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.title('训练损失变化')
    plt.xlabel('Epoch')
    plt.ylabel('损失值')
    plt.legend()
    
    # 绘制验证MSE
    plt.subplot(1, 2, 2)
    plt.plot(valid_mses, label='验证MSE')
    plt.title('验证MSE变化')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_folder, 'training_curves.png'))
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_folder', type=str, default='results/battery_results',
                        help='保存结果的文件夹路径')
    parser.add_argument('--data_file', type=str, default='data/MITdata/2017-05-12.mat',
                        help='数据文件路径')
    args = parser.parse_args()
    
    # 可视化预处理后的数据
    visualize_processed_data(args.data_file)
    
    # 可视化预测结果
    visualize_predictions(args)
    
    # 可视化训练过程
    visualize_training_log(args) 