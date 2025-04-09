import argparse
import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_args():
    parser = argparse.ArgumentParser(description='电池SOH预测训练')

    # 数据相关
    parser.add_argument('--data_path', type=str, default='data/MITdata/2017-05-12.mat', help='数据文件路径')
    parser.add_argument('--time_step', type=int, default=20, help='采样间隔')
    parser.add_argument('--sequence_length', type=int, default=40, help='序列长度')
    parser.add_argument('--batch_size', type=int, default=256, help='批量大小')

    # 训练相关
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--log_interval', type=int, default=10, help='日志打印间隔')
    parser.add_argument('--save_folder', type=str, default=r'G:\BS\results\battery_results', help='保存结果的目录')

    # 模型相关
    parser.add_argument('--hidden_dim', type=int, default=64, help='LSTM隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2, help='LSTM层数')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout比率')

    # 损失权重
    parser.add_argument('--physics_weight', type=float, default=0.1, help='物理约束损失权重')

    return parser.parse_args()


def train(args):
    # 创建保存目录
    os.makedirs(args.save_folder, exist_ok=True)
    log_file = os.path.join(args.save_folder, 'training_log.txt')

    def log(message):
        """记录日志到文件并打印到控制台"""
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
        print(message)

    # 加载数据
    from data_preprocessing import main as load_data  # 假设数据加载函数在 data_preprocessing.py 中
    train_loader, valid_loader, test_loader, metadata = load_data(
        data_path=args.data_path,
        time_step=args.time_step,
        sequence_length=args.sequence_length
        # Remove batch_size parameter
    )

    # 创建模型
    from battery_pinn import BatteryPINN  # 假设模型定义在 battery_pinn.py 中
    model = BatteryPINN(
        sequence_length=args.sequence_length,
        n_features=5,  # [电压, 电流, 温度, 时间, 循环次数]
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # 训练循环
    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # 前向传播
            y_pred = model(x)

            # 计算损失
            mse_loss = criterion(y_pred, y)
            physics_loss = model.physics_loss(x, y_pred)
            total_loss = mse_loss + args.physics_weight * physics_loss

            # 反向传播和优化
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()

            if batch_idx % args.log_interval == 0:
                log(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                    f'Loss: {total_loss.item():.6f} '
                    f'MSE: {mse_loss.item():.6f} '
                    f'Physics: {physics_loss.item():.6f}')

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                val_loss += criterion(y_pred, y).item()

        val_loss /= len(valid_loader)
        log(f'Epoch {epoch} Validation Loss: {val_loss:.6f}')

        # 学习率调整
        scheduler.step(val_loss)

        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            # 保存最佳模型
            os.makedirs(args.save_folder, exist_ok=True)
            save_path = os.path.join(args.save_folder, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'metadata': metadata
            }, save_path)
            log(f"New best model saved at epoch {epoch} with loss {val_loss:.6f}")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                log(f'Early stopping after {epoch} epochs')
                break

    # 测试
    model.eval()
    test_loss = 0
    predictions = []
    true_values = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            test_loss += criterion(y_pred, y).item()
            predictions.extend(y_pred.cpu().numpy())
            true_values.extend(y.cpu().numpy())

    test_loss /= len(test_loader)
    log(f'Test Loss: {test_loss:.6f}')

    # 计算评估指标
    mae = mean_absolute_error(true_values, predictions)
    rmse = mean_squared_error(true_values, predictions, squared=False)
    r2 = r2_score(true_values, predictions)
    log(f'Test Metrics - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}')

    # 绘制预测结果
    plt.figure(figsize=(10, 6))
    plt.plot(true_values, label='真实值')
    plt.plot(predictions, label='预测值')
    plt.xlabel('样本索引')
    plt.ylabel('SOH')
    plt.title('SOH预测结果对比')
    plt.legend()
    plt.savefig(os.path.join(args.save_folder, 'prediction_results.png'))
    plt.close()

    # 绘制误差分布图
    errors = [true - pred for true, pred in zip(true_values, predictions)]
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, color='blue', alpha=0.7)
    plt.title('Error Distribution')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(args.save_folder, 'error_distribution.png'))
    plt.close()


if __name__ == '__main__':
    args = get_args()
    train(args)