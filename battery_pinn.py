import torch
import torch.nn as nn
import numpy as np
from torch.autograd import grad

from model import LR_Scheduler
from utils.util import AverageMeter, get_logger, eval_metrix
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class BatteryMLP(nn.Module):
    def __init__(self, input_dim=40, output_dim=1, layers_num=4, hidden_dim=50, dropout=0.2):
        super(BatteryMLP, self).__init__()
        assert layers_num >= 2, "layers must be greater than 2"

        self.layers = []
        for i in range(layers_num):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, hidden_dim))
                self.layers.append(Sin())
            elif i == layers_num - 1:
                self.layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.layers.append(Sin())
                self.layers.append(nn.Dropout(p=dropout))

        self.net = nn.Sequential(*self.layers)
        self._init()

    def _init(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        # 确保输入维度正确
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        return self.net(x)


class BatteryPINN(nn.Module):
    def __init__(self, sequence_length=40, n_features=5, hidden_dim=64, num_layers=2, dropout=0.1):
        """
        初始化电池PINN模型
        Args:
            sequence_length: 输入序列的长度
            n_features: 特征数量（电压、电流、温度、时间、循环次数）
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            dropout: dropout比率
        """
        super(BatteryPINN, self).__init__()
        
        self.sequence_length = sequence_length
        self.n_features = n_features
        
        # LSTM层处理时序数据
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True  # 输入形状为 (batch, seq_len, features)
        )
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入数据，形状为 (batch_size, sequence_length, n_features)
        Returns:
            soh_pred: 预测的SOH值
        """
        # LSTM处理序列数据
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, sequence_length, hidden_dim)
        
        # 只使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # 全连接层
        x = self.activation(self.fc1(last_output))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        soh_pred = self.fc3(x)
        return soh_pred.squeeze(-1)  # Convert from [batch_size, 1] to [batch_size]
        
    def physics_loss(self, x, y_pred):
        """
        计算物理约束损失
        Args:
            x: 输入数据，形状为 (batch_size, sequence_length, n_features)
            y_pred: 模型预测的SOH值
        """
        # 1. SOH单调递减约束
        cycle_numbers = x[:, :, 4]  # 获取循环次数特征
        batch_size = x.shape[0]
        physics_loss = 0.0
        
        # 对于每个批次中的样本
        for i in range(batch_size):
            # 找到不同的循环次数
            unique_cycles = torch.unique(cycle_numbers[i])
            if len(unique_cycles) > 1:
                # 计算SOH的差分
                cycle_diff = torch.diff(unique_cycles)
                # 添加单调递减约束
                monotonic_loss = torch.relu(y_pred[i] - y_pred[i-1])
                physics_loss += monotonic_loss.mean()
        
        # 2. SOH范围约束 (0 < SOH <= 1)
        range_loss = torch.mean(torch.relu(-y_pred) + torch.relu(y_pred - 1))
        
        return physics_loss + range_loss

    def train_one_epoch(self, epoch, dataloader):
        self.train()
        loss1_meter = AverageMeter()
        loss2_meter = AverageMeter()
        loss3_meter = AverageMeter()

        for iter, (x1, x2, y1, y2) in enumerate(dataloader):
            x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device), y2.to(device)

            # 前向传播
            u1 = self.forward(x1)
            u2 = self.forward(x2)

            # 数据损失
            loss1 = 0.5 * self.loss_func(u1, y1) + 0.5 * self.loss_func(u2, y2)

            # PDE损失
            f_target = torch.zeros_like(u1)
            loss2 = self.physics_loss(x1, u1)

            # 物理约束损失 (SOH应该随时间单调递减)
            loss3 = self.relu(u2 - u1).sum()

            # 总损失
            loss = loss1 + self.alpha * loss2 + self.beta * loss3

            # 反向传播
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            loss.backward()
            self.optimizer1.step()
            self.optimizer2.step()

            # 更新损失记录
            loss1_meter.update(loss1.item())
            loss2_meter.update(loss2.item())
            loss3_meter.update(loss3.item())

            if (iter + 1) % 50 == 0:
                print(
                    f"[epoch:{epoch} iter:{iter + 1}] data loss:{loss1:.6f}, PDE loss:{loss2:.6f}, physics loss:{loss3:.6f}")

        return loss1_meter.avg, loss2_meter.avg, loss3_meter.avg

    def Train(self, trainloader, testloader=None, validloader=None):
        min_valid_mse = float('inf')
        early_stop = 0

        for e in range(1, self.args.epochs + 1):
            early_stop += 1

            # 训练一个epoch
            loss1, loss2, loss3 = self.train_one_epoch(e, trainloader)
            current_lr = self.scheduler.step()

            # 记录训练信息
            info = f'[Train] epoch:{e}, lr:{current_lr:.6f}, total loss:{loss1 + self.alpha * loss2 + self.beta * loss3:.6f}'
            self.logger.info(info)

            # 验证
            if e % 1 == 0 and validloader is not None:
                valid_mse = self.Valid(validloader)
                info = f'[Valid] epoch:{e}, MSE: {valid_mse}'
                self.logger.info(info)

                # 测试
                if valid_mse < min_valid_mse and testloader is not None:
                    min_valid_mse = valid_mse
                    true_label, pred_label = self.Test(testloader)
                    [MAE, MAPE, MSE, RMSE] = eval_metrix(pred_label, true_label)
                    info = f'[Test] MSE: {MSE:.8f}, MAE: {MAE:.6f}, MAPE: {MAPE:.6f}, RMSE: {RMSE:.6f}'
                    self.logger.info(info)
                    early_stop = 0

                    # 保存最佳模型
                    self.best_model = {
                        'solution_u': self.solution_u.state_dict(),
                        'dynamical_F': self.dynamical_F.state_dict()
                    }
                    if self.args.save_folder is not None:
                        np.save(os.path.join(self.args.save_folder, 'true_label.npy'), true_label)
                        np.save(os.path.join(self.args.save_folder, 'pred_label.npy'), pred_label)

            # 早停
            if self.args.early_stop is not None and early_stop > self.args.early_stop:
                info = f'early stop at epoch {e}'
                self.logger.info(info)
                break

        # 保存最终模型
        if self.args.save_folder is not None and self.best_model is not None:
            torch.save(self.best_model, os.path.join(self.args.save_folder, 'model.pth'))

    def Valid(self, validloader):
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for iter, (x1, _, y1, _) in enumerate(validloader):
                x1 = x1.to(device)
                u1 = self.forward(x1)
                true_label.append(y1)
                pred_label.append(u1.cpu().detach().numpy())

        pred_label = np.concatenate(pred_label, axis=0)
        true_label = np.concatenate(true_label, axis=0)
        mse = self.loss_func(torch.tensor(pred_label), torch.tensor(true_label))
        return mse.item()

    def Test(self, testloader):
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for iter, (x1, _, y1, _) in enumerate(testloader):
                x1 = x1.to(device)
                u1 = self.forward(x1)
                true_label.append(y1)
                pred_label.append(u1.cpu().detach().numpy())

        pred_label = np.concatenate(pred_label, axis=0)
        true_label = np.concatenate(true_label, axis=0)
        return true_label, pred_label

    def predict(self, xt):
        return self.solution_u(xt)


def count_parameters(model):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {count} trainable parameters') 