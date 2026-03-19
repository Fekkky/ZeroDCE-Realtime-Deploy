import torch
import torch.nn as nn

import torch.optim as optim
import os

import argparse
import time
import dataloader
import model
import Myloss
import numpy as np

import torchmetrics #
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns



# 训练指标记录器
class TrainingMetricsLogger:
    def __init__(self):
        self.loss_history = []
        self.psnr_history = []
        self.ssim_history = []
        self.epochs = []
        self.current_epoch_loss = []
        self.current_epoch_psnr = []
        self.current_epoch_ssim = []

    def record_iteration(self, loss, psnr, ssim):
        """记录每个迭代的指标"""
        self.current_epoch_loss.append(loss)
        self.current_epoch_psnr.append(psnr)
        self.current_epoch_ssim.append(ssim)

    def epoch_finished(self, epoch):
        """完成一个epoch后计算平均值并记录"""
        avg_loss = np.mean(self.current_epoch_loss) if self.current_epoch_loss else 0
        avg_psnr = np.mean(self.current_epoch_psnr) if self.current_epoch_psnr else 0
        avg_ssim = np.mean(self.current_epoch_ssim) if self.current_epoch_ssim else 0

        self.loss_history.append(avg_loss)
        self.psnr_history.append(avg_psnr)
        self.ssim_history.append(avg_ssim)
        self.epochs.append(epoch + 1)

        # 重置当前epoch的记录
        self.current_epoch_loss = []
        self.current_epoch_psnr = []
        self.current_epoch_ssim = []

        return avg_loss, avg_psnr, avg_ssim

    def _plot_single_metric(self, ax, epochs, values, title, ylabel, color, marker='o'):
        """绘制单个指标的辅助函数"""
        ax.plot(epochs, values, f'{color}-{marker}', linewidth=2, markersize=5)
        ax.set_title(title, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # 添加值标注
        for i, (epoch, value) in enumerate(zip(epochs, values)):
            if i % max(1, len(epochs) // 10) == 0 or i == len(epochs) - 1:
                ax.annotate(f'{value:.4f}',
                            (epoch, value),
                            textcoords="offset points",
                            xytext=(0, 10),
                            ha='center',
                            fontsize=9)

    def plot_and_save_separate(self, save_dir, title_prefix="Training Progress"):
        """生成并保存三个单独的指标图表"""
        sns.set_style("whitegrid")

        # 创建损失图
        fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
        self._plot_single_metric(
            ax_loss,
            self.epochs,
            self.loss_history,
            f'{title_prefix} - Loss Function',
            'Loss Value',
            'b'
        )
        ax_loss.set_xlabel('Training Epoch', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_loss_1.png"), dpi=300, bbox_inches='tight')
        plt.close(fig_loss)

        # 创建PSNR图
        fig_psnr, ax_psnr = plt.subplots(figsize=(10, 6))
        self._plot_single_metric(
            ax_psnr,
            self.epochs,
            self.psnr_history,
            f'{title_prefix} - PSNR',
            'PSNR (dB)',
            'g'
        )
        ax_psnr.set_xlabel('Epoch', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_psnr_1.png"), dpi=300, bbox_inches='tight')
        plt.close(fig_psnr)

        # 创建SSIM图
        fig_ssim, ax_ssim = plt.subplots(figsize=(10, 6))
        self._plot_single_metric(
            ax_ssim,
            self.epochs,
            self.ssim_history,
            f'{title_prefix} - SSIM',
            'SSIM',
            'r'
        )
        ax_ssim.set_xlabel('Training Epoch', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_ssim_3.png"), dpi=300, bbox_inches='tight')
        plt.close(fig_ssim)

    def plot_and_save_combined(self, save_path, title="Training Progress Summary"):
        """生成并保存组合的指标图表"""
        sns.set_style("whitegrid")
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 18))
        fig.suptitle(title, fontsize=16, y=0.99)

        # 绘制损失曲线
        self._plot_single_metric(ax1, self.epochs, self.loss_history, 'Loss Function', 'Loss Value', 'b')

        # 绘制PSNR曲线
        self._plot_single_metric(ax2, self.epochs, self.psnr_history, 'PSNR', 'PSNR (dB)', 'g')

        # 绘制SSIM曲线
        self._plot_single_metric(ax3, self.epochs, self.ssim_history, 'SSIM', 'SSIM', 'r')
        ax3.set_xlabel('Training Epoch', fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.96]) # 调整布局，避免标题重叠
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


# 权重初始化函数
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# 训练函数
def train(config):
    # 创建可视化结果保存目录
    if not os.path.exists(config.visualization_folder):
        os.makedirs(config.visualization_folder)

    # 初始化指标记录器
    metrics_logger = TrainingMetricsLogger()

    # 指定 GPU 设备
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型并移动到指定设备
    DCE_net = model.enhance_net_nopool().to(device)

    # 应用权重初始化
    DCE_net.apply(weights_init)

    # 加载预训练模型权重（如果配置允许）
    if config.load_pretrain:
        try:
            DCE_net.load_state_dict(torch.load(config.pretrain_dir, map_location=device))
            print(f"Loaded pretrained weights from: {config.pretrain_dir}")
        except FileNotFoundError:
            print(f"Warning: Pretrained weights not found at: {config.pretrain_dir}. Training from scratch.")
        except Exception as e:
            print(f"Error loading pretrained weights from {config.pretrain_dir}: {e}. Training from scratch.")

    # 初始化数据集加载器
    train_dataset = dataloader.lowlight_loader(config.data_root_path)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if device.type == "cuda" else False # 仅在 CUDA 可用时 pin memory
    )

    # 初始化损失函数
    L_color = Myloss.L_color()
    L_spa = Myloss.L_spa()
    L_exp = Myloss.L_exp(16, 0.6)
    L_TV = Myloss.L_TV()

    # 初始化评估指标并移动到设备
    psnr_metric = torchmetrics.PeakSignalNoiseRatio().to(device)
    ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure().to(device)

    # 初始化优化器
    optimizer = optim.Adam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)


    # 设置模型为训练模式
    DCE_net.train()

    # 初始化最佳指标
    best_psnr = 0.0
    best_ssim = 0.0

    for epoch in range(config.num_epochs):
        epoch_start_time = time.time()

        for iteration, (img_lowlight, target_image) in enumerate(train_loader):
            # 将数据移动到指定设备
            img_lowlight = img_lowlight.to(device)
            target_image = target_image.to(device)

            # 前向传播
            enhanced_image_1, enhanced_image, A = DCE_net(img_lowlight)

            # 计算损失
            Loss_TV = 20 * L_TV(A)
            loss_spa = torch.mean(L_spa(enhanced_image, img_lowlight))
            loss_col = 0.5 * torch.mean(L_color(enhanced_image))
            loss_exp = torch.mean(L_exp(enhanced_image))

            loss = Loss_TV + loss_spa + loss_col + loss_exp

            # 优化器清零梯度，反向传播，更新权重
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(DCE_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            # 每隔一定迭代次数显示训练信息和评估指标
            if (iteration + 1) % config.display_iter == 0:
                # 计算评估指标
                with torch.no_grad():
                    current_psnr = psnr_metric(enhanced_image.detach(), target_image.detach())
                    current_ssim = ssim_metric(enhanced_image.detach(), target_image.detach())

                # 记录指标
                metrics_logger.record_iteration(loss.item(), current_psnr.item(), current_ssim.item())

                print(f"Epoch [{epoch + 1}/{config.num_epochs}], "
                      f"Iter [{iteration + 1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}, "
                      f"PSNR: {current_psnr.item():.4f}, "
                      f"SSIM: {current_ssim.item():.4f}")

        # 每个 epoch 结束后计算平均指标
        avg_loss, avg_psnr, avg_ssim = metrics_logger.epoch_finished(epoch)

        # 更新最佳指标并保存模型
        if avg_psnr > best_psnr or avg_ssim > best_ssim:
            best_psnr = max(best_psnr, avg_psnr)
            best_ssim = max(best_ssim, avg_ssim)
            best_model_path = os.path.join(config.snapshots_folder, "best_model_best_1.pth")
            torch.save(DCE_net.state_dict(), best_model_path)
            print(f"Epoch {epoch + 1} model saved at: {best_model_path}")


        # 计算 epoch 耗时
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1}/{config.num_epochs} completed, time elapsed: {epoch_time:.2f} seconds")
        print(f"Average metrics for this epoch: Loss={avg_loss:.4f}, PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}")
        print(f"Best metrics so far: PSNR={best_psnr:.4f}, SSIM={best_ssim:.4f}")

    # 训练结束后生成可视化
    print("Training completed, generating visualization charts...")

    # 生成三个单独的图表
    metrics_logger.plot_and_save_separate(config.visualization_folder, "Full Training Progress")
    print(f"Individual training metric charts generated:")
    print(f"- {os.path.join(config.visualization_folder, 'training_loss1.png')}")
    print(f"- {os.path.join(config.visualization_folder, 'training_psnr1.png')}")
    print(f"- {os.path.join(config.visualization_folder, 'training_ssim1.png')}")

    # 生成汇总图表
    final_viz_path = os.path.join(config.visualization_folder, "training_progress_1.png")
    metrics_logger.plot_and_save_combined(final_viz_path, "Full Training Progress Summary")
    print(f"Combined training chart generated: {final_viz_path}")


# 主程序入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 输入参数配置
    parser.add_argument('--data_root_path', type=str, default="data/train_data")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=100)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--load_pretrain', action='store_true', help='Whether to load pretrained model weights.')
    parser.add_argument('--pretrain_dir', type=str, default="snapshots/Epoch50.pth",
                        help='Path to pretrained model weights.')
    parser.add_argument('--visualization_folder', type=str, default="visualization/",
                        help='Path to save training visualization results.')

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.makedirs(config.snapshots_folder)

    if not os.path.exists(config.visualization_folder):
        os.makedirs(config.visualization_folder)

    train(config)