import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
from networks import PatchDiscriminator  # 沿用之前的判别器
from loss import lsgan_d_loss, lsgan_g_loss, kernel_regularization

# ==========================================
# 1. 定义简单的感知 CNN (Selector)
# ==========================================
class SelectorNet(nn.Module):
    def __init__(self, in_ch=5, num_classes=10):
        super().__init__()
        # 一个非常轻量级的 CNN，用于感知内容纹理
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1), # 128 -> 64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 32 -> 16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d(1) # [B, 128, 1, 1] 全局池化，提取内容特征
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        logits = self.classifier(feat)
        return logits

# ==========================================
# 2. 定义包含"核库"的自适应退化模型
# ==========================================
class ContentAdaptiveDegradation(nn.Module):
    def __init__(self, n_kernels=10, n_channels=5, kernel_size=13):
        super().__init__()
        self.n_kernels = n_kernels
        self.k_size = kernel_size
        self.pad = kernel_size // 2
        
        # A. 感知器：决定用哪个核
        self.selector = SelectorNet(in_ch=n_channels, num_classes=n_kernels)
        
        # B. 核库 (Kernel Bank): [10, 5, 13, 13]
        # 直接作为参数学习，初始化为近似 Delta 分布（中心为1，四周为0）
        # 这样训练开始时不会太模糊
        initial_kernels = torch.zeros(n_kernels, n_channels, kernel_size, kernel_size)
        center = kernel_size // 2
        initial_kernels[:, :, center, center] = 1.0
        # 加一点点随机噪声打破对称性
        initial_kernels += torch.randn_like(initial_kernels) * 0.01
        self.kernel_bank = nn.Parameter(initial_kernels)
        
        # C. 噪声库 (Noise Bank): [10, 5]
        # 初始化噪声 sigma 为 0.5 左右
        self.sigma_bank = nn.Parameter(torch.ones(n_kernels, n_channels) * 0.5)

    def get_effective_kernels(self):
        """
        获取当前库中物理有效的核（经过 Softmax 保证和为1，非负）
        """
        # 对空间维度 (H, W) 做 Softmax，保证核的物理性质：和为1，且非负
        # Shape: [10, 5, 13, 13]
        k = F.softmax(self.kernel_bank, dim=-1) # 先对最后一维
        # 注意：通常需要对 H*W 展平做 softmax，或者利用 Spatial Softmax
        # 这里为了严谨，我们手动 reshape 做 softmax
        B_k, C_k, H_k, W_k = self.kernel_bank.shape
        k_flat = self.kernel_bank.view(B_k, C_k, -1)
        k_norm = F.softmax(k_flat, dim=-1).view(B_k, C_k, H_k, W_k)
        return k_norm

    def get_effective_sigmas(self):
        """
        获取有效的噪声值（保证非负）
        """
        return F.softplus(self.sigma_bank)

    def forward(self, x, temp=1.0, hard=False):
        """
        Args:
            x: [B, 5, H, W]
            temp: Gumbel Softmax 的温度，越小越趋向于 one-hot (硬选择)
            hard: 是否强制进行硬选择 (True时，权重非0即1)
        """
        B, C, H, W = x.shape
        
        # 1. 感知内容，输出选择权重 [B, 10]
        logits = self.selector(x)
        
        # 使用 Gumbel-Softmax 技巧：
        # 训练初期是 Soft 的（为了梯度传播），所有核都参与一点点；
        # 训练后期趋向于 Hard（为了明确分类），每张图只选一个核。
        weights = F.gumbel_softmax(logits, tau=temp, hard=hard, dim=1) 
        # weights shape: [B, 10]
        
        # 2. 从库中组合出当前 Batch 对应的核与噪声
        valid_kernels = self.get_effective_kernels() # [10, 5, kh, kw]
        valid_sigmas = self.get_effective_sigmas()   # [10, 5]
        
        # 混合核: [B, 10] x [10, 5, kh, kw] -> [B, 5, kh, kw]
        #利用爱因斯坦求和约定进行加权求和
        batch_kernels = torch.einsum('bk, kchw -> bchw', weights, valid_kernels)
        
        # 混合噪声: [B, 10] x [10, 5] -> [B, 5]
        batch_sigmas = torch.einsum('bk, kc -> bc', weights, valid_sigmas)
        
        # 3. 执行退化操作 (卷积)
        # 只能用 Group Conv 来实现逐样本不同的核卷积比较麻烦
        # 这里为了简单，我们使用 Loop 或者 reshape 分组卷积
        # PyTorch 的 F.conv2d 不支持 batch 维度的不同核。
        # 高效写法：将 Batch 维度视为 Group
        
        # Input: [1, B*C, H, W]
        x_reshape = x.view(1, B*C, H, W)
        # Weights: [B*C, 1, kh, kw]
        k_reshape = batch_kernels.view(B*C, 1, self.k_size, self.k_size)
        
        out = F.conv2d(x_reshape, k_reshape, padding=self.pad, groups=B*C)
        out = out.view(B, C, H, W)
        
        # 4. 下采样 (这里简单用隔点采样，也可以学一个步长为2的卷积)
        # 模拟 Landsat (30m) -> GOCI (250m) 约 8 倍，但为了训练 GAN 通常先由 PatchSize 决定
        # 假设这里做 4 倍下采样
        out = out[:, :, ::4, ::4]
        
        # 5. 加噪声
        noise = torch.randn_like(out) * batch_sigmas.view(B, C, 1, 1)
        out = out + noise
        
        return out, weights, valid_kernels

# ==========================================
# 3. 主训练流程
# ==========================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化
    model = ContentAdaptiveDegradation(n_kernels=10, n_channels=5).to(device)
    D = PatchDiscriminator(in_ch=5).to(device)
    
    # 你的数据加载代码 (load_patches_from_folder 等) ...
    # 假设 patches 是 [B, 5, 256, 256], real_ds 是 [B, 5, 64, 64]
    # 这里省略数据加载部分，直接写训练循环逻辑
    
    opt_G = optim.Adam(model.parameters(), lr=1e-4)
    opt_D = optim.Adam(D.parameters(), lr=1e-4)
    
    # 温度衰减：从 5.0 降到 0.5，让选择越来越明确
    total_iters = 5000
    temp_schedule = np.linspace(5.0, 0.5, total_iters)
    
    # 模拟数据 (请替换为真实 DataLoader)
    dummy_patches = torch.randn(8, 5, 256, 256).to(device)
    dummy_real = torch.randn(8, 5, 64, 64).to(device)

    print("开始训练 Mixture of Kernels...")
    
    for i in tqdm(range(total_iters)):
        # 1. 获取数据 (请替换)
        patches = dummy_patches 
        real_ds = dummy_real
        
        # 获取当前温度
        temp = temp_schedule[i]
        
        # ====================
        # Train Discriminator
        # ====================
        with torch.no_grad():
            # Generate fake LR
            fake_ds, weights, _ = model(patches, temp=temp, hard=False)
            
        pred_real = D(real_ds)
        pred_fake = D(fake_ds.detach())
        loss_D = lsgan_d_loss(pred_real, pred_fake)
        
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()
        
        # ====================
        # Train Generator (Selector + Banks)
        # ====================
        fake_ds, weights, current_kernels = model(patches, temp=temp, hard=False)
        pred_fake = D(fake_ds)
        loss_G_adv = lsgan_g_loss(pred_fake)
        
        # 正则化：希望 10 个核尽量不一样 (Diversity Loss)
        # 计算核之间的余弦相似度，希望越小越好
        # Flatten kernels: [10, 5*13*13]
        k_flat = current_kernels.view(10, -1)
        # 简单的多样性惩罚：也就是 -1 * 距离
        
        # 修改后的代码
        # 1. 先计算 10 个模版核的平均核，形状为 [5, 13, 13]
        avg_kernel_per_channel = current_kernels.mean(dim=0)

        # 2. 对 5 个波段分别计算正则化 Loss，再求均值
        reg_list = []
        for c in range(avg_kernel_per_channel.shape[0]):
            # 取出第 c 个波段的核 [13, 13]
            k_2d = avg_kernel_per_channel[c]
            # 计算 Loss
            reg_list.append(kernel_regularization(k_2d, alpha=0.5))

        reg_loss = torch.mean(torch.stack(reg_list))
                
                # 熵正则化：鼓励 weights 接近 one-hot (非黑即白)，或者依赖 Gumbel 自身的特性
                # entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=1).mean()
            
        loss_G = loss_G_adv + reg_loss
        
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()
        
        if i % 100 == 0:
            # 打印当前哪个核被选中的最多
            selected_indices = weights.argmax(dim=1)
            counts = torch.bincount(selected_indices, minlength=10)
            tqdm.write(f"Iter {i} | Temp: {temp:.2f} | D: {loss_D.item():.3f} | Selection Dist: {counts.cpu().numpy()}")

    # ==========================================
    # 4. 保存结果
    # ==========================================
    save_dir = './moe_kernels'
    os.makedirs(save_dir, exist_ok=True)
    
    # 提取最终学到的 10 个核和噪声
    final_kernels = model.get_effective_kernels().detach().cpu().numpy() # [10, 5, 13, 13]
    final_sigmas = model.get_effective_sigmas().detach().cpu().numpy()   # [10, 5]
    
    print(f"\n训练完成！保存 10 个退化核到 {save_dir}")
    for k_idx in range(10):
        # 保存核
        np.save(os.path.join(save_dir, f'kernel_{k_idx}.npy'), final_kernels[k_idx])
        # 保存噪声
        np.save(os.path.join(save_dir, f'sigma_{k_idx}.npy'), final_sigmas[k_idx])
    
    # 保存模型权重 (Selector + Banks)
    torch.save(model.state_dict(), os.path.join(save_dir, 'moe_model.pth'))

if __name__ == '__main__':
    main()