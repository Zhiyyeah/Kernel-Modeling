import torch
import torch.nn.functional as F


def lsgan_d_loss(pred_real: torch.Tensor, pred_fake: torch.Tensor) -> torch.Tensor:
    """
    LSGAN 判别器损失：0.5 * [ (D(real)-1)^2 + (D(fake)-0)^2 ]，按均值聚合。

    参数:
        pred_real (torch.Tensor): 判别器对真实样本的预测，任意形状。
        pred_fake (torch.Tensor): 判别器对生成样本的预测，任意形状。

    返回:
        torch.Tensor: 标量损失值。
    """
    loss_real = 0.5 * torch.mean((pred_real - 1.0) ** 2)
    loss_fake = 0.5 * torch.mean((pred_fake - 0.0) ** 2)
    return loss_real + loss_fake


def lsgan_g_loss(pred_fake: torch.Tensor) -> torch.Tensor:
    """
    LSGAN 生成器损失：0.5 * (D(fake)-1)^2，按均值聚合。

    参数:
        pred_fake (torch.Tensor): 判别器对生成样本的预测，任意形状。

    返回:
        torch.Tensor: 标量损失值。
    """
    return 0.5 * torch.mean((pred_fake - 1.0) ** 2)


def kernel_regularization(
    k: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 5.0,
    delta: float = 1.0,
    epsilon: float = 2.0,  # 中心最大值约束
    zeta: float = 3.0,     # 中心区域集中度
) -> torch.Tensor:
    """
    六项核正则化损失：Sum-to-1、Boundaries、Sparse、Center、CenterMax、Concentration。

    参数:
        k (torch.Tensor): 模糊核，形状 [kH, kW]，应为非负且和约为 1。
        alpha (float): Sum-to-1 项权重，惩罚核和偏离 1 的程度。默认 0.5。
        beta (float): Boundaries 项权重，惩罚边界像素非零。默认 0.5。
        gamma (float): Sparse 项权重，鼓励核的稀疏性（集中度）。默认 5.0。
        delta (float): Center 项权重，惩罚质心偏离几何中心。默认 1.0。
        epsilon (float): CenterMax 项权重，强制中心点为最大值。默认 2.0。
        zeta (float): Concentration 项权重，强制中心区域集中。默认 3.0。

    返回:
        torch.Tensor: 标量损失值，六项加权和。

    建议参数: alpha=0.5, beta=0.5, gamma=5.0, delta=1.0, epsilon=2.0, zeta=3.0
    """
    kH, kW = k.shape
    
    # Sum-to-1
    sum1 = (k.sum() - 1.0) ** 2
    
    # Boundaries：边框元素惩罚
    top = k[0, :].pow(2).sum()
    bottom = k[-1, :].pow(2).sum()
    left = k[:, 0].pow(2).sum()
    right = k[:, -1].pow(2).sum()
    boundaries = top + bottom + left + right
    
    # Sparse：0.5次方求和（鼓励稀疏）
    sparse = torch.sqrt(input=torch.clamp(input=k, min=0)).sum()
    
    # Center：重心到中心的距离平方
    yy, xx = torch.meshgrid(torch.arange(kH, device=k.device), torch.arange(kW, device=k.device), indexing='ij')
    mass = torch.clamp(input=k, min=0) + 1e-12
    cy = (yy.float() * mass).sum() / mass.sum()
    cx = (xx.float() * mass).sum() / mass.sum()
    center_y = (kH - 1) / 2.0
    center_x = (kW - 1) / 2.0
    center = (cy - center_y) ** 2 + (cx - center_x) ** 2
    
    # CenterMax：强制中心点为最大值
    # 惩罚中心点不是最大值的情况
    center_y_int = int(center_y)
    center_x_int = int(center_x)
    center_val = k[center_y_int, center_x_int]
    k_max = k.max()
    center_max_loss = (k_max - center_val) ** 2  # 中心值距离最大值的差距
    
    # Concentration：强制中心区域集中
    # 计算中心 3x3 区域的质量，应该占据大部分权重
    c_start_y = max(0, center_y_int - 1)
    c_end_y = min(kH, center_y_int + 2)
    c_start_x = max(0, center_x_int - 1)
    c_end_x = min(kW, center_x_int + 2)
    center_region_mass = k[c_start_y:c_end_y, c_start_x:c_end_x].sum()
    concentration_loss = (1.0 - center_region_mass) ** 2  # 中心区域应该接近总质量的1

    return (alpha * sum1 + 
            beta * boundaries + 
            gamma * sparse + 
            delta * center + 
            epsilon * center_max_loss + 
            zeta * concentration_loss)


if __name__ == "__main__":
    # 简易测试
    k = torch.zeros(25, 25)
    k[12, 12] = 1.0
    loss = kernel_regularization(k)
    print(f"kernel reg loss: {loss.item():.4f}")
