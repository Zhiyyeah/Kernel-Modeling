import torch
import torch.nn.functional as F


def lsgan_d_loss(pred_real: torch.Tensor, pred_fake: torch.Tensor) -> torch.Tensor:
    loss_real = 0.5 * torch.mean((pred_real - 1.0) ** 2)
    loss_fake = 0.5 * torch.mean((pred_fake - 0.0) ** 2)
    return loss_real + loss_fake


def lsgan_g_loss(pred_fake: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.mean((pred_fake - 1.0) ** 2)


def kernel_regularization(
    k: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 5.0,
    delta: float = 1.0,
) -> torch.Tensor:
    kH, kW = k.shape
    sum1 = (k.sum() - 1.0) ** 2
    top = k[0, :].pow(2).sum()
    bottom = k[-1, :].pow(2).sum()
    left = k[:, 0].pow(2).sum()
    right = k[:, -1].pow(2).sum()
    boundaries = top + bottom + left + right
    sparse = torch.sqrt(input=torch.clamp(input=k, min=0)).sum()
    yy, xx = torch.meshgrid(torch.arange(kH, device=k.device), torch.arange(kW, device=k.device), indexing='ij')
    mass = torch.clamp(input=k, min=0) + 1e-12
    cy = (yy.float() * mass).sum() / mass.sum()
    cx = (xx.float() * mass).sum() / mass.sum()
    center_y = (kH - 1) / 2.0
    center_x = (kW - 1) / 2.0
    center = (cy - center_y) ** 2 + (cx - center_x) ** 2
    return alpha * sum1 + beta * boundaries + gamma * sparse + delta * center


def noise_reg_loss(sigma: torch.Tensor, target: float = 0.01, mode: str = 'l2') -> torch.Tensor:
    """
    噪声强度正则：约束 sigma 不要过大，避免把高频细节全部吸收到噪声里。
    sigma: [C]
    mode: 'l1' 或 'l2'
    """
    if mode == 'l1':
        return torch.mean(torch.abs(sigma - target))
    return torch.mean((sigma - target) ** 2)


if __name__ == "__main__":
    k = torch.zeros(13, 13)
    k[6, 6] = 1.0
    loss = kernel_regularization(k)
    print(f"kernel reg loss: {loss.item():.4f}")
    sigma = torch.tensor([0.05, 0.02, 0.01])
    print(f"noise reg: {noise_reg_loss(sigma).item():.6f}")
