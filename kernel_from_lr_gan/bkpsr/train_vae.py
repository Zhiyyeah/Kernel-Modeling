from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import KernelGenerationDataset
from networks import KernelVAE


def kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def train(
    epochs: int,
    batch_size: int,
    latent_dim: int,
    hidden_dim: int,
    lr: float,
    device: torch.device,
    out_dir: Path,
    sigma_range: Tuple[float, float],
) -> None:
    dataset = KernelGenerationDataset(
        num_samples=50000,
        size=21,
        sigma_range=sigma_range,
        device=device,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = KernelVAE(latent_dim=latent_dim, hidden_dim=hidden_dim, kernel_size=21).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    out_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for kernels in loader:
            kernels = kernels.to(device)
            recon, mu, logvar = model(kernels)
            recon_loss = F.mse_loss(recon, kernels)
            kld = kl_divergence(mu, logvar)
            loss = recon_loss + kld

            optim.zero_grad()
            loss.backward()
            optim.step()

            total += loss.item() * kernels.size(0)

        avg_loss = total / len(loader.dataset)
        print(f"Epoch {epoch:03d} | loss={avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = out_dir / "kernel_vae.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"✓ Saved best VAE to {ckpt_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: train Kernel VAE")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sigma_min", type=float, default=0.8)
    parser.add_argument("--sigma_max", type=float, default=3.5)
    parser.add_argument("--out_dir", type=Path, default=Path("./bkpsr/checkpoints"))
    parser.add_argument("--use_cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu")
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        device=device,
        out_dir=args.out_dir,
        sigma_range=(args.sigma_min, args.sigma_max),
    )


if __name__ == "__main__":
    main()
