from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import DegradationDataset
from networks import KernelPredictor, KernelVAE


def l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    return torch.mean(torch.abs(pred - target))


def train(
    hr_folder: str,
    vae_ckpt: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    out_dir: Path,
    scale: int,
    sigma_range: Tuple[float, float],
    patch_size: int,
) -> None:
    vae = KernelVAE(latent_dim=10, hidden_dim=256, kernel_size=21).to(device)
    vae.load_state_dict(torch.load(vae_ckpt, map_location=device))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    encoder = lambda k: vae.encode_only(k)  # noqa: E731

    dataset = DegradationDataset(
        hr_folder=hr_folder,
        encoder=encoder,
        scale=scale,
        kernel_size=21,
        sigma_range=sigma_range,
        patch_size=patch_size,
        device=device,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    predictor = KernelPredictor(in_channels=5, base_channels=64, latent_dim=10, num_blocks=4).to(device)
    optim = torch.optim.Adam(predictor.parameters(), lr=lr)

    out_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")

    for epoch in range(1, epochs + 1):
        predictor.train()
        total = 0.0
        for batch in loader:
            lr_img = batch["lr"].to(device)
            target_code = batch["kernel_code"].to(device)

            pred_code = predictor(lr_img)
            loss = l1_loss(pred_code, target_code)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total += loss.item() * lr_img.size(0)

        avg_loss = total / len(loader.dataset)
        print(f"Epoch {epoch:03d} | L1={avg_loss:.6f}")

        if avg_loss < best:
            best = avg_loss
            ckpt_path = out_dir / "kernel_predictor.pth"
            torch.save(predictor.state_dict(), ckpt_path)
            print(f"✓ Saved best predictor to {ckpt_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2: train kernel predictor")
    parser.add_argument("--hr_folder", type=str, required=True, help="HR .npy patches folder")
    parser.add_argument("--vae_ckpt", type=Path, default=Path("./bkpsr/checkpoints/kernel_vae.pth"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--sigma_min", type=float, default=0.8)
    parser.add_argument("--sigma_max", type=float, default=3.5)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--out_dir", type=Path, default=Path("./bkpsr/checkpoints"))
    parser.add_argument("--use_cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu")
    train(
        hr_folder=args.hr_folder,
        vae_ckpt=args.vae_ckpt,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        out_dir=args.out_dir,
        scale=args.scale,
        sigma_range=(args.sigma_min, args.sigma_max),
        patch_size=args.patch_size,
    )


if __name__ == "__main__":
    main()
