"""Train a 64x64 DCGAN on the WikiArt dataset.

Example:
    conda run -n mjlab python train.py --data-root 1 --epochs 20
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from code.data import WikiArtDataset
from code.models import Discriminator, Generator, count_parameters, dcgan_weight_init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("1"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/dcgan64"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--real-label", type=float, default=0.9,
                        help="One-sided label smoothing target")
    parser.add_argument("--seed", type=int, default=285)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max-images", type=int, default=None,
                        help="Optional subset size, useful for smoke tests")
    parser.add_argument("--max-steps-per-epoch", type=int, default=None,
                        help="Optional step cap, useful for smoke tests")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_losses(csv_path: Path, output_path: Path) -> None:
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    epochs = [int(row["epoch"]) for row in rows]
    d_losses = [float(row["d_loss"]) for row in rows]
    g_losses = [float(row["g_loss"]) for row in rows]
    plt.figure(figsize=(5.2, 3.2))
    plt.plot(epochs, d_losses, "o-", label="Discriminator")
    plt.plot(epochs, g_losses, "o-", label="Generator")
    plt.xlabel("Epoch")
    plt.ylabel("BCE loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def atomic_torch_save(payload: dict, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.real_label <= 1.0:
        raise ValueError("--real-label must be in (0, 1]")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = args.output_dir / "samples"
    checkpoint_dir = args.output_dir / "checkpoints"
    sample_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    transform = transforms.Compose(
        [
            transforms.Resize(72, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ]
    )
    dataset = WikiArtDataset(args.data_root, transform, args.max_images, args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        drop_last=True,
    )
    if len(loader) == 0:
        raise RuntimeError("The dataset is smaller than one batch; reduce --batch-size")

    generator = Generator(args.latent_dim, args.base_channels).to(device)
    discriminator = Discriminator(args.base_channels).to(device)
    generator.apply(dcgan_weight_init)
    discriminator.apply(dcgan_weight_init)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    fixed_noise = torch.randn(args.sample_count, args.latent_dim, 1, 1, device=device)
    start_epoch = 1

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        fixed_noise = checkpoint.get("fixed_noise", fixed_noise).to(device)
        start_epoch = int(checkpoint["epoch"]) + 1

    config = vars(args).copy()
    config.update(
        device=str(device),
        amp=use_amp,
        dataset_size=len(dataset),
        generator_parameters=count_parameters(generator),
        discriminator_parameters=count_parameters(discriminator),
    )
    config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2), flush=True)

    csv_path = args.output_dir / "losses.csv"
    if start_epoch == 1 or not csv_path.exists():
        with csv_path.open("w", newline="") as handle:
            csv.writer(handle).writerow(
                ["epoch", "d_loss", "g_loss", "D_x", "D_G_z1", "D_G_z2", "seconds"]
            )

    for epoch in range(start_epoch, args.epochs + 1):
        generator.train()
        discriminator.train()
        totals = {key: 0.0 for key in ("d", "g", "dx", "dgz1", "dgz2")}
        examples = 0
        epoch_start = time.perf_counter()

        for step, real in enumerate(loader, start=1):
            if args.max_steps_per_epoch is not None and step > args.max_steps_per_epoch:
                break
            real = real.to(device, non_blocking=True)
            batch_size = real.shape[0]
            real_targets = torch.full((batch_size,), args.real_label, device=device)
            fake_targets = torch.zeros(batch_size, device=device)
            generator_targets = torch.ones(batch_size, device=device)

            # Maximize log D(x) + log(1 - D(G(z))).
            optimizer_d.zero_grad(set_to_none=True)
            noise = torch.randn(batch_size, args.latent_dim, 1, 1, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                fake = generator(noise)
                real_logits = discriminator(real)
                fake_logits = discriminator(fake.detach())
                loss_d = criterion(real_logits, real_targets) + criterion(fake_logits, fake_targets)
            scaler.scale(loss_d).backward()
            scaler.step(optimizer_d)
            scaler.update()

            # Non-saturating generator objective: maximize log D(G(z)).
            optimizer_g.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                generated_logits = discriminator(fake)
                loss_g = criterion(generated_logits, generator_targets)
            scaler.scale(loss_g).backward()
            scaler.step(optimizer_g)
            scaler.update()

            with torch.no_grad():
                dx = real_logits.sigmoid().mean().item()
                dgz1 = fake_logits.sigmoid().mean().item()
                dgz2 = generated_logits.sigmoid().mean().item()
            totals["d"] += loss_d.item() * batch_size
            totals["g"] += loss_g.item() * batch_size
            totals["dx"] += dx * batch_size
            totals["dgz1"] += dgz1 * batch_size
            totals["dgz2"] += dgz2 * batch_size
            examples += batch_size

            if step % args.log_every == 0 or step == len(loader):
                print(
                    f"epoch {epoch:02d}/{args.epochs} step {step:04d}/{len(loader):04d} "
                    f"D={loss_d.item():.4f} G={loss_g.item():.4f} "
                    f"D(x)={dx:.3f} D(G(z))={dgz1:.3f}/{dgz2:.3f}",
                    flush=True,
                )

        elapsed = time.perf_counter() - epoch_start
        averages = {key: value / examples for key, value in totals.items()}
        with csv_path.open("a", newline="") as handle:
            csv.writer(handle).writerow(
                [epoch, averages["d"], averages["g"], averages["dx"],
                 averages["dgz1"], averages["dgz2"], elapsed]
            )

        generator.eval()
        with torch.inference_mode():
            samples = generator(fixed_noise).float().cpu()
        nrow = max(1, round(math.sqrt(args.sample_count)))
        save_image(samples, sample_dir / f"epoch_{epoch:03d}.png", nrow=nrow,
                   normalize=True, value_range=(-1, 1))
        plot_losses(csv_path, args.output_dir / "loss_curves.png")

        payload = {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "scaler": scaler.state_dict(),
            "fixed_noise": fixed_noise.detach().cpu(),
            "config": config,
        }
        atomic_torch_save(payload, checkpoint_dir / "latest.pt")
        if epoch % args.checkpoint_every == 0:
            atomic_torch_save(payload, checkpoint_dir / f"epoch_{epoch:03d}.pt")
        print(
            f"epoch {epoch:02d} complete: D={averages['d']:.4f}, "
            f"G={averages['g']:.4f}, {elapsed:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
