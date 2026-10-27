from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

class ConvVAE(nn.Module):
    """Convolutional VAE for square, single-channel images."""

    def __init__(self, image_size: int = 128, latent_dim: int = 128) -> None:
        super().__init__()
        if image_size < 32 or image_size % 16 != 0:
            raise ValueError("image_size must be >= 32 and divisible by 16")

        self.image_size = image_size
        self.latent_dim = latent_dim
        self.feature_size = image_size // 16
        self.feature_shape = (256, self.feature_size, self.feature_size)
        self.flat_dim = int(np.prod(self.feature_shape))

        self.encoder = nn.Sequential(
            self._encoder_block(1, 32),
            self._encoder_block(32, 64),
            self._encoder_block(64, 128),
            self._encoder_block(128, 256),
        )
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, self.flat_dim)
        self.decoder = nn.Sequential(
            self._decoder_block(256, 128),
            self._decoder_block(128, 64),
            self._decoder_block(64, 32),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
        )

    @staticmethod
    def _encoder_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    @staticmethod
    def _decoder_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x).flatten(start_dim=1)
        return self.fc_mu(features), self.fc_logvar(features)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            # Using the mean makes validation/reconstruction deterministic.
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        features = self.decoder_input(z).view(-1, *self.feature_shape)
        return self.decoder(features)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar


def vae_loss(
    reconstruction_logits: torch.Tensor,
    image: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return batch-mean ELBO loss, BCE reconstruction loss, and KL loss."""
    batch_size = image.size(0)
    # BCE-with-logits is numerically stable and safe under CUDA autocast.
    reconstruction_loss = F.binary_cross_entropy_with_logits(
        reconstruction_logits, image, reduction="sum"
    ) / batch_size
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    return reconstruction_loss + beta * kl_loss, reconstruction_loss, kl_loss


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_loaders(
    args: argparse.Namespace, splits: tuple[str, ...] = ("train", "val", "test")
) -> dict[str, DataLoader[Any]]:
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
    )
    loaders: dict[str, DataLoader[Any]] = {}
    for split in splits:
        split_dir = args.data_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing dataset directory: {split_dir}")
        dataset = datasets.ImageFolder(split_dir, transform=transform)
        generator = torch.Generator().manual_seed(args.seed)
        loaders[split] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.num_workers > 0,
            generator=generator,
        )
        print(f"{split:>5}: {len(dataset):5d} images, classes={dataset.classes}")
    return loaders


def run_epoch(
    model: ConvVAE,
    loader: DataLoader[Any],
    device: torch.device,
    beta: float,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "reconstruction": 0.0, "kl": 0.0}
    sample_count = 0

    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                reconstruction_logits, mu, logvar = model(images)
                loss, reconstruction_loss, kl_loss = vae_loss(
                    reconstruction_logits, images, mu, logvar, beta
                )
            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batch_size = images.size(0)
        totals["loss"] += loss.item() * batch_size
        totals["reconstruction"] += reconstruction_loss.item() * batch_size
        totals["kl"] += kl_loss.item() * batch_size
        sample_count += batch_size

    return {name: value / sample_count for name, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: ConvVAE,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val: float,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val": best_val,
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        },
        path,
    )


@torch.inference_mode()
def save_reconstructions(
    model: ConvVAE,
    loader: DataLoader[Any],
    device: torch.device,
    path: Path,
    count: int,
) -> None:
    model.eval()
    images, _ = next(iter(loader))
    images = images[:count].to(device)
    reconstruction_logits, _, _ = model(images)
    reconstruction = torch.sigmoid(reconstruction_logits)
    # Each row contains an original image followed by its reconstruction.
    pairs = torch.stack((images.cpu(), reconstruction.cpu()), dim=1).flatten(0, 1)
    save_image(pairs, path, nrow=2, padding=2)


@torch.inference_mode()
def evaluate_test(
    model: ConvVAE,
    loader: DataLoader[Any],
    device: torch.device,
    beta: float,
    compute_fid_is: bool = True,
    inception_splits: int = 10,
) -> dict[str, float]:
    """Evaluate reconstruction and optional Inception-based distribution metrics."""
    metrics = run_epoch(model, loader, device, beta)
    squared_error = 0.0
    pixel_count = 0
    model.eval()

    fid_metric = None
    inception_metric = None
    if compute_fid_is:
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
            from torchmetrics.image.inception import InceptionScore
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError(
                "FID/Inception Score require torchmetrics and torch-fidelity. "
                "Install them with: pip install torchmetrics torch-fidelity"
            ) from error

        # normalize=True means that float images in [0, 1] are accepted. Both
        # metrics use an ImageNet Inception-v3 and resize inputs internally.
        fid_metric = FrechetInceptionDistance(
            feature=2048, normalize=True
        ).to(device)
        inception_metric = InceptionScore(
            normalize=True, splits=inception_splits
        ).to(device)

    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        reconstruction_logits, _, _ = model(images)
        reconstruction = torch.sigmoid(reconstruction_logits)
        squared_error += F.mse_loss(reconstruction, images, reduction="sum").item()
        pixel_count += images.numel()

        if fid_metric is not None and inception_metric is not None:
            # Inception-v3 expects RGB. Repeating a grayscale channel preserves
            # intensity while satisfying the required three-channel shape.
            real_rgb = images.repeat(1, 3, 1, 1)
            reconstructed_rgb = reconstruction.repeat(1, 3, 1, 1)
            fid_metric.update(real_rgb, real=True)
            fid_metric.update(reconstructed_rgb, real=False)
            inception_metric.update(reconstructed_rgb)

    metrics["mse_per_pixel"] = squared_error / pixel_count
    metrics["psnr_db"] = -10.0 * np.log10(metrics["mse_per_pixel"])
    if fid_metric is not None and inception_metric is not None:
        inception_mean, inception_std = inception_metric.compute()
        metrics["fid"] = float(fid_metric.compute().item())
        metrics["inception_score_mean"] = float(inception_mean.item())
        metrics["inception_score_std"] = float(inception_std.item())
        metrics["inception_score_splits"] = float(inception_splits)
    return metrics


@torch.inference_mode()
def export_latents(
    model: ConvVAE,
    loader: DataLoader[Any],
    device: torch.device,
    path: Path,
) -> None:
    """Save deterministic latent means and labels for compressed test images."""
    model.eval()
    means, labels = [], []
    for images, batch_labels in loader:
        mu, _ = model.encode(images.to(device, non_blocking=True))
        means.append(mu.cpu())
        labels.append(batch_labels)
    dataset = loader.dataset
    image_paths = [path for path, _ in dataset.samples]  # type: ignore[attr-defined]
    torch.save(
        {
            "latent_mu": torch.cat(means),
            "labels": torch.cat(labels),
            "classes": dataset.classes,  # type: ignore[attr-defined]
            "image_paths": image_paths,
        },
        path,
    )


def plot_history(history: list[dict[str, float]], path: Path) -> None:
    epochs = [int(item["epoch"]) for item in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [item["train_loss"] for item in history], label="train")
    plt.plot(epochs, [item["val_loss"] for item in history], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Negative ELBO per image")
    plt.title("VAE training curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a convolutional VAE on chest X-ray images."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("chest_xray/chest_xray"))
    parser.add_argument("--output-dir", type=Path, default=Path("vae_outputs"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0, help="KL loss weight")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and evaluate an existing checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint for --eval-only (default: OUTPUT_DIR/best_model.pt)",
    )
    parser.add_argument(
        "--skip-fid-is",
        action="store_true",
        help="Skip FID and Inception Score computation",
    )
    parser.add_argument(
        "--inception-splits",
        type=int,
        default=10,
        help="Number of splits used for Inception Score",
    )
    parser.add_argument("--reconstruction-count", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Keep downloaded Inception weights with the experiment. This also works
    # in environments where the default ~/.cache/torch directory is read-only.
    torch.hub.set_dir(str(args.output_dir / ".torch_cache"))
    device = choose_device(args.device)
    print(f"device: {device}")

    if args.inception_splits < 1:
        raise ValueError("--inception-splits must be at least 1")

    requested_splits = ("test",) if args.eval_only else ("train", "val", "test")
    loaders = create_loaders(args, requested_splits)
    model = ConvVAE(args.image_size, args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    start_epoch, best_val = 1, float("inf")
    history: list[dict[str, float]] = []

    if args.eval_only:
        checkpoint_path = args.checkpoint or args.output_dir / "best_model.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"evaluating checkpoint: {checkpoint_path}")
    elif args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint["best_val"])
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    if not args.eval_only:
        for epoch in range(start_epoch, args.epochs + 1):
            train_metrics = run_epoch(
                model, loaders["train"], device, args.beta, optimizer, scaler
            )
            val_metrics = run_epoch(model, loaders["val"], device, args.beta)
            record = {
                "epoch": float(epoch),
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            history.append(record)
            print(
                f"epoch {epoch:03d}/{args.epochs:03d} | "
                f"train {train_metrics['loss']:.3f} | val {val_metrics['loss']:.3f} "
                f"(recon {val_metrics['reconstruction']:.3f}, "
                f"KL {val_metrics['kl']:.3f})"
            )

            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                save_checkpoint(
                    args.output_dir / "best_model.pt",
                    model,
                    optimizer,
                    epoch,
                    best_val,
                    args,
                )
            save_checkpoint(
                args.output_dir / "last_model.pt",
                model,
                optimizer,
                epoch,
                best_val,
                args,
            )
            with (args.output_dir / "history.json").open(
                "w", encoding="utf-8"
            ) as file:
                json.dump(history, file, indent=2)

        best_checkpoint = torch.load(
            args.output_dir / "best_model.pt", map_location=device
        )
        model.load_state_dict(best_checkpoint["model_state"])

    test_metrics = evaluate_test(
        model,
        loaders["test"],
        device,
        args.beta,
        compute_fid_is=not args.skip_fid_is,
        inception_splits=args.inception_splits,
    )
    with (args.output_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(test_metrics, file, indent=2)
    save_reconstructions(
        model,
        loaders["test"],
        device,
        args.output_dir / "test_reconstructions.png",
        args.reconstruction_count,
    )
    export_latents(
        model, loaders["test"], device, args.output_dir / "test_latents.pt"
    )
    if history:
        plot_history(history, args.output_dir / "training_curve.png")

    print(f"test metrics: {json.dumps(test_metrics, indent=2)}")
    print(f"outputs saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
