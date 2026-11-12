"""Evaluate a trained generator with Ignite's FID and Inception Score.

The implementation follows the PyTorch-Ignite GAN evaluation recipe: ImageNet
Inception-v3 features are fed to ``ignite.metrics.gan.FID`` and class
probabilities are fed to ``ignite.metrics.gan.InceptionScore``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from ignite.metrics.gan import FID, InceptionScore
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import Inception_V3_Weights, inception_v3

from code.data import WikiArtDataset
from code.models import Generator


class InceptionOutputs(nn.Module):
    """Return pool3 features and ImageNet probabilities in one forward pass."""

    def __init__(self) -> None:
        super().__init__()
        weights = Inception_V3_Weights.DEFAULT
        self.model = inception_v3(weights=weights, transform_input=False)
        self.model.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self._features: torch.Tensor | None = None

        def capture_pool3(_module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
            self._features = output.flatten(1)

        self.model.avgpool.register_forward_hook(capture_pool3)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        images = torch.nn.functional.interpolate(
            images, size=(299, 299), mode="bilinear", align_corners=False, antialias=True
        )
        logits = self.model((images - self.mean) / self.std)
        if self._features is None:
            raise RuntimeError("The Inception pool3 hook did not run")
        return self._features, logits.softmax(dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("1"))
    parser.add_argument("--num-images", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2851)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.num_images < 2:
        raise ValueError("--num-images must be at least 2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = checkpoint["config"]
    latent_dim = int(config["latent_dim"])
    generator = Generator(latent_dim, int(config["base_channels"])).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(72, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
        ]
    )
    dataset = WikiArtDataset(args.data_root, transform, args.num_images, args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    inception = InceptionOutputs().to(device).eval()
    # Features and probabilities have already been extracted, so Identity is
    # intentional here. Ignite performs its online moment/KL accumulation.
    fid = FID(num_features=2048, feature_extractor=nn.Identity(), device=device)
    inception_score = InceptionScore(
        num_features=1000, feature_extractor=nn.Identity(), device=device
    )
    fid.reset()
    inception_score.reset()
    noise_rng = torch.Generator(device=device).manual_seed(args.seed)

    seen = 0
    with torch.inference_mode():
        for step, real in enumerate(loader, start=1):
            real = real.to(device, non_blocking=True)
            noise = torch.randn(
                real.shape[0], latent_dim, 1, 1, device=device, generator=noise_rng
            )
            fake = generator(noise).float().add(1).div(2).clamp(0, 1)
            real_features, _ = inception(real)
            fake_features, fake_probabilities = inception(fake)
            fid.update((real_features, fake_features))
            inception_score.update(fake_probabilities)
            seen += real.shape[0]
            if step % 10 == 0 or step == len(loader):
                print(f"evaluated {seen}/{len(dataset)} images", flush=True)

    results = {
        "checkpoint": str(args.checkpoint),
        "num_images": seen,
        "fid": float(fid.compute()),
        "inception_score": float(inception_score.compute()),
        "seed": args.seed,
        "preprocessing": "resize-72-center-crop-64; ImageNet Inception-v3",
    }
    output = args.output or args.checkpoint.parents[1] / "metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
