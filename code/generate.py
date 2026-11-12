"""Generate a grid of paintings from a trained checkpoint."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torchvision.utils import save_image

from code.models import Generator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("generated_paintings.png"))
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2850)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = checkpoint["config"]
    generator = Generator(int(config["latent_dim"]), int(config["base_channels"])).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    random = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(args.count, generator.latent_dim, 1, 1, generator=random, device=device)
    with torch.inference_mode():
        images = generator(noise).cpu()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(images, args.output, nrow=max(1, round(math.sqrt(args.count))),
               normalize=True, value_range=(-1, 1))
    print(f"Saved {args.count} generated paintings to {args.output}")


if __name__ == "__main__":
    main()
