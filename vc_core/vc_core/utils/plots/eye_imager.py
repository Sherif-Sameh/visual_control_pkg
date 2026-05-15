from __future__ import annotations

from itertools import product
from pathlib import Path

import fire
import torch
from torchvision.utils import save_image

BACKENDS = ["cuda", "nvdiffrast"]
MODELS = ["simple", "mipmap", "hashenc"]
FILES = ["initial", "interm", "final", "target"]


def main(dir: str, idx: int = 0) -> None:
    path = Path(dir)
    for backend, model in product(BACKENDS, MODELS):
        output = Path(__file__).parent / f"figures/{backend}_{model}"
        output.mkdir(parents=True, exist_ok=True)
        p = path / f"{backend}_{model}"
        try:
            for f in FILES:
                t = torch.load(p / f"{f}.pt", weights_only=False)
                save_image(t[idx], output / f"{f}.png")
        except Exception as e:
            print(f"Failed to generate figures for {backend} and {model}: {e}")


if __name__ == "__main__":
    fire.Fire(main)
