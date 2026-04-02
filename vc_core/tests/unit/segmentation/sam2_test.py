from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest
import torch
from torchvision.transforms.functional import to_tensor

from vc_core.segmentation.sam import SAM2, SAMPromptConfig

if TYPE_CHECKING:
    from numpy.typing import NDArray

Devices = [torch.device("cuda")] if torch.cuda.is_available() else []


@pytest.mark.unit
@pytest.mark.parametrize("device", Devices)
def test_sam2_points(device: torch.device) -> None:
    # Download and load test image
    url = "http://images.cocodataset.org/val2017/000000000139.jpg"
    path = Path(__file__).parent / "data/coco_test.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 480))

    # Load model and test sample points method
    model = SAM2(cfg=SAMPromptConfig(n_pos=0, n_neg=0), device=device)
    model.sample_points(img)
    model.sample_points(to_tensor(img))
    model.sample_points(to_tensor(img).to(device=device))

    # Sample real prompts and test model
    model = SAM2(cfg=SAMPromptConfig(n_pos=2, n_neg=2), device=device)
    model.sample_points(img)
    mask = model.segment(img)
    assert mask.shape == img.shape[:2]

    # Blend and save input image and segmentation mask
    out_img = blend_img_mask(img, mask, rgb=(30, 144, 255))
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path.parent / "coco_test_point.jpg", out_img)


@pytest.mark.unit
@pytest.mark.parametrize("device", Devices)
def test_sam2_box(device: torch.device) -> None:
    # Download and load test image
    url = "http://images.cocodataset.org/val2017/000000000139.jpg"
    path = Path(__file__).parent / "data/coco_test.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 480))

    # Load model and test sample box method
    model = SAM2(device=device)
    model.sample_box(img)

    # Test model
    mask = model.segment(img)
    assert mask.shape == img.shape[:2]

    # Blend and save input image and segmentation mask
    out_img = blend_img_mask(img, mask, rgb=(30, 144, 255))
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path.parent / "coco_test_box.jpg", out_img)


def blend_img_mask(img: NDArray, mask: NDArray, rgb: tuple[int, int, int] | None = None) -> None:
    rgb = np.random.random(3) if rgb is None else np.array(rgb)
    alpha = 0.6
    h, w = mask.shape
    mask_image = mask.reshape(h, w, 1) * rgb.reshape(1, 1, 3)
    out = np.where(mask_image > 0, img * (1 - alpha) + alpha * mask_image, img)
    return out.astype(np.uint8)
