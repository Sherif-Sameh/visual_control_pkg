from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest
import torch

from vc_core.segmentation.sam import SAM2LiveVideo, SAMPromptConfig

if TYPE_CHECKING:
    from numpy.typing import NDArray

Devices = [torch.device("cuda")] if torch.cuda.is_available() else []


@pytest.mark.unit
@pytest.mark.parametrize("device", Devices)
def test_sam2_video_points(device: torch.device) -> None:
    # Load test video
    path = Path(__file__).parent / "data"
    path.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(path / "input.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Load model and get sample points
    model = SAM2LiveVideo(
        cfg=SAMPromptConfig(n_pos=2, n_neg=0), device=device, overrides={"imgsz": 480}
    )
    obj_ids = [0, 0]

    # Run segmentation over all video frames
    prompts_set = False
    segmented_frames = []
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 480))
        img = img[:480, :480]
        if not prompts_set:
            model.sample_points(img)
        mask = model.segment(img, obj_ids, update_memory=not prompts_set)
        prompts_set = True
        mask = mask.cpu().numpy()
        assert mask.shape == img.shape[:2]
        out_img = blend_img_mask(img, mask, rgb=(30, 144, 255))
        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        segmented_frames.append(out_img)
    cap.release()

    # Write segmented frames to output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path / "output_points.mp4"), fourcc, fps, (480, 480))
    for frame in segmented_frames:
        out.write(frame)
    out.release()
    cv2.destroyAllWindows()


@pytest.mark.unit
@pytest.mark.parametrize("device", Devices)
def test_sam2_video_box(device: torch.device) -> None:
    # Load test video
    path = Path(__file__).parent / "data"
    path.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(path / "input.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Load model
    model = SAM2LiveVideo(
        cfg=SAMPromptConfig(n_pos=2, n_neg=0), device=device, overrides={"imgsz": 480}
    )
    obj_ids = [0]

    # Run segmentation over all video frames
    prompts_set = False
    segmented_frames = []
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 480))
        img = img[:480, :480]
        if not prompts_set:
            model.sample_box(img)
        mask = model.segment(img, obj_ids, update_memory=not prompts_set)
        prompts_set = True
        mask = mask.cpu().numpy()
        assert mask.shape == img.shape[:2]
        out_img = blend_img_mask(img, mask, rgb=(30, 144, 255))
        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        segmented_frames.append(out_img)
    cap.release()

    # Write segmented frames to output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path / "output_box.mp4"), fourcc, fps, (480, 480))
    for frame in segmented_frames:
        out.write(frame)
    out.release()
    cv2.destroyAllWindows()


def blend_img_mask(img: NDArray, mask: NDArray, rgb: tuple[int, int, int] | None = None) -> None:
    rgb = np.random.random(3) if rgb is None else np.array(rgb)
    alpha = 0.6
    h, w = mask.shape
    mask_image = mask.reshape(h, w, 1) * rgb.reshape(1, 1, 3)
    out = np.where(mask_image > 0, img * (1 - alpha) + alpha * mask_image, img)
    return out.astype(np.uint8)
