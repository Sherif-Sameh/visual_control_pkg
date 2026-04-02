from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_hf
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torch import Tensor

from .common import SAMPromptConfig

if TYPE_CHECKING:
    from numpy.typing import NDArray


SAM2_VARIANT = Literal[
    "sam2-hiera-tiny",
    "sam2-hiera-small",
    "sam2-hiera-base-plus",
    "sam2-hiera-large",
    "sam2.1-hiera-tiny",
    "sam2.1-hiera-small",
    "sam2.1-hiera-base-plus",
    "sam2.1-hiera-large",
]


class SAM2:
    """Wrapper around SAM2 segmentation model.

    The wrapper provides the following functionalities:

    - Sampling +ve and -ve point prompts through the `sample_points` method.
    - Sampling box prompts through the `sample_box` method.
    - Applying the model using gathered prompts through the `segment` method.

    Args:
        var: Variant of the SAM2/2.1 models to load.
        cfg: Configuration for collecting prompts for the SAM2 model.
        device: Device to use. Resorts to `cpu ` if `cuda` fails. Default value is `cuda`.
    """

    def __init__(
        self,
        var: SAM2_VARIANT = "sam2.1-hiera-tiny",
        cfg: SAMPromptConfig = SAMPromptConfig(),
        *,
        device: str | torch.device = "cuda",
    ):
        sam2_model = build_sam2_hf(f"facebook/{var}", device=device)
        self._predictor = SAM2ImagePredictor(sam2_model)
        self._cfg = cfg
        self._device = device if isinstance(device, str) else device.type

        # Prompt buffers
        self._point_prompts = []
        self._label_prompts = []
        self._box_prompt = None

    def sample_points(self, img: NDArray | Tensor, *, clear: bool = False) -> None:
        """Sample positive and negative point prompts for segmentation.

        The input image is displayed in a new window and the user is prompted to select points.
        Points selected through a mouse left-click are recorded internally. The number of expected
        points is set according to the configuration of `SAMPromptConfig`.

        Args:
            img: Input RGB image. If an instance of `np.ndarray`, it's expected to have shape
                (H, W, 3) and dtype of `np.uint8`. If an instance of `torch.Tensor`, it's expected
                to have shape (3, H, W) and dtype of `torch.float32` in the range [0, 1].
            clear: Optional flag to clear all prompt buffers before sampling new prompt. Default
                value is `False`.
        """
        if clear:
            self._clear_prompts()
        img_cv = cv2.cvtColor(self._to_numpy(img), cv2.COLOR_RGB2BGR)
        self._point_prompts, self._label_prompts = [], []
        total_needed = self._cfg.n_pos + self._cfg.n_neg

        def mouse_callback(event: int, x: int, y: int, flags: int, params: Any) -> None:
            if event == cv2.EVENT_LBUTTONDOWN and len(self._point_prompts) < total_needed:
                is_pos = len(self._point_prompts) < self._cfg.n_pos
                self._point_prompts.append((x, y))
                self._label_prompts.append(int(is_pos))
                color = (0, 255, 0) if is_pos else (0, 0, 255)
                cv2.circle(img_cv, (x, y), 4, color, -1)
                cv2.imshow("Point Sampling", img_cv)

        cv2.imshow("Point Sampling", img_cv)
        cv2.setMouseCallback("Point Sampling", mouse_callback)
        print(
            f"\nCollecting {total_needed} points.",
            f"First {self._cfg.n_pos} points are positive.",
            f"Last {self._cfg.n_neg} are negative.",
        )
        while len(self._point_prompts) < total_needed:
            cv2.waitKey(1)
        cv2.destroyWindow("Point Sampling")

    def sample_box(self, img: NDArray | Tensor, *, clear: bool = False) -> None:
        """Sample bounding box prompt for segmentation.

        The input image is displayed in a new window and the user is prompted to select two points.
        Points selected through a mouse left-click are recorded internally. These points correspond
        to the top-left and bottom-right corners of the bounding box respectively.

        Args:
            img: Input RGB image. If an instance of `np.ndarray`, it's expected to have shape
                (H, W, 3) and dtype of `np.uint8`. If an instance of `torch.Tensor`, it's expected
                to have shape (3, H, W) and dtype of `torch.float32` in the range [0, 1].
            clear: Optional flag to clear all prompt buffers before sampling new prompt. Default
                value is `False`.
        """
        if clear:
            self._clear_prompts()
        img_cv = cv2.cvtColor(self._to_numpy(img), cv2.COLOR_RGB2BGR)
        self._box_prompt = []

        def mouse_callback(event: int, x: int, y: int, flags: int, params: Any) -> None:
            if event == cv2.EVENT_LBUTTONDOWN and len(self._box_prompt) < 4:
                self._box_prompt.extend((x, y))
                cv2.circle(img_cv, (x, y), 4, (255, 255, 0), -1)
                cv2.imshow("Box Sampling", img_cv)

        cv2.imshow("Box Sampling", img_cv)
        cv2.setMouseCallback("Box Sampling", mouse_callback)
        print(
            "\nCollecting 2 points. for top-left and bottom-right corners of bounding box.",
            "First point is the top-left corner of the bounding box.",
            "Second point is the bottom-right corner of the bounding box.",
        )
        while len(self._box_prompt) < 4:
            cv2.waitKey(1)
        cv2.destroyWindow("Box Sampling")

    def segment(self, img: NDArray | Tensor, multimask_output: bool = False) -> NDArray:
        """Perform segmentation using collected prompts.

        Args:
            img: Input RGB image. If an instance of `np.ndarray`, it's expected to have shape
                (H, W, 3) and dtype of `np.uint8`. If an instance of `torch.Tensor`, it's expected
                to have shape (3, H, W) and dtype of `torch.float32` in the range [0, 1].
            multimask_output: If true, the model will return three masks and the one with the
                highest predicted quality score will be returned. Default value is `False`.

        Returns:
            Segmentation mask returned by SAM2 model. Shape is (H, W).
        """
        img_input = self._to_numpy(img)
        # Run inference
        with torch.autocast(self._device, dtype=torch.bfloat16):
            self._predictor.set_image(img_input)
            masks, scores, _ = self._predictor.predict(
                point_coords=np.array(self._point_prompts) if self._point_prompts else None,
                point_labels=np.array(self._label_prompts) if self._label_prompts else None,
                box=np.array(self._box_prompt) if self._box_prompt is not None else None,
                multimask_output=multimask_output,
            )
        # Return the mask with the highest score
        return masks[np.argmax(scores)]

    def _to_numpy(self, img: NDArray | Tensor) -> NDArray:
        """Convert input image to the NumPy format."""
        if isinstance(img, Tensor):
            img = (img * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        return img

    def _clear_prompts(self) -> None:
        """Clear all prompt buffers."""
        self._point_prompts.clear()
        self._label_prompts.clear()
        self._box_prompt = None
