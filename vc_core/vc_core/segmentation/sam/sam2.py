from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np
import torch
from torch import Tensor
from torchvision.transforms.functional import to_tensor
from ultralytics.models.sam import SAM2Predictor

if TYPE_CHECKING:
    from numpy.typing import NDArray


SAM2_VARIANT = Literal[
    "sam2_t", "sam2_s", "sam2_b", "sam2_l", "sam2.1_t", "sam2.1_s", "sam2.1_b", "sam2.1_l"
]


@dataclass
class PromptConfig:
    """Configuration for collecting prompts for the SAM2 model."""

    n_pos: int = 1
    """Number of positive points to expect for point prompts. Default value is 1."""

    n_neg: int = 0
    """Number of negative points to expect for point prompts. Default value is 0."""

    n_mask_pts: int = 4
    """Number of points to expect for polygon for mask prompts. Default value is 4."""


class SAM2:
    """Wrapper around SAM2 segmentation model from Ultralytics.

    The wrapper provides the following functionalities:

    - Sampling +ve and -ve point prompts through the `sample_points` method.
    - Sampling box prompts through the `sample_box` method.
    - Sampling polygon-based mask prompts through the `sample_mask` method.
    - Applying the model using gathered prompts through the `segment` method.

    Args:
        var: Variant of the SAM2/2.1 models to load.
        cfg: Configuration for collecting prompts for the SAM2 model.
        device: Device to use. Resorts to `cpu ` if `cuda` fails. Default value is `cuda`.
    """

    def __init__(
        self,
        var: SAM2_VARIANT = "sam2.1_t",
        cfg: PromptConfig = PromptConfig(),
        *,
        device: str | torch.device = "cuda",
    ):
        self._predictor = SAM2Predictor(overrides=dict(model=var, device=device))
        self._cfg = cfg
        self._device = device

        # Prompt buffers
        self._point_prompts = []
        self._point_prompts = []
        self._box_prompt = None
        self._mask_prompt = None

    def sample_points(self, img: NDArray | Tensor, *, clear: bool = False) -> None:
        """Sample positive and negative point prompts for segmentation.

        The input image is displayed in a new window and the user is prompted to select points.
        Points selected through a mouse left-click are recorded internally. The number of expected
        points is set according to the configuration of `PromptConfig`.

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
        self._point_prompts, self._point_prompts = [], []
        total_needed = self._cfg.n_pos + self._cfg.n_neg

        def mouse_callback(event: int, x: int, y: int, flags: int, params: Any) -> None:
            if event == cv2.EVENT_LBUTTONDOWN and len(self._point_prompts) < total_needed:
                is_pos = len(self._point_prompts) < self._cfg.n_pos
                self._point_prompts.append((x, y))
                self._point_prompts.append(int(is_pos))
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

    def sample_mask(self, img: NDArray | Tensor, *, clear: bool = False) -> None:
        """Sample polygon-based mask prompt for segmentation.

        The input image is displayed in a new window and the user is prompted to select points.
        Points selected through a mouse left-click are recorded internally. The polygon made up
        from these points is then used to extract an image mask. The number of expected points is
        set according to the configuration of `PromptConfig`.

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
        h, w, _ = img_cv.shape
        points = []

        def mouse_callback(event: int, x: int, y: int, flags: int, params: Any) -> None:
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < self._cfg.n_mask_pts:
                points.append((x, y))
                cv2.circle(img_cv, (x, y), 5, (255, 0, 255), -1)
                if len(points) > 1:
                    cv2.line(img_cv, points[-2], points[-1], (255, 0, 255), 2)
                cv2.imshow("Mask Sampling", img_cv)

        cv2.imshow("Mask Sampling", img_cv)
        cv2.setMouseCallback("Mask Sampling", mouse_callback)
        while len(points) < self._cfg.n_mask_pts:
            cv2.waitKey(1)
        cv2.destroyWindow("Mask Sampling")

        self._mask_prompt = np.zeros((h, w), dtype=np.uint8)
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(self._mask_prompt, [points], 1)

    def segment(self, img: NDArray | Tensor) -> NDArray:
        img_input = self._to_torch(img, device=self._device)
        # Run inference
        self._predictor.set_image(img_input)
        results = self._predictor(
            source=img_input,
            points=np.array([self._point_prompts]) if self._point_prompts else None,
            labels=np.array([self._point_prompts]) if self._point_prompts else None,
            bboxes=np.array([self._box_prompt]) if self._box_prompt is not None else None,
            masks=cv2.resize(self._mask_prompt)[None] if self._mask_prompt is not None else None,
        )
        results = self.model.predict(
            source=img_input,
            points=np.arrayself.point_prompts if self._point_prompts else None,
            labels=self._point_prompts if self._point_prompts else None,
            bboxes=self._box_prompt if self._box_prompt is not None else None,
            mask_input=self._mask_prompt if self._mask_prompt is not None else None,
            device=self._device,
        )

        # Return the first mask of the first result
        if results and len(results[0].masks.data) > 0:
            return results[0].masks.data[0].cpu().numpy()
        return np.zeros(img_input.shape[:2])

    def _to_numpy(self, img: NDArray | Tensor) -> NDArray:
        if isinstance(img, Tensor):
            img = (img * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        return img

    def _to_torch(self, img: NDArray | Tensor, device: str | torch.device) -> Tensor:
        if isinstance(img, np.ndarray):
            img = to_tensor(img).to(device=device)
        return img.to(device=device)

    def _clear_prompts(self) -> None:
        """Clear all prompt buffers."""
        self._point_prompts.clear()
        self._point_prompts.clear()
        self._box_prompt = None
        self._mask_prompt = None
