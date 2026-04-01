from dataclasses import dataclass


@dataclass
class SAMPromptConfig:
    """Configuration for collecting prompts for the SAM-like model."""

    n_pos: int = 1
    """Number of positive points to expect for point prompts. Default value is 1."""

    n_neg: int = 0
    """Number of negative points to expect for point prompts. Default value is 0."""

    n_mask_pts: int = 4
    """Number of points to expect for polygon for mask prompts. Default value is 4."""
