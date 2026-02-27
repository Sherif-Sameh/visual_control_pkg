from __future__ import annotations

from typing import Any


def apply_overrides_dict(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply overrides to a configuration dictionary from given dictionary.

    Args:
        cfg: Configuration dictionary to apply given overrides to.
        overrides: Dictionary of overrides to apply to configuration.

    Returns:
        Modified configuration with applied overrides.
    """
    for name, value in cfg.items():
        if name not in overrides:
            continue
        cfg[name] = overrides[name]
    return cfg
