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
    for name in cfg.keys():
        if name not in overrides:
            continue
        cfg[name] = overrides[name]
    return cfg


def process_wandb_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Process a WandB hyperparameter configuration and returned processed configuration.

    This function combines any list parameters that are separated in the input config back into a
    list with all its elements. It goes through all the entries of the input config and checks if
    any keys contain a `.idx_` string literal, which indicates an element or range of elements
    belonging to a specific list parameter. If any parameters match this condition, they're
    processed in order to be combined back into a single list.

    Args:
        cfg: Configuration dictionary from WandB to process.

    Returns:
        Processed configuration with combined list parameters from the input configuration if they
            exist.

    Examples:
        >>> cfg = {"a": 0.5, "b": 1, "c.idx_01": 0.0, "c.idx_2": 1.5, "c.idx_46": -1.0}
        >>> process_wandb_config(cfg)
        {'a': 0.5, 'b': 1, 'c': [0.0, 0.0, 1.5, -1.0, -1.0, -1.0]}
    """
    out = {}
    for name, value in cfg.items():
        if ".idx_" not in name:
            out[name] = value
            continue

        pname, idx = name.split(".idx_")
        if pname not in out:
            out[pname] = []
        assert len(idx) in [1, 2]
        for i in range(int(idx[0]), int(idx[1]) + 1 if len(idx) > 1 else int(idx[0]) + 1):
            out[pname].insert(i, value)
    return out
