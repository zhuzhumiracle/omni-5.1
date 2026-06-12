#!/usr/bin/env python3
"""Legacy-compatible real-tree eval for old LiDAE-KU/LC checkpoints.

The checkpoint goodpt/5-16-vlim-lc-tree_best_return_2460.18.pt was trained with
the older 3-bin camera-risk layout:

    camera_risk_dim = 3 bins * 4 features + stale_ratio = 13

Current forest_lc.yaml defaults to a 4x4 camera-risk grid:

    camera_risk_dim = 4 rows * 4 cols * 4 features + stale_ratio = 65

That changes the first camera-risk encoder layer shape and makes the checkpoint
look incompatible.  This wrapper keeps the real-tree eval code unchanged, but
prepends the legacy camera-risk overrides before delegating to it.
"""

from __future__ import annotations

import sys

from realtree_sweep_camlidar_eval import main as _base_main


LEGACY_LC_OVERRIDES = [
    "task.camera_risk_num_rows=0",
    "task.camera_risk_num_cols=0",
    "task.camera_risk_num_bins=3",
    "task.camera_risk_features_per_bin=4",
    "task.camera_risk_add_stale_ratio=true",
    "task.camera_risk_gate_alpha=0.2",
    "task.camera_risk_fusion_mode=gate",
]


def main(argv: list[str] | None = None) -> int:
    user_argv = list(sys.argv[1:] if argv is None else argv)
    existing_keys = {
        token.split("=", 1)[0].lstrip("+")
        for token in user_argv
        if "=" in token and not token.startswith("--")
    }
    compat_overrides = [
        token
        for token in LEGACY_LC_OVERRIDES
        if token.split("=", 1)[0].lstrip("+") not in existing_keys
    ]
    return int(_base_main(compat_overrides + user_argv) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
