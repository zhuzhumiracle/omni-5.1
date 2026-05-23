#!/usr/bin/env python3
"""LC-gate real-tree density/speed sweep entrypoint.

This thin wrapper keeps the naming parallel to
contrast_with_sota_realtree_lidar/realtree_sweep_lidar_eval.py while reusing
the camera-gated camera+LiDAR implementation.
"""

from realtree_sweep_camlidar_gate_eval import main


if __name__ == "__main__":
    raise SystemExit(main())
