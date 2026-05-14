#!/usr/bin/env python3
"""
验证 DepthSensorOfficial 在 forest_lc 环境中的集成是否正确。

用法:
  cd OmniDrones/scripts/ZK
  python test_forest_lc_depth_integration.py --steps 10
"""

import argparse
import logging
import os

from omegaconf import OmegaConf
from omni_drones import init_simulation_app

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

logging.basicConfig(level=logging.INFO)


def _build_launch_cfg(args):
    headless = not bool(args.gui)
    enable_viewport = bool(args.gui) and bool(os.environ.get("DISPLAY", "").strip())
    return OmegaConf.create(
        {
            "headless": headless,
            "force_headless_no_display": True,
            "sim": {
                "device": args.device,
                "dt": args.dt,
                "substeps": 1,
                "enable_viewport": enable_viewport,
                "enable_replicator": True,
                "anti_aliasing": 0,
                "multi_gpu": False,
                "active_gpu": args.active_gpu,
                "physics_gpu": args.physics_gpu,
            },
        }
    )


def _print_tensor_stats(name, tensor):
    import torch

    finite_mask = torch.isfinite(tensor)
    finite_ratio = finite_mask.float().mean().item()
    finite_values = tensor[finite_mask]
    if finite_values.numel() == 0:
        print(f"  {name}: shape={tuple(tensor.shape)} finite_ratio={finite_ratio:.4f} no finite values")
        return
    non_zero = (finite_values > 1e-6).float().mean().item()
    print(
        f"  {name}: shape={tuple(tensor.shape)} finite={finite_ratio:.4f} "
        f"nonzero={non_zero:.4f} "
        f"min={finite_values.min().item():.4f} max={finite_values.max().item():.4f} "
        f"mean={finite_values.mean().item():.4f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--active-gpu", type=int, default=0)
    parser.add_argument("--physics-gpu", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=96)
    args = parser.parse_args()

    cfg = _build_launch_cfg(args)
    print("=" * 60)
    print("Launching Isaac Sim for forest_lc depth camera integration test...")
    print("=" * 60)
    simulation_app = init_simulation_app(cfg)

    try:
        import torch
        from isaacsim.core.api.simulation_context import SimulationContext
        from omni_drones.robots.drone import MultirotorBase
        from omni_drones.sensors.camera_official import DepthSensorOfficial, DepthSensorOfficialCfg
        from omni_drones.sensors.config import PinholeCameraCfg
        import isaacsim.core.utils.prims as prim_utils

        print("\n[1/5] Creating SimulationContext...")
        sim = SimulationContext(
            stage_units_in_meters=1.0,
            physics_dt=args.dt,
            rendering_dt=args.dt,
            sim_params=cfg.sim,
            backend="torch",
            device=args.device,
        )

        print("[2/5] Creating a single drone (mimics forest_lc setup)...")
        drone, controller = MultirotorBase.make("Hummingbird", "LeePositionController")
        drone.spawn(translations=[(0.0, 0.0, 2.0)])

        # Create simple obstacles for depth verification
        prim_utils.create_prim("/World/TestObjects", "Xform")
        from pxr import Gf, UsdGeom
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        from pxr import UsdLux
        light = UsdLux.DistantLight.Define(stage, "/World/Light")
        light.CreateIntensityAttr(4000.0)

        cube_specs = [
            ("/World/TestObjects/CubeNear", (3.0, 2.0, 1.0), (1.0, 1.0, 1.0)),
            ("/World/TestObjects/CubeFar", (6.0, 3.0, 1.5), (1.5, 1.5, 1.5)),
        ]
        for prim_path, translation, scale in cube_specs:
            prim = prim_utils.create_prim(prim_path, "Cube", translation=translation, scale=scale)
            cube = UsdGeom.Cube(prim)
            cube.CreateSizeAttr(1.0)
            cube.CreateDisplayColorAttr([Gf.Vec3f(0.3, 0.7, 0.3)])

        print("[3/5] Setting up DepthSensorOfficial (forest_lc config)...")
        depth_cfg = DepthSensorOfficialCfg(
            sensor_tick=0,
            resolution=(args.width, args.height),
            data_types=["distance_to_camera"],
            usd_params=PinholeCameraCfg.UsdCameraCfg(
                focal_length=12.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 10.0),
            ),
            warmup_renders=10,
            read_retries=10,
        )

        sensor = DepthSensorOfficial(depth_cfg)
        sensor.spawn(
            [f"/World/envs/env_0/{drone.name}_0/base_link/DepthCamera"],
            translations=[(0.22, 0.0, 0.18)],
            targets=[(2.0, 0.0, 0.18)],
        )

        print("[4/5] Initializing (mimics forest_lc.__init__ flow)...")
        sim.reset()
        drone.initialize()
        sensor.initialize(
            f"/World/envs/env_.*/{drone.name}_0/base_link/DepthCamera"
        )

        print(f"  Sensor prim paths: {sensor.prim_paths}")
        print(f"  Configured data types: {depth_cfg.data_types}")
        print(f"  Resolution: {depth_cfg.resolution}")

        print("\n[5/5] Running simulation steps and checking depth output...")
        print("-" * 60)

        for step in range(args.steps):
            sim.step(render=True)
            sensor.update()
            images = sensor.get_images()

            depth = images["distance_to_camera"]
            # Mimic forest_lc processing: [E, 1, H, W] or [1, H, W]
            if depth.ndim == 3:
                depth = depth.unsqueeze(0)

            print(f"\nStep {step}:")
            _print_tensor_stats("distance_to_camera", depth)

            # Validation checks
            if depth.numel() == 0:
                raise RuntimeError("FAIL: depth tensor is empty!")
            if depth.max().item() <= 0:
                raise RuntimeError(f"FAIL: depth is all zeros (max={depth.max().item()})!")

            # Check that not all values are the same (indicates stale frame)
            unique_vals = torch.unique(depth.flatten())
            if len(unique_vals) < 5:
                logging.warning(
                    "  ⚠ Depth frame has very few unique values (%d) — may be stale.", len(unique_vals)
                )

        print("\n" + "=" * 60)
        print("✅ ALL CHECKS PASSED — DepthSensorOfficial works correctly in forest_lc context.")
        print("=" * 60)

    except Exception:
        import traceback
        print("\n" + "=" * 60)
        print("❌ INTEGRATION TEST FAILED")
        print("=" * 60)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
