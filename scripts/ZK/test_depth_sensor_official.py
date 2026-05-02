import argparse
import os
import traceback

from omegaconf import OmegaConf

from omni_drones import init_simulation_app


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


def _create_scene():
    import isaacsim.core.utils.prims as prim_utils
    from pxr import Gf, UsdGeom, UsdLux
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    light = UsdLux.DistantLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(4000.0)
    prim_utils.create_prim("/World/TestObjects", "Xform")
    cube_specs = [
        ("/World/TestObjects/CubeNear", (2.5, 0.0, 1.0), (0.8, 0.8, 0.8)),
        ("/World/TestObjects/CubeFar", (4.5, 0.6, 1.2), (1.2, 1.2, 1.2)),
        ("/World/TestObjects/CubeSide", (3.2, -1.2, 0.8), (0.6, 0.6, 0.6)),
    ]
    for prim_path, translation, scale in cube_specs:
        prim = prim_utils.create_prim(
            prim_path,
            "Cube",
            translation=translation,
            scale=scale,
        )
        cube = UsdGeom.Cube(prim)
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([Gf.Vec3f(0.2, 0.6, 0.9)])


def _print_tensor_stats(name, tensor):
    import torch

    finite_mask = torch.isfinite(tensor)
    finite_ratio = finite_mask.float().mean().item()
    finite_values = tensor[finite_mask]
    if finite_values.numel() == 0:
        print(f"{name}: shape={tuple(tensor.shape)} finite_ratio={finite_ratio:.4f} no finite values")
        return
    print(
        f"{name}: shape={tuple(tensor.shape)} finite_ratio={finite_ratio:.4f} "
        f"min={finite_values.min().item():.4f} max={finite_values.max().item():.4f} "
        f"mean={finite_values.mean().item():.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Smoke test for DepthSensorOfficial.")
    parser.add_argument("--gui", action="store_true", help="Run with a GUI viewport when DISPLAY is available.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch/PhysX device.")
    parser.add_argument("--active-gpu", type=int, default=0, help="Physical Vulkan GPU index for Kit.")
    parser.add_argument("--physics-gpu", type=int, default=0, help="CUDA ordinal for PhysX.")
    parser.add_argument("--dt", type=float, default=0.02, help="Physics/render step size.")
    parser.add_argument("--steps", type=int, default=8, help="Number of frames to sample.")
    parser.add_argument("--width", type=int, default=640, help="Depth image width.")
    parser.add_argument("--height", type=int, default=480, help="Depth image height.")
    parser.add_argument(
        "--data-types",
        nargs="+",
        # default=["depth_sensor_distance", "distance_to_image_plane"],
        default=[ "distance_to_image_plane"],
        help="Depth annotators to request.",
    )
    args = parser.parse_args()

    cfg = _build_launch_cfg(args)
    print("Launching Isaac Sim for DepthSensorOfficial smoke test...")
    print(OmegaConf.to_yaml(cfg))
    simulation_app = init_simulation_app(cfg)

    try:
        print("[1/6] Importing Isaac Sim and OmniDrones camera classes...")
        try:
            from isaacsim.core.api.simulation_context import SimulationContext
        except ImportError:
            from omni.isaac.core.simulation_context import SimulationContext

        from omni_drones.sensors.camera_official import DepthSensorOfficial, DepthSensorOfficialCfg
        from omni_drones.sensors.config import PinholeCameraCfg

        print("[2/6] Creating SimulationContext...")
        sim = SimulationContext(
            stage_units_in_meters=1.0,
            physics_dt=args.dt,
            rendering_dt=args.dt,
            sim_params=cfg.sim,
            backend="torch",
            device=args.device,
        )
        if bool(cfg.sim.enable_viewport):
            sim.set_camera_view(eye=[6.0, 4.0, 3.5], target=[2.8, 0.0, 1.0])

        print("[3/6] Creating simple test scene...")
        _create_scene()

        print("[4/6] Spawning DepthSensorOfficial...")
        depth_cfg = DepthSensorOfficialCfg(
            sensor_tick=args.dt,
            resolution=(args.width, args.height),
            data_types=list(args.data_types),
            usd_params=PinholeCameraCfg.UsdCameraCfg(
                focal_length=12.0,
                focus_distance=20.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 20.0),
            ),
            baseline_mm=95.0,
            warmup_renders=20,
            read_retries=20,
        )

        sensor = DepthSensorOfficial(depth_cfg)
        sensor.spawn(
            ["/World/TestRig/DepthCamera"],
            translations=[(0.0, 0.0, 1.5)],
            targets=[(3.0, 0.0, 1.0)],
        )

        print("[5/6] Resetting sim and initializing depth sensor...")
        sim.reset()
        sensor.initialize()

        print("DepthSensorOfficial initialized.")
        print(f"Prim paths: {sensor.prim_paths}")
        print(f"Configured data types: {depth_cfg.data_types}")

        # Debug: inspect the raw frame structure after initialization.
        print("\n--- Post-init frame debug ---")
        for ds in sensor.depth_sensors:
            print(f"Sensor: {sensor._describe_sensor(ds)}")
            for accessor_name, accessor in [
                ("get_current_frame(clone=False)", lambda: ds.get_current_frame(clone=False)),
                ("get_current_frame()", lambda: ds.get_current_frame()),
                ("_current_frame", lambda: getattr(ds, "_current_frame", None)),
            ]:
                try:
                    result = accessor()
                    if isinstance(result, dict):
                        print(f"  {accessor_name}: dict with keys={list(result.keys())}")
                        for k, v in result.items():
                            if isinstance(v, dict) and "data" in v:
                                data_val = v["data"]
                                v_desc = f"dict[data={type(data_val).__name__}, shape={getattr(data_val, 'shape', 'N/A')}]"
                            elif v is None:
                                v_desc = "None"
                            elif hasattr(v, 'shape'):
                                v_desc = f"{type(v).__name__}(shape={v.shape})"
                            else:
                                v_desc = f"{type(v).__name__}"
                            print(f"    {k}: {v_desc}")
                    else:
                        print(f"  {accessor_name}: {type(result).__name__} (not a dict)")
                except Exception as exc:
                    print(f"  {accessor_name}: ERROR - {exc}")

        print("[6/6] Sampling depth frames...")
        for step in range(args.steps):
            sim.step(render=True)
            sensor.update()  # explicitly step the sensors before reading
            images = sensor.get_images()
            print(f"\nStep {step}:")
            for key in depth_cfg.data_types:
                _print_tensor_stats(key, images[key])
    except Exception:
        print("DepthSensorOfficial smoke test failed with exception:")
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
