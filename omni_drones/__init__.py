# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import logging
import os

import torch
from tensordict import TensorDict

CONFIG_PATH = os.path.join(os.path.dirname(__file__), os.path.pardir, "cfg")

def init_simulation_app(cfg):
    # launch the simulator
    sim_cfg = cfg.get("sim", {}) if hasattr(cfg, "get") else {}
    headless = bool(cfg["headless"])

    # Safety fallback: if DISPLAY is absent, force headless to avoid Vulkan swapchain failures.
    display_env = os.environ.get("DISPLAY", "").strip()
    force_headless_no_display = bool(cfg.get("force_headless_no_display", True)) if hasattr(cfg, "get") else True
    if force_headless_no_display and (not display_env):
        headless = True

    active_gpu = int(sim_cfg.get("active_gpu", 0))
    physics_gpu = int(sim_cfg.get("physics_gpu", 0))

    # When CUDA_VISIBLE_DEVICES is set, CUDA/PyTorch only see the listed GPUs
    # re-indexed from 0.  Omniverse's Vulkan renderer still enumerates ALL
    # physical GPUs, so active_gpu must use the *physical* Vulkan index while
    # physics_gpu (PhysX CUDA) must always be 0 (the first CUDA-visible device).
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    # if visible_devices:
    #     first_token = visible_devices.split(",")[0].strip()
    #     try:
    #         vulkan_gpu = int(first_token)          # physical GPU id for Vulkan
    #         if "active_gpu" not in sim_cfg:
    #             active_gpu = vulkan_gpu             # Vulkan uses physical index
    #         # PhysX uses CUDA ordinal — always 0 when CUDA_VISIBLE_DEVICES is set
    #         if "physics_gpu" not in sim_cfg:
    #             physics_gpu = 0
    #         logging.warning(
    #             "Detected CUDA_VISIBLE_DEVICES=%s, auto-align active_gpu=%d (Vulkan) "
    #             "physics_gpu=%d (CUDA ordinal) for SimulationApp.",
    #             visible_devices,
    #             active_gpu,
    #             physics_gpu,
    #         )
    if visible_devices:
        first_token = visible_devices.split(",")[0].strip()
        try:
            vulkan_gpu = int(first_token)

            if headless:
                active_gpu = vulkan_gpu
                physics_gpu = 0
                if str(sim_cfg.get("device", "")).startswith("cuda"):
                    sim_cfg["device"] = "cuda:0"
                logging.warning(
                    "Headless mode with CUDA_VISIBLE_DEVICES=%s -> active_gpu=%d (physical Vulkan GPU), "
                    "physics_gpu=%d (CUDA ordinal), sim.device=%s.",
                    visible_devices, active_gpu, physics_gpu, sim_cfg.get("device", "unset")
                )
            else:
                if "physics_gpu" not in sim_cfg:
                    physics_gpu = 0
                if str(sim_cfg.get("device", "")).startswith("cuda"):
                    sim_cfg["device"] = "cuda:0"
                logging.warning(
                    "Windowed mode with CUDA_VISIBLE_DEVICES=%s -> keep active_gpu=%d, "
                    "physics_gpu=%d, sim.device=%s.",
                    visible_devices, active_gpu, physics_gpu, sim_cfg.get("device", "unset")
                )
        except ValueError:
            logging.warning(
                "CUDA_VISIBLE_DEVICES=%s is not numeric; keep active_gpu=%d physics_gpu=%d.",
                visible_devices,
                active_gpu,
                physics_gpu,
            )

    anti_aliasing = int(sim_cfg.get("anti_aliasing", 1))
    config = {
        "headless": headless,
        "anti_aliasing": anti_aliasing,
        # Default to single-GPU renderer for stability on mixed display/compute multi-GPU hosts.
        "multi_gpu": bool(sim_cfg.get("multi_gpu", False)),
        "active_gpu": active_gpu,
        "physics_gpu": physics_gpu,
    }

    # Depth render products must keep color and depth textures at exactly the
    # requested resolution.  In Isaac Sim 5.x, disabling AA in SimulationApp
    # alone can still leave RTX/DLSS defaults active while Replicator render
    # products are created, so push the same settings through Kit startup args.
    if anti_aliasing == 0 and bool(sim_cfg.get("disable_dlss_for_depth", True)):
        depth_render_args = [
            "--/rtx/post/aa/op=0",
            "--/rtx-defaults/post/aa/op=0",
            "--/rtx-transient/post/aa/limitedOps=false",
            "--/rtx-transient/dlssg/enabled=false",
            "--/rtx/post/dlss/manualScaling=1.0",
            "--/rtx-defaults/post/dlss/manualScaling=1.0",
            "--/rtx/index/resolutionScale=1.0",
            "--/rtx/descriptorSets=60000",
            "--/rtx/reservedDescriptors=500000",
            "--/app/viewport/defaults/resolutionScale=1.0",
            "--/persistent/app/viewport/defaults/resolutionScale=1.0",
        ]
        user_extra_args = list(sim_cfg.get("extra_args", []))
        config["extra_args"] = depth_render_args + user_extra_args
    
    # Always set the RTX renderer when Replicator is enabled or a renderer is explicitly requested.
    # RTX post-processing (e.g., depth sensor) requires the RTX renderer even in headless mode.
    enable_replicator = bool(sim_cfg.get("enable_replicator", False))
    enable_viewport = bool(sim_cfg.get("enable_viewport", False))
    if enable_replicator or enable_viewport:
        config["renderer"] = "RayTracedLighting"
    if "enable_replicator" in sim_cfg:
        config["enable_replicator"] = enable_replicator
    
    from isaacsim import SimulationApp
    simulation_app = SimulationApp(config)
    return simulation_app

def _get_shapes(self: TensorDict):
    return {
        k: v.shape if isinstance(v, torch.Tensor) else v.shapes for k, v in self.items()
    }


def _get_devices(self: TensorDict):
    return {
        k: v.device if isinstance(v, torch.Tensor) else v.devices
        for k, v in self.items()
    }


TensorDict.shapes = property(_get_shapes)
TensorDict.devices = property(_get_devices)
