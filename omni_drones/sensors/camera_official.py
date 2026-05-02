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

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping, Optional, Sequence

import torch
from pxr import Gf, Sdf, UsdGeom
from tensordict import TensorDict

from .config import PinholeCameraCfg


@dataclass
class OpenCvPinholeCfg:
    """OpenCV pinhole calibration parameters for official Isaac Sim camera APIs."""

    fx: float
    fy: float
    cx: float
    cy: float
    pinhole: tuple[float, ...] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )

# @dataclass
# class DepthSensorOfficialCfg:
#     sensor_tick = 相机多久更新一次
#     resolution = 图像分辨率
#     data_types = 要输出哪些图像/深度数据
#     usd_params = USD 相机光学参数
#     baseline_mm = 深度传感器基线距离
#     opencv_pinhole = OpenCV 相机标定参数
#     depth_sensor_attributes = 额外的 depth sensor 底层属性
#     warmup_renders = 初始化后先渲染几帧
#     read_retries = 读取失败最多重试几次
@dataclass
class DepthSensorOfficialCfg:
    """Configuration for the official Isaac Sim single-view depth sensor wrapper."""
    # 深度相机的更新频率
    sensor_tick: float = 0.1
    resolution: tuple[int, int] = (640, 480)
    data_types: list[str] = field(
        default_factory=lambda: ["distance_to_image_plane"]
    )
    usd_params: PinholeCameraCfg.UsdCameraCfg = field(default_factory=PinholeCameraCfg.UsdCameraCfg)
    baseline_mm: Optional[float] = None
    opencv_pinhole: Optional[OpenCvPinholeCfg] = None
    depth_sensor_attributes: dict[str, Any] = field(default_factory=dict)
    warmup_renders: int = 5
    read_retries: int = 5
    disable_dlss_for_depth: bool = True


class DepthSensorOfficial:
    """
    Official Isaac Sim single-view depth sensor wrapper.

    This class is designed to keep a similar call pattern to the existing
    ``Camera`` wrapper in this repo:

    - ``spawn(...)`` creates USD ``Camera`` prims
    - ``initialize(...)`` wraps them with
      ``isaacsim.sensors.camera.SingleViewDepthSensor``
    - ``get_images()`` returns a channel-first ``TensorDict`` stack

    Notes:
    - The implementation intentionally lazy-imports Isaac Sim APIs so the file
      can be imported from a plain Python environment without crashing.
    - The exact ``SingleViewDepthSensor`` runtime API is discovered
      defensively because it is not available in this shell environment.
    """

    _ANNOTATOR_ALIASES = {
        "depth_sensor_distance": "DepthSensorDistance",
        "distance_to_image_plane": "distance_to_image_plane",
        "distance_to_camera": "distance_to_camera",
        "rgb": "rgb",
        "rgba": "rgba",
    }

    def __init__(self, cfg: Optional[DepthSensorOfficialCfg] = None) -> None:
        if cfg is None:
            cfg = DepthSensorOfficialCfg(
                sensor_tick=0.0,
                resolution=(640, 480),
                usd_params=PinholeCameraCfg.UsdCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.1, 1.0e5),
                ),
            )
        self.cfg = cfg
        self.resolution = cfg.resolution
        self.shape = (self.resolution[1], self.resolution[0])
        self.sim = self._get_simulation_context()
        self.device = self.sim.device
        if isinstance(self.device, str) and "cuda" in self.device:
            self.device = self.device.split(":")[0]

        self.prim_paths: list[str] = []
        self.depth_sensors: list[Any] = []
        self.count = 0
        if self.cfg.disable_dlss_for_depth:
            self._configure_full_resolution_depth_rendering()

    def spawn(
        self,
        prim_paths: Sequence[str],
        translations=None,
        targets=None,
    ) -> None:
        prim_utils = self._import_prim_utils()
        n = len(prim_paths)
        self.prim_paths = list(prim_paths)

        if translations is None:
            translations = [(0.0, 0.0, 0.0) for _ in range(n)]
        translations = torch.atleast_2d(torch.as_tensor(translations)).expand(n, 3).tolist()

        if targets is None:
            targets = [(1.0, 0.0, 0.0) for _ in range(n)]
        targets = torch.atleast_2d(torch.as_tensor(targets)).expand(n, 3).tolist()

        if not len(translations) == len(prim_paths) == len(targets):
            raise ValueError("prim_paths, translations, and targets must have the same length.")

        for prim_path, translation, target in zip(prim_paths, translations, targets):
            if prim_utils.is_prim_path_valid(prim_path):
                raise RuntimeError(f"Duplicate prim at {prim_path}.")
            prim_utils.create_prim(
                prim_path,
                prim_type="Camera",
                translation=translation,
                orientation=orientation_from_view(translation, target),
            )
            self._define_usd_camera_attributes(prim_path)

    def initialize(self, prim_paths_expr: Optional[str] = None) -> None:
        prim_utils = self._import_prim_utils()
        depth_sensor_cls = self._import_sensor_class()
        self.depth_sensors = []

        if prim_paths_expr is None and len(self.prim_paths) > 0:
            prim_paths = list(self.prim_paths)
        else:
            if prim_paths_expr is None:
                prim_paths_expr = r"/World/envs/.*/.*Camera.*"
            prim_paths = prim_utils.find_matching_prim_paths(prim_paths_expr)
            prim_paths = sorted(prim_paths, key=self._prim_path_sort_key)
        if len(prim_paths) == 0:
            raise RuntimeError(f"No camera prims matched expression: {prim_paths_expr}")
        self.prim_paths = list(prim_paths)

        for prim_path in prim_paths:
            depth_sensor = self._construct_depth_sensor(depth_sensor_cls, prim_path)
            if hasattr(depth_sensor, "initialize"):
                try:
                    depth_sensor.initialize(attach_rgb_annotator=False)
                except TypeError:
                    depth_sensor.initialize()
            if self.cfg.disable_dlss_for_depth:
                self._configure_full_resolution_depth_rendering()
                self._apply_render_product_full_resolution_settings(depth_sensor)
            self._maybe_apply_opencv_pinhole(depth_sensor)
            self._apply_depth_sensor_attributes(depth_sensor)
            self._attach_requested_annotators(depth_sensor)
            if self.cfg.disable_dlss_for_depth:
                self._configure_full_resolution_depth_rendering()
                self._apply_render_product_full_resolution_settings(depth_sensor)
            self.depth_sensors.append(depth_sensor)

        self.count = len(self.depth_sensors)
        for i in range(max(1, int(self.cfg.warmup_renders))):
            for depth_sensor in self.depth_sensors:
                self._step_sensor(depth_sensor, render=False)
            self._render_once()
        # Post-warmup verification: ensure each sensor can produce at least one valid frame.
        for depth_sensor in self.depth_sensors:
            for annotator_type in self.cfg.data_types:
                try:
                    img = self._read_annotator_tensor_with_retry(depth_sensor, annotator_type)
                    if img.numel() == 0:
                        raise RuntimeError(
                            f"After warmup, annotator '{annotator_type}' on "
                            f"'{self._describe_sensor(depth_sensor)}' still returns empty."
                        )
                except Exception as exc:
                    raise RuntimeError(
                        f"Depth sensor '{self._describe_sensor(depth_sensor)}' failed post-warmup "
                        f"verification for annotator '{annotator_type}'. "
                        f"The sensor may need a different render setup (e.g., a visible viewport, "
                        f"or explicit render product resolution)."
                    ) from exc

    def update(self, dt=None) -> None:
        for depth_sensor in self.depth_sensors:
            self._step_sensor(depth_sensor, render=False)
        self._render_once()

    def get_images(self) -> TensorDict:
        if len(self.depth_sensors) == 0:
            raise RuntimeError("DepthSensorOfficial has not been initialized yet.")

        images_list = []
        for depth_sensor in self.depth_sensors:
            images_dict = {}
            for annotator_type in self.cfg.data_types:
                img_tensor = self._read_annotator_tensor_with_retry(depth_sensor, annotator_type)
                images_dict[annotator_type] = self._format_annotator_tensor(img_tensor, annotator_type)
            images_list.append(TensorDict(images_dict, []))
        return torch.stack(images_list)

    def _construct_depth_sensor(self, depth_sensor_cls, prim_path: str):
        kwargs = {
            "prim_path": prim_path,
            "name": prim_path.rsplit("/", 1)[-1],
            "resolution": self.resolution,
        }
        if self.cfg.sensor_tick and self.cfg.sensor_tick > 0.0:
            kwargs["dt"] = self.cfg.sensor_tick

        constructor_attempts = (
            kwargs,
            {k: v for k, v in kwargs.items() if k != "name"},
            {"prim_path": prim_path, "resolution": self.resolution},
            {"prim_path": prim_path},
        )
        last_error = None
        for attempt_kwargs in constructor_attempts:
            try:
                return depth_sensor_cls(**attempt_kwargs)
            except TypeError as exc:
                last_error = exc
        raise RuntimeError(
            f"Failed to construct SingleViewDepthSensor for '{prim_path}'. "
            f"Last constructor error: {last_error}"
        )

    def _attach_requested_annotators(self, depth_sensor) -> None:
        for annotator_type in self.cfg.data_types:
            official_name = self._to_official_annotator_name(annotator_type)
            try:
                depth_sensor.attach_annotator(official_name)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to attach annotator '{official_name}' to depth sensor "
                    f"at '{self._describe_sensor(depth_sensor)}'."
                ) from exc

    def _read_sensor_frame(self, depth_sensor) -> Mapping[str, Any]:
        access_attempts = (
            lambda: depth_sensor.get_current_frame(clone=False),
            lambda: depth_sensor.get_current_frame(),
            lambda: getattr(depth_sensor, "_current_frame"),
        )
        for accessor in access_attempts:
            try:
                frame = accessor()
            except Exception:
                continue
            if isinstance(frame, Mapping) and len(frame) > 0:
                # Validate that at least one annotator has non-None data.
                has_valid = False
                for key, value in frame.items():
                    actual = value
                    if isinstance(actual, Mapping) and "data" in actual:
                        actual = actual["data"]
                    if actual is not None:
                        has_valid = True
                        break
                if has_valid:
                    return frame
        raise RuntimeError(
            f"Unable to read valid annotator outputs from depth sensor '{self._describe_sensor(depth_sensor)}'. "
            "Expected get_current_frame() or _current_frame to provide a non-empty mapping with valid data."
        )

    def _step_sensor(self, depth_sensor, render: bool = True) -> None:
        """Trigger a render step for the depth sensor using all available methods."""
        # First, try the sensor's own update/step methods.
        for method_name in ("update", "step", "render"):
            method = getattr(depth_sensor, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass
        if render:
            self._render_once()

    def _render_once(self) -> None:
        """Render once for the whole depth-sensor batch."""
        self.sim.render()

    def _configure_full_resolution_depth_rendering(self) -> None:
        """Avoid DLSS internal downscaling so color/depth textures stay the same size."""
        try:
            import carb
            import os

            settings = carb.settings.get_settings()
            # 0 disables RTX post AA. `limitedOps=false` is still needed in
            # Isaac Sim 5.1; otherwise the RTX post stack may keep DLSS's
            # small-input enforcement alive for tiny render products.
            settings.set("/rtx/post/aa/op", 0)
            settings.set("rtx/post/aa/op", 0)
            settings.set("/rtx-defaults/post/aa/op", 0)
            settings.set("rtx-defaults/post/aa/op", 0)
            settings.set("/rtx-transient/post/aa/limitedOps", False)
            settings.set("rtx-transient/post/aa/limitedOps", False)
            settings.set("/rtx-transient/dlssg/enabled", False)
            settings.set("rtx-transient/dlssg/enabled", False)
            settings.set("/rtx/post/dlss/manualScaling", 1.0)
            settings.set("rtx/post/dlss/manualScaling", 1.0)
            settings.set("/rtx-defaults/post/dlss/manualScaling", 1.0)
            settings.set("rtx-defaults/post/dlss/manualScaling", 1.0)
            settings.set("/rtx/index/resolutionScale", 1.0)
            settings.set("rtx/index/resolutionScale", 1.0)
            settings.set("/app/viewport/defaults/resolutionScale", 1.0)
            settings.set("app/viewport/defaults/resolutionScale", 1.0)
            settings.set("/persistent/app/viewport/defaults/resolutionScale", 1.0)
            settings.set("persistent/app/viewport/defaults/resolutionScale", 1.0)
            # The RTX DepthSensor AOV needs USD/Fabric render settings, but
            # standard Replicator depth annotators should stay on global carb
            # settings. Reading every USD render-product schema can reintroduce
            # the schema default "dlss" op on hidden render products.
            use_render_setting_attrs = self._requires_depth_sensor_api()
            settings.set("/app/hydra/renderSettings/useUsdAttributes", use_render_setting_attrs)
            settings.set("/app/hydra/renderSettings/useFabricAttributes", use_render_setting_attrs)
            if os.environ.get("OMNI_DRONES_DEBUG_DEPTH_SETTINGS") == "1":
                carb.log_warn(
                    "[DepthSensorOfficial] carb render settings: "
                    f"aa/op={settings.get('/rtx/post/aa/op')!r}, "
                    f"aa/op(int)={settings.get_as_int('/rtx/post/aa/op')!r}, "
                    f"dlss/manualScaling={settings.get('/rtx/post/dlss/manualScaling')!r}, "
                    f"useUsdAttributes={settings.get('/app/hydra/renderSettings/useUsdAttributes')!r}, "
                    f"useFabricAttributes={settings.get('/app/hydra/renderSettings/useFabricAttributes')!r}"
                )
        except Exception:
            pass

    def _read_annotator_tensor_with_retry(self, depth_sensor, annotator_type: str) -> torch.Tensor:
        last_error = None
        retry_count = max(1, int(self.cfg.read_retries))

        # Map our internal annotator names to the names used by SingleViewDepthSensor.
        official_name = self._to_official_annotator_name(annotator_type)

        for attempt in range(retry_count):
            # Strategy A: try get_current_frame() first (fast cached path if callback fired).
            try:
                frame = self._read_sensor_frame(depth_sensor)
                img_tensor = self._extract_annotator_tensor(frame, annotator_type)
                if img_tensor.numel() > 0:
                    return img_tensor
            except Exception:
                pass

            # Strategy B: directly call get_data() on the underlying Replicator annotator.
            # This bypasses the async callback and works reliably in synchronous mode.
            try:
                custom_annotators = getattr(depth_sensor, "_custom_annotators", {})
                annotator = custom_annotators.get(official_name)
                if annotator is None:
                    # Try the raw annotator_type name as fallback.
                    annotator = custom_annotators.get(annotator_type)

                if annotator is not None and hasattr(annotator, "get_data"):
                    import warp as wp

                    # Try get_data with device first (GPU path), then without (CPU fallback).
                    raw = None
                    for get_data_kwargs in ({"device": self.device}, {}):
                        try:
                            raw = annotator.get_data(**get_data_kwargs)
                            if raw is not None:
                                break
                        except Exception:
                            continue

                    if raw is not None:
                        # get_data may return a dict with "data" key or raw array.
                        if isinstance(raw, dict) and "data" in raw:
                            raw = raw["data"]
                        if raw is not None:
                            try:
                                img_tensor = wp.to_torch(raw)
                            except Exception:
                                img_tensor = torch.as_tensor(raw, device=self.device)
                            if img_tensor.numel() > 0:
                                return img_tensor
            except Exception:
                pass

            # Strategy C: re-read frame after step (may have been updated by callback).
            self._step_sensor(depth_sensor)
            try:
                frame = self._read_sensor_frame(depth_sensor)
                img_tensor = self._extract_annotator_tensor(frame, annotator_type)
                if img_tensor.numel() > 0:
                    return img_tensor
            except Exception as exc:
                last_error = exc

        raise RuntimeError(
            f"Failed to read valid '{annotator_type}' data from depth sensor "
            f"'{self._describe_sensor(depth_sensor)}' after {retry_count} attempts."
        ) from last_error

    def _extract_annotator_tensor(self, frame: Mapping[str, Any], annotator_type: str) -> torch.Tensor:
        candidate_keys = []
        official_name = self._to_official_annotator_name(annotator_type)
        candidate_keys.extend(
            [
                annotator_type,
                official_name,
                official_name.lower(),
                official_name.upper(),
                official_name[0].lower() + official_name[1:] if official_name else official_name,
                self._to_snake_case(official_name),
            ]
        )
        for key in candidate_keys:
            if key in frame:
                return self._coerce_to_torch(frame[key], annotator_type)
        raise KeyError(
            f"Depth sensor frame does not contain '{annotator_type}' or '{official_name}'. "
            f"Available keys: {list(frame.keys())}"
        )

    def _coerce_to_torch(self, value: Any, annotator_type: str) -> torch.Tensor:
        if isinstance(value, Mapping) and "data" in value:
            value = value["data"]
        if value is None:
            raise RuntimeError(
                f"Annotator '{annotator_type}' output is not ready yet "
                f"(data is None — render may not have completed for this sensor)."
            )
        if torch.is_tensor(value):
            return value
        if hasattr(value, "__cuda_array_interface__") or hasattr(value, "__array_interface__"):
            try:
                return torch.as_tensor(value, device=self.device)
            except Exception:
                return torch.as_tensor(value)
        try:
            import numpy as np

            if isinstance(value, np.ndarray):
                return torch.from_numpy(value).to(self.device)
        except Exception:
            pass
        try:
            import warp as wp

            return wp.to_torch(value)
        except Exception:
            pass
        try:
            return torch.as_tensor(value, device=self.device)
        except Exception as exc:
            raise RuntimeError(
                f"Unable to convert annotator '{annotator_type}' output of type {type(value)} to torch."
            ) from exc

    def _format_annotator_tensor(self, img_tensor: torch.Tensor, annotator_type: str) -> torch.Tensor:
        """Normalize sensor output to channel-first [C, H, W]."""
        height, width = self.shape
        num_pixels = height * width

        if img_tensor.dim() == 1:
            if img_tensor.numel() == num_pixels:
                return img_tensor.reshape(1, height, width)
            if img_tensor.numel() > 0 and num_pixels > 0 and img_tensor.numel() % num_pixels == 0:
                channels = img_tensor.numel() // num_pixels
                return img_tensor.reshape(height, width, channels).permute(2, 0, 1)
            raise RuntimeError(
                f"Unexpected 1D annotator output for '{annotator_type}': "
                f"numel={img_tensor.numel()}, expected {num_pixels} or a multiple of it "
                f"for resolution={self.resolution}."
            )

        if img_tensor.dim() == 2:
            return img_tensor.unsqueeze(0)

        if img_tensor.dim() == 3:
            if img_tensor.shape[0] in (1, 3, 4) and img_tensor.shape[-2:] == (height, width):
                return img_tensor
            if img_tensor.shape[:2] == (height, width):
                return img_tensor.permute(2, 0, 1)

        if img_tensor.dim() == 4 and img_tensor.shape[0] == 1:
            return self._format_annotator_tensor(img_tensor[0], annotator_type)

        raise RuntimeError(
            f"Unexpected annotator output shape for '{annotator_type}': "
            f"shape={tuple(img_tensor.shape)}, resolution={self.resolution}."
        )

    def _define_usd_camera_attributes(self, prim_path: str) -> None:
        prim_utils = self._import_prim_utils()
        prim = prim_utils.get_prim_at_path(prim_path)
        camera = UsdGeom.Camera(prim)
        usd_params = self.cfg.usd_params

        if usd_params.clipping_range is not None:
            camera.GetClippingRangeAttr().Set(usd_params.clipping_range)
        if usd_params.focal_length is not None:
            camera.GetFocalLengthAttr().Set(usd_params.focal_length)
        if usd_params.focus_distance is not None:
            camera.GetFocusDistanceAttr().Set(usd_params.focus_distance)
        if usd_params.f_stop is not None:
            camera.GetFStopAttr().Set(usd_params.f_stop)
        if usd_params.horizontal_aperture is not None:
            camera.GetHorizontalApertureAttr().Set(usd_params.horizontal_aperture)

        vertical_aperture = getattr(usd_params, "vertical_aperture", None)
        if vertical_aperture is None and usd_params.horizontal_aperture is not None:
            vertical_aperture = (
                float(usd_params.horizontal_aperture) * float(self.resolution[1]) / float(max(1, self.resolution[0]))
            )
        if vertical_aperture is not None:
            camera.GetVerticalApertureAttr().Set(vertical_aperture)

        if usd_params.horizontal_aperture_offset is not None:
            camera.GetHorizontalApertureOffsetAttr().Set(usd_params.horizontal_aperture_offset)
        if usd_params.vertical_aperture_offset is not None:
            camera.GetVerticalApertureOffsetAttr().Set(usd_params.vertical_aperture_offset)

    def _maybe_apply_opencv_pinhole(self, depth_sensor) -> None:
        opencv_cfg = self.cfg.opencv_pinhole
        if opencv_cfg is None:
            return

        target = depth_sensor
        for attr_name in ("camera", "_camera", "wrapped_camera", "_wrapped_camera"):
            candidate = getattr(depth_sensor, attr_name, None)
            if candidate is not None:
                target = candidate
                break

        if not hasattr(target, "set_opencv_pinhole_properties"):
            raise RuntimeError(
                "Depth sensor was created successfully, but the wrapped official camera does not expose "
                "`set_opencv_pinhole_properties` in this Isaac Sim runtime."
            )

        kwargs = {
            "cx": opencv_cfg.cx,
            "cy": opencv_cfg.cy,
            "fx": opencv_cfg.fx,
            "fy": opencv_cfg.fy,
            "pinhole": opencv_cfg.pinhole,
        }
        target.set_opencv_pinhole_properties(**kwargs)

    def _apply_depth_sensor_attributes(self, depth_sensor) -> None:
        render_product_path = self._get_render_product_path(depth_sensor)
        if render_product_path is None:
            return

        prim_utils = self._import_prim_utils()
        prim = prim_utils.get_prim_at_path(render_product_path)
        if not prim or not prim.IsValid():
            return

        attr_values = dict(self.cfg.depth_sensor_attributes)
        if self.cfg.baseline_mm is not None:
            attr_values.setdefault("baselineMM", float(self.cfg.baseline_mm))

        for attr_name, attr_value in attr_values.items():
            full_name = attr_name
            if not full_name.startswith("omni:rtx:post:depthSensor:"):
                full_name = f"omni:rtx:post:depthSensor:{attr_name}"
            attribute = prim.GetAttribute(full_name)
            if not attribute.IsValid():
                attribute = prim.CreateAttribute(full_name, self._infer_sdf_type(attr_value))
            attribute.Set(attr_value)

    def _apply_render_product_full_resolution_settings(self, depth_sensor) -> None:
        render_product_path = self._get_render_product_path(depth_sensor)
        if render_product_path is None:
            return

        prim_utils = self._import_prim_utils()
        prim = prim_utils.get_prim_at_path(render_product_path)
        if not prim or not prim.IsValid():
            return

        for api_name in ("OmniRtxPostDebugSettingsAPI_1", "OmniRtxPostDebugSettingsAPI"):
            try:
                prim.ApplyAPI(api_name)
                break
            except Exception:
                pass

        # These are the USD render-product equivalents of the RTX carb settings.
        # The token value is important here: USD render settings use "none",
        # while the deprecated carb path accepts the integer 0.
        self._set_render_product_attr(prim, "omni:rtx:post:aa:op", Sdf.ValueTypeNames.Token, "none")
        self._set_render_product_attr(
            prim,
            "omni:rtx:post:dlss:manualScaling",
            Sdf.ValueTypeNames.Float,
            1.0,
        )
        self._set_render_product_attr(
            prim,
            "omni:rtx:dlss:frameGeneration",
            Sdf.ValueTypeNames.Bool,
            False,
        )
        self._debug_render_product_full_resolution_settings(render_product_path, prim)

    def _set_render_product_attr(self, prim, attr_name: str, sdf_type, value) -> None:
        attribute = prim.GetAttribute(attr_name)
        if not attribute.IsValid():
            attribute = prim.CreateAttribute(attr_name, sdf_type)
        attribute.Set(value)

    def _debug_render_product_full_resolution_settings(self, render_product_path: str, prim) -> None:
        try:
            import os

            if os.environ.get("OMNI_DRONES_DEBUG_DEPTH_SETTINGS") != "1":
                return
            import carb

            def _get_attr(attr_name: str):
                attribute = prim.GetAttribute(attr_name)
                return attribute.Get() if attribute.IsValid() else None

            carb.log_warn(
                "[DepthSensorOfficial] render product settings: "
                f"path={render_product_path}, "
                f"apiSchemas={prim.GetMetadata('apiSchemas')}, "
                f"aa/op={_get_attr('omni:rtx:post:aa:op')!r}, "
                f"dlss/manualScaling={_get_attr('omni:rtx:post:dlss:manualScaling')!r}, "
                f"dlss/frameGeneration={_get_attr('omni:rtx:dlss:frameGeneration')!r}"
            )
        except Exception:
            pass

    def _get_render_product_path(self, depth_sensor) -> Optional[str]:
        candidate_attrs = (
            "render_product_path",
            "_render_product_path",
            "render_product",
            "_render_product",
        )
        for attr_name in candidate_attrs:
            value = getattr(depth_sensor, attr_name, None)
            if isinstance(value, str) and value:
                return value
            if hasattr(value, "path") and isinstance(value.path, str):
                return value.path
        for method_name in ("get_render_product_path", "get_render_product"):
            method = getattr(depth_sensor, method_name, None)
            if callable(method):
                try:
                    value = method()
                except Exception:
                    continue
                if isinstance(value, str) and value:
                    return value
                if hasattr(value, "path") and isinstance(value.path, str):
                    return value.path
        return None

    def _infer_sdf_type(self, value):
        if isinstance(value, bool):
            return Sdf.ValueTypeNames.Bool
        if isinstance(value, int):
            return Sdf.ValueTypeNames.Int
        if isinstance(value, float):
            return Sdf.ValueTypeNames.Float
        if isinstance(value, str):
            return Sdf.ValueTypeNames.String
        raise TypeError(f"Unsupported depth sensor attribute value type: {type(value)}")

    def _to_official_annotator_name(self, annotator_type: str) -> str:
        return self._ANNOTATOR_ALIASES.get(annotator_type, annotator_type)

    def _to_snake_case(self, name: str) -> str:
        chars = []
        for index, char in enumerate(name):
            if char.isupper() and index > 0 and name[index - 1] != "_":
                chars.append("_")
            chars.append(char.lower())
        return "".join(chars)

    def _describe_sensor(self, depth_sensor) -> str:
        for attr_name in ("prim_path", "_prim_path", "name", "_name"):
            value = getattr(depth_sensor, attr_name, None)
            if isinstance(value, str) and value:
                return value
        return repr(depth_sensor)

    def _prim_path_sort_key(self, prim_path: str) -> tuple[object, ...]:
        return tuple(
            int(token) if token.isdigit() else token
            for token in re.split(r"(\d+)", prim_path)
        )

    def _import_depth_sensor_class(self):
        try:
            from isaacsim.sensors.camera import SingleViewDepthSensor
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "isaacsim.sensors.camera.SingleViewDepthSensor is not available in the current Python runtime. "
                "Please launch this code from the Isaac Sim 5.1 environment."
            ) from exc
        return SingleViewDepthSensor

    def _import_camera_class(self):
        try:
            from isaacsim.sensors.camera import Camera
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "isaacsim.sensors.camera.Camera is not available in the current Python runtime. "
                "Please launch this code from the Isaac Sim 5.1 environment."
            ) from exc
        return Camera

    def _import_sensor_class(self):
        # The SingleViewDepthSensor applies OmniSensorDepthSensorSingleViewAPI and
        # enables the RTX DepthSensor post-process.  That path is only needed for
        # DepthSensor* AOVs; standard Replicator depth annotators avoid the color
        # / depth texture-size mismatch at small resolutions.
        if self._requires_depth_sensor_api():
            return self._import_depth_sensor_class()
        return self._import_camera_class()

    def _requires_depth_sensor_api(self) -> bool:
        if self.cfg.baseline_mm is not None or len(self.cfg.depth_sensor_attributes) > 0:
            return True
        return any(
            self._to_official_annotator_name(data_type).startswith("DepthSensor")
            for data_type in self.cfg.data_types
        )

    def _import_prim_utils(self):
        try:
            import isaacsim.core.utils.prims as prim_utils
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "isaacsim.core.utils.prims is not available in the current Python runtime. "
                "Please launch this code from the Isaac Sim 5.1 environment."
            ) from exc
        return prim_utils

    def _get_simulation_context(self):
        try:
            from isaacsim.core.api.simulation_context import SimulationContext
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "isaacsim.core.api.simulation_context.SimulationContext is not available in the current "
                "Python runtime. Please launch this code from the Isaac Sim 5.1 environment."
            ) from exc
        return SimulationContext.instance()


def orientation_from_view(camera, target):
    camera_position = Gf.Vec3d(camera)
    target_position = Gf.Vec3d(target)
    up_axis = Gf.Vec3d(0, 0, 1)
    matrix_gf = Gf.Matrix4d(1).SetLookAt(camera_position, target_position, up_axis)
    quat = matrix_gf.GetInverse().ExtractRotationQuat()
    return (quat.real, *quat.imaginary)
