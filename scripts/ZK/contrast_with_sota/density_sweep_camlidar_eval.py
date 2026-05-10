#!/usr/bin/env python3
import argparse
import contextlib
import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ZK_DIR = SCRIPT_DIR.parent
OMNIDRONES_DIR = ZK_DIR.parent.parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "density_sweep_camlidar"


def make_int_range(min_value, max_value, step):
    min_value = int(min_value)
    max_value = int(max_value)
    step = int(step)
    if step <= 0:
        raise ValueError("obstacle step must be positive")
    if max_value < min_value:
        raise ValueError("obstacle max must be >= min")
    return list(range(min_value, max_value + 1, step))


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, payload):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    tmp.replace(path)


def read_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def _finite_float(value, default=float("nan")):
    try:
        value = float(value)
    except Exception:
        return default
    return value if np.isfinite(value) else default


def _weighted_success_mean(rows, key):
    total_weight = 0
    weighted_sum = 0.0
    for row in rows:
        value = _finite_float(row.get(key))
        weight = int(row.get("success_count", 0) or 0)
        if weight <= 0 or not np.isfinite(value):
            continue
        weighted_sum += value * weight
        total_weight += weight
    return weighted_sum / total_weight if total_weight > 0 else float("nan")


def write_results(output_dir, rows):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "density_sweep_results.csv"
    fieldnames = [
        "obstacles_per_tile",
        "trial",
        "seed",
        "success_rate",
        "success_count",
        "episode_count",
        "mean_return",
        "mean_episode_len",
        "mean_completion_pct",
        "mean_arrival_time_s",
        "mean_path_length_m",
        "mean_speed_mps",
        "result",
        "duration_s",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    summary = []
    densities = sorted({int(r["obstacles_per_tile"]) for r in rows})
    for density in densities:
        combo = [r for r in rows if int(r["obstacles_per_tile"]) == density]
        if not combo:
            continue
        total_success = sum(int(r.get("success_count", 0)) for r in combo)
        total_episodes = sum(int(r.get("episode_count", 0)) for r in combo)
        avg_completion = float(np.mean([float(r.get("mean_completion_pct", 0.0)) for r in combo]))
        summary.append(
            {
                "obstacles_per_tile": density,
                "trials": len(combo),
                "seeds": sorted({int(r["seed"]) for r in combo}),
                "success_count": total_success,
                "episode_count": total_episodes,
                "success_rate": total_success / max(1, total_episodes),
                "mean_return": float(np.mean([float(r.get("mean_return", 0.0)) for r in combo])),
                "mean_episode_len": float(np.mean([float(r.get("mean_episode_len", 0.0)) for r in combo])),
                "avg_completion_pct": round(avg_completion, 1),
                "mean_arrival_time_s": round(_weighted_success_mean(combo, "mean_arrival_time_s"), 3),
                "mean_path_length_m": round(_weighted_success_mean(combo, "mean_path_length_m"), 3),
                "mean_speed_mps": round(_weighted_success_mean(combo, "mean_speed_mps"), 3),
            }
        )
    write_json(output_dir / "density_sweep_summary.json", summary)
    if not summary:
        return csv_path
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull, contextlib.redirect_stderr(devnull):
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

        xs = [x["obstacles_per_tile"] for x in summary]
        ys = [x["success_rate"] * 100.0 for x in summary]
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=140)
        ax.plot(xs, ys, marker="o", linewidth=2.0, color="#2563eb")
        ax.set_xlabel("Obstacles per 8x8m tile")
        ax.set_ylabel("Success rate (%)")
        ax.set_title("OmniDrones Cam+LiDAR Policy Density Sweep")
        ax.set_ylim(-2, 102)
        ax.set_xlim(min(xs) - 1, max(xs) + 1)
        ax.grid(True, alpha=0.35)
        fig.tight_layout()
        fig.savefig(output_dir / "success_rate.png")
        plt.close(fig)

        # ---- completion percentage chart ----
        fig2, ax2 = plt.subplots(figsize=(7.2, 4.2), dpi=140)
        xs2 = [x["obstacles_per_tile"] for x in summary]
        ys2 = [x["avg_completion_pct"] for x in summary]
        ax2.plot(xs2, ys2, marker="o", linewidth=2.0, color="#16a34a")
        ax2.set_xlabel("Obstacles per 8x8m tile")
        ax2.set_ylabel("Avg completion (%)")
        ax2.set_title("OmniDrones Cam+LiDAR Policy Density Sweep — Completion %")
        ax2.set_ylim(-2, 102)
        ax2.set_xlim(min(xs2) - 1, max(xs2) + 1)
        ax2.grid(True, alpha=0.35)
        fig2.tight_layout()
        fig2.savefig(output_dir / "completion_pct.png")
        plt.close(fig2)

        metric_specs = [
            ("mean_arrival_time_s", "arrival_time.png", "Avg Arrival Time (success only)", "Avg arrival time (s)", "#dc2626"),
            ("mean_path_length_m", "path_length.png", "Avg Path Length (success only)", "Avg path length (m)", "#9333ea"),
            ("mean_speed_mps", "mean_speed.png", "Avg Speed (success only)", "Avg speed (m/s)", "#ea580c"),
        ]
        for key, filename, title_suffix, ylabel, color in metric_specs:
            metric_summary = [x for x in summary if np.isfinite(_finite_float(x.get(key)))]
            if not metric_summary:
                continue
            fig_m, ax_m = plt.subplots(figsize=(7.2, 4.2), dpi=140)
            xs_m = [x["obstacles_per_tile"] for x in metric_summary]
            ys_m = [x[key] for x in metric_summary]
            ax_m.plot(xs_m, ys_m, marker="o", linewidth=2.0, color=color)
            ax_m.set_xlabel("Obstacles per 8x8m tile")
            ax_m.set_ylabel(ylabel)
            ax_m.set_title(f"OmniDrones Cam+LiDAR Policy Density Sweep — {title_suffix}")
            ax_m.set_xlim(min(xs_m) - 1, max(xs_m) + 1)
            ax_m.grid(True, alpha=0.35)
            fig_m.tight_layout()
            fig_m.savefig(output_dir / filename)
            plt.close(fig_m)
    except Exception as exc:
        print(f"[density sweep] plot skipped: {exc}")
    return csv_path


def _sanitize_run_label(label):
    cleaned = re.sub(r"[^A-Za-z0-9._+=-]+", "_", str(label or "").strip())
    cleaned = cleaned.strip("._")
    return cleaned or "run"


def _extract_override_value(hydra_overrides, key):
    prefix = f"{key}="
    for token in reversed(list(hydra_overrides or [])):
        if str(token).startswith(prefix):
            return str(token)[len(prefix):].strip().strip("\"'")
    return ""


def _checkpoint_path_from_play_yaml():
    cfg_path = ZK_DIR / "play_camlidar.yaml"
    if not cfg_path.exists():
        return ""
    try:
        for line in cfg_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("checkpoint_path:"):
                return stripped.split(":", 1)[1].strip().strip("\"'")
    except Exception:
        return ""
    return ""


def infer_run_label(args, hydra_overrides):
    checkpoint_path = str(getattr(args, "checkpoint_path", "") or "").strip()
    if not checkpoint_path:
        checkpoint_path = _extract_override_value(hydra_overrides, "checkpoint_path")
    if not checkpoint_path:
        checkpoint_path = _checkpoint_path_from_play_yaml()
    if not checkpoint_path:
        return ""

    name = Path(checkpoint_path).name or Path(checkpoint_path).stem
    if name.endswith(".pt"):
        name = Path(name).stem
    return _sanitize_run_label(name)


def make_run_output_dir(base_output_dir, run_subdir=True, run_label=""):
    base_output_dir = Path(base_output_dir).expanduser().resolve()
    base_output_dir.mkdir(parents=True, exist_ok=True)
    if not bool(run_subdir):
        return base_output_dir

    if run_label:
        stem = _sanitize_run_label(run_label)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        stem = f"run_{stamp}"

    candidate = base_output_dir / stem
    suffix = 2
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            candidate = base_output_dir / f"{stem}_{suffix:02d}"
            suffix += 1


WEB_HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OmniDrones Density Sweep</title>
  <style>
    html, body { margin: 0; height: 100%; background: #101418; color: #e8edf2; font: 14px system-ui, sans-serif; }
    #wrap { display: grid; grid-template-columns: 1fr 320px; height: 100%; }
    canvas { width: 100%; height: 100%; display: block; background: #111820; }
    aside { border-left: 1px solid #2b3540; padding: 14px; background: #151b22; overflow: auto; }
    h1 { font-size: 18px; margin: 0 0 12px; }
    .metric { display: flex; justify-content: space-between; gap: 12px; padding: 7px 0; border-bottom: 1px solid #26313c; }
    .label { color: #95a3b3; }
    .value { font-weight: 650; text-align: right; }
    .ok { color: #65d889; } .bad { color: #ff7878; } .run { color: #7cb7ff; }
    table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 12px; }
    th, td { border-bottom: 1px solid #26313c; padding: 5px 3px; text-align: right; }
    th:first-child, td:first-child { text-align: left; }
  </style>
</head>
<body>
<div id="wrap">
  <canvas id="view"></canvas>
  <aside>
    <h1>OmniDrones Density Sweep</h1>
    <div id="metrics"></div>
    <table><thead><tr><th>density</th><th>trials</th><th>success</th><th>speed</th></tr></thead><tbody id="summary"></tbody></table>
  </aside>
</div>
<script>
const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
const metrics = document.getElementById("metrics");
const summaryEl = document.getElementById("summary");
function resize(){ canvas.width = canvas.clientWidth * devicePixelRatio; canvas.height = canvas.clientHeight * devicePixelRatio; }
addEventListener("resize", resize); resize();
function worldToCanvas(p){
  const w = canvas.width, h = canvas.height;
  const sx = w / 44, sy = h / 66, s = Math.min(sx, sy);
  return [w/2 + p[0]*s, h/2 - p[1]*s];
}
function drawGrid(){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.lineWidth = 1 * devicePixelRatio;
  ctx.strokeStyle = "#22303a";
  for(let x=-20; x<=20; x+=4){ const a=worldToCanvas([x,-30]), b=worldToCanvas([x,30]); ctx.beginPath(); ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]); ctx.stroke(); }
  for(let y=-30; y<=30; y+=4){ const a=worldToCanvas([-20,y]), b=worldToCanvas([20,y]); ctx.beginPath(); ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]); ctx.stroke(); }
}
function dot(p, r, color){ const q=worldToCanvas(p); ctx.fillStyle=color; ctx.beginPath(); ctx.arc(q[0], q[1], r*devicePixelRatio, 0, Math.PI*2); ctx.fill(); }
function line(points, color, width){
  if(!points || points.length < 2) return;
  ctx.strokeStyle=color; ctx.lineWidth=width*devicePixelRatio; ctx.beginPath();
  points.forEach((p,i)=>{ const q=worldToCanvas(p); if(i===0) ctx.moveTo(q[0],q[1]); else ctx.lineTo(q[0],q[1]); });
  ctx.stroke();
}
function render(state){
  drawGrid();
  const obstacles = state.obstacles || [];
  ctx.fillStyle = "rgba(148, 163, 184, 0.34)";
  ctx.strokeStyle = "rgba(203, 213, 225, 0.55)";
  ctx.lineWidth = 1 * devicePixelRatio;
  for(const o of obstacles){
    const a = worldToCanvas([o.x - o.width/2, o.y - o.height/2]);
    const b = worldToCanvas([o.x + o.width/2, o.y + o.height/2]);
    const x = Math.min(a[0], b[0]), y = Math.min(a[1], b[1]);
    const w = Math.abs(b[0] - a[0]), h = Math.abs(b[1] - a[1]);
    ctx.fillRect(x, y, w, h);
    ctx.strokeRect(x, y, w, h);
  }
  const lidar = state.lidar_points || [];
  for(const p of lidar) dot(p, 1.2, "rgba(180,190,200,0.28)");
  line(state.trajectory || [], "#67b7ff", 2);
  if(state.target) dot(state.target, 7, "#6ee787");
  if(state.position) dot(state.position, 6, "#ffcf5a");
  const cls = state.phase === "running" ? "run" : (state.result === "success" ? "ok" : (state.result ? "bad" : ""));
  const rows = [
    ["phase", `<span class="${cls}">${state.phase || "waiting"}</span>`],
    ["density", state.obstacles_per_tile ?? "-"],
    ["trial", `${state.trial ?? "-"} / ${state.trials ?? "-"}`],
    ["seed", state.seed ?? "-"],
    ["step", state.step ?? "-"],
    ["success", state.success_rate == null ? "-" : (state.success_rate*100).toFixed(1)+"%"],
    ["position", state.position ? state.position.map(x=>x.toFixed(2)).join(", ") : "-"],
    ["target", state.target ? state.target.map(x=>x.toFixed(2)).join(", ") : "-"],
    ["result", state.result || "-"]
  ];
  metrics.innerHTML = rows.map(r=>`<div class="metric"><span class="label">${r[0]}</span><span class="value">${r[1]}</span></div>`).join("");
  const summary = state.summary || [];
  summaryEl.innerHTML = summary.map(s=>{
    const speed = Number.isFinite(s.mean_speed_mps) ? s.mean_speed_mps.toFixed(2) : "-";
    return `<tr><td>${s.obstacles_per_tile}</td><td>${s.trials}</td><td>${(s.success_rate*100).toFixed(1)}%</td><td>${speed}</td></tr>`;
  }).join("");
}
async function tick(){
  try { const r = await fetch("/state", {cache:"no-store"}); render(await r.json()); } catch(e) {}
  requestAnimationFrame(()=>setTimeout(tick, 250));
}
tick();
</script>
</body>
</html>"""


def start_web_server(host, port, live_state_path, summary_path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index"):
                data = WEB_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if self.path.startswith("/state"):
                state = read_json(live_state_path, {})
                if not state.get("obstacles") and int(state.get("obstacles_per_tile", 0) or 0) > 0:
                    state["obstacles"] = make_preview_obstacles(
                        int(state.get("obstacles_per_tile", 0) or 0),
                        int(state.get("seed", 0) or 0),
                    )
                state["summary"] = read_json(summary_path, [])
                data = json.dumps(state).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer((host, int(port)), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


@contextlib.contextmanager
def patched_obstacle_density(obstacles_per_tile, terrain_seed, obstacle_height_mode="choice"):
    import isaaclab.terrains as terrains

    original_obstacle_cfg = terrains.HfDiscreteObstaclesTerrainCfg
    original_generator_cfg = terrains.TerrainGeneratorCfg

    def obstacle_cfg_wrapper(*args, **kwargs):
        kwargs["num_obstacles"] = int(obstacles_per_tile)
        kwargs["obstacle_height_mode"] = obstacle_height_mode
        return original_obstacle_cfg(*args, **kwargs)

    def generator_cfg_wrapper(*args, **kwargs):
        kwargs["seed"] = int(terrain_seed)
        return original_generator_cfg(*args, **kwargs)

    terrains.HfDiscreteObstaclesTerrainCfg = obstacle_cfg_wrapper
    terrains.TerrainGeneratorCfg = generator_cfg_wrapper
    try:
        yield
    finally:
        terrains.HfDiscreteObstaclesTerrainCfg = original_obstacle_cfg
        terrains.TerrainGeneratorCfg = original_generator_cfg


def _sample_first_env_lidar_points(base_env, max_points=800):
    try:
        import torch

        hits = base_env.lidar.data.ray_hits_w.reshape(base_env.num_envs, base_env.num_lidar_points, 3)[0]
        hits = hits.detach()
        finite = torch.isfinite(hits).all(dim=-1)
        hits = hits[finite]
        if hits.numel() == 0:
            return []
        if hits.shape[0] > max_points:
            idx = torch.linspace(0, hits.shape[0] - 1, max_points, device=hits.device).long()
            hits = hits[idx]
        return [[round(float(x), 3), round(float(y), 3), round(float(z), 3)] for x, y, z in hits.cpu().tolist()]
    except Exception:
        return []


def make_preview_obstacles(obstacles_per_tile, seed, rows=5, cols=5, tile_size=8.0, max_boxes=1200):
    """Approximate the forest_lc heightfield obstacle layout for the web top-down preview."""
    count = int(obstacles_per_tile)
    if count <= 0:
        return []
    rng = np.random.default_rng(int(seed))
    boxes = []
    x0 = -0.5 * rows * tile_size
    y0 = -0.5 * cols * tile_size
    platform_half = 0.75
    for row in range(rows):
        for col in range(cols):
            tile_min_x = x0 + row * tile_size
            tile_min_y = y0 + col * tile_size
            tile_cx = tile_min_x + 0.5 * tile_size
            tile_cy = tile_min_y + 0.5 * tile_size
            for _ in range(count):
                width = float(rng.choice(np.arange(0.4, 0.8, 0.4)))
                height = float(rng.choice(np.arange(0.4, 0.8, 0.4)))
                cx = float(rng.uniform(tile_min_x + width * 0.5, tile_min_x + tile_size - width * 0.5))
                cy = float(rng.uniform(tile_min_y + height * 0.5, tile_min_y + tile_size - height * 0.5))
                if abs(cx - tile_cx) < platform_half + width * 0.5 and abs(cy - tile_cy) < platform_half + height * 0.5:
                    continue
                boxes.append(
                    {
                        "x": round(cx, 3),
                        "y": round(cy, 3),
                        "width": round(width, 3),
                        "height": round(height, 3),
                    }
                )
                if len(boxes) >= max_boxes:
                    return boxes
    return boxes


def run_worker(args, hydra_overrides):
    sys.path.insert(0, str(ZK_DIR))
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import torch
    from torchrl.data import CompositeSpec
    from torchrl.envs.transforms import Compose, InitTracker, TransformedEnv
    from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp

    from omni_drones import init_simulation_app
    from omni_drones.learning import ALGOS
    from omni_drones.utils.torchrl.transforms import FromDiscreteAction, FromMultiDiscreteAction, ravel_composite

    from play_camlidar import (
        _inject_camlidar_backbone,
        _load_checkpoint_strictish,
        _preflight_runtime_checks,
        _resolve_checkpoint_path,
        _select_first_env_value,
    )

    overrides = [f"hydra.searchpath=[file://{OMNIDRONES_DIR / 'cfg'}]"]
    overrides += list(hydra_overrides)
    overrides += [
        f"seed={int(args.worker_seed)}",
        f"eval_num_envs={int(args.eval_num_envs)}",
        f"num_episodes={int(args.num_episodes)}",
        f"max_steps={int(args.max_steps)}",
        "final_eval_rounds=1",
        "headless=true",
        "enable_viewport=false",
        "task.show_depth_preview_window=false",
    ]
    if args.checkpoint_path:
        overrides.append(f"checkpoint_path={args.checkpoint_path}")

    with initialize_config_dir(version_base=None, config_dir=str(ZK_DIR), job_name="density_sweep_worker"):
        cfg = compose(config_name="play_camlidar", overrides=overrides)
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    cfg.task.obstacles_per_tile = int(args.worker_density)
    if "env" in cfg:
        cfg.env.num_envs = int(args.eval_num_envs)
    if "task" in cfg and "env" in cfg.task:
        cfg.task.env.num_envs = int(args.eval_num_envs)
    cfg.sim.device = f"cuda:{int(cfg.get('sim_gpu_index', 0))}"
    cfg.sim.active_gpu = int(cfg.get("sim_gpu_index", 0))
    cfg.sim.physics_gpu = int(cfg.get("sim_gpu_index", 0))
    cfg.sim.enable_viewport = False
    cfg.sim.enable_replicator = True

    live_state_path = Path(args.live_state)
    result_path = Path(args.worker_result)
    trajectory = []
    preview_obstacles = make_preview_obstacles(int(args.worker_density), int(args.worker_seed))
    start_time = time.time()
    write_json(
        live_state_path,
        {
            "phase": "starting",
            "obstacles_per_tile": int(args.worker_density),
            "trial": int(args.worker_trial),
            "trials": int(args.trials),
            "seed": int(args.worker_seed),
            "trajectory": trajectory,
            "obstacles": preview_obstacles,
        },
    )

    _preflight_runtime_checks(cfg, int(args.eval_num_envs))
    simulation_app = None
    try:
        simulation_app = init_simulation_app(cfg)
        with patched_obstacle_density(int(args.worker_density), int(args.worker_seed), obstacle_height_mode=args.obstacle_height_mode):
            from omni_drones.envs.isaac_env import IsaacEnv
            import importlib

            task_name = str(cfg.task.name)
            try:
                importlib.import_module(f"omni_drones.envs.single.{task_name.lower()}")
            except ModuleNotFoundError:
                pass

            env_class = IsaacEnv.REGISTRY[cfg.task.name]
            base_env = env_class(cfg, headless=True)

        transforms = [InitTracker()]
        if cfg.task.get("ravel_obs", False):
            transforms.append(ravel_composite(base_env.observation_spec, ("agents", "observation")))
        if cfg.task.get("ravel_obs_central", False):
            transforms.append(ravel_composite(base_env.observation_spec, ("agents", "observation_central")))
        if (
            cfg.task.get("flatten_intrinsics", True)
            and ("agents", "intrinsics") in base_env.observation_spec.keys(True)
            and isinstance(base_env.observation_spec[("agents", "intrinsics")], CompositeSpec)
        ):
            transforms.append(ravel_composite(base_env.observation_spec, ("agents", "intrinsics"), start_dim=-1))

        action_transform = cfg.task.get("action_transform", None)
        if action_transform is not None:
            if action_transform.startswith("multidiscrete"):
                transforms.append(FromMultiDiscreteAction(nbins=int(action_transform.split(":")[1])))
            elif action_transform.startswith("discrete"):
                transforms.append(FromDiscreteAction(nbins=int(action_transform.split(":")[1])))
            else:
                raise NotImplementedError(f"Unknown action transform: {action_transform}")

        env = TransformedEnv(base_env, Compose(*transforms)).eval()
        env.set_seed(int(args.worker_seed))
        policy = ALGOS[cfg.algo.name.lower()](
            cfg.algo,
            env.observation_spec,
            env.action_spec,
            env.reward_spec,
            device=base_env.device,
        )
        _inject_camlidar_backbone(policy, base_env, env, cfg)
        checkpoint_path = _resolve_checkpoint_path(cfg.get("checkpoint_path"))
        _load_checkpoint_strictish(policy, checkpoint_path, base_env.device)
        policy.eval()
        base_env.enable_render(False)
        base_env.eval()
        env.eval()

        def _run_one_trial(trial_idx):
            """Run one trial (num_episodes episodes) and return the result row dict."""
            trial_start = time.time()
            episode_success = []
            episode_returns = []
            episode_lengths = []
            episode_completion_pct = []
            episode_arrival_times = []
            episode_path_lengths = []
            episode_speeds = []
            latest_success_rate = 0.0
            trajectory = []
            with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
                for ep in range(int(args.num_episodes)):
                    td = env.reset()
                    num_envs_eval = int(base_env.num_envs)
                    # Capture start / target positions for completion_pct
                    init_pos = base_env.drone.pos.detach().clone().reshape(num_envs_eval, 3)
                    target_pos = base_env.target_pos.detach().clone().reshape(num_envs_eval, 3)
                    start_dist = torch.norm(target_pos - init_pos, dim=-1)
                    final_positions = torch.full_like(init_pos, float("nan"))
                    prev_positions = init_pos.clone()
                    path_lengths = torch.zeros(num_envs_eval, dtype=torch.float32, device=base_env.device)
                    arrival_steps = torch.full(
                        (num_envs_eval,),
                        -1,
                        dtype=torch.int32,
                        device=base_env.device,
                    )
                    finish_steps = torch.full(
                        (num_envs_eval,),
                        int(args.max_steps),
                        dtype=torch.int32,
                        device=base_env.device,
                    )
                    finished = torch.zeros(num_envs_eval, dtype=torch.bool, device=base_env.device)
                    ep_returns = torch.zeros(num_envs_eval, dtype=torch.float32, device=base_env.device)
                    ep_success = torch.zeros(num_envs_eval, dtype=torch.int32, device=base_env.device)
                    step_count = 0

                    for step in range(int(args.max_steps)):
                        step_count = step + 1
                        td = policy(td)
                        td = env.step(td)
                        reward = td[("next", "agents", "reward")].reshape(-1).float()
                        done = td[("next", "done")].reshape(-1).bool()
                        stats_td = td[("next", "stats")]
                        current_pos = base_env.drone.pos.detach().clone().reshape(num_envs_eval, 3)

                        active = ~finished
                        if active.any():
                            path_lengths[active] += torch.norm(current_pos[active] - prev_positions[active], dim=-1)
                            prev_positions[active] = current_pos[active]

                        # Capture final positions for envs that just finished
                        just_finished = ~finished & done
                        if just_finished.any():
                            final_positions[just_finished] = current_pos[just_finished]
                            finish_steps[just_finished] = step_count

                        if active.any():
                            ep_returns[active] += reward[active]
                        finished = finished | done
                        if "success" in stats_td.keys():
                            success_now = stats_td["success"].reshape(-1) >= 0.5
                            first_success = success_now & (arrival_steps < 0)
                            if first_success.any():
                                arrival_steps[first_success] = step_count
                            ep_success = torch.maximum(ep_success, success_now.to(torch.int32))

                        if step % int(args.web_update_interval) == 0 or finished.all():
                            try:
                                pos = _select_first_env_value(base_env.drone.pos).detach().cpu().reshape(-1)[:3].tolist()
                                target = _select_first_env_value(base_env.target_pos).detach().cpu().reshape(-1)[:3].tolist()
                                pos = [round(float(v), 3) for v in pos]
                                target = [round(float(v), 3) for v in target]
                                trajectory.append(pos)
                                trajectory = trajectory[-600:]
                                latest_success_rate = (sum(episode_success) + int(ep_success.sum().item())) / max(
                                    1, len(episode_success) + num_envs_eval
                                )
                                write_json(
                                    live_state_path,
                                    {
                                        "phase": "running",
                                        "obstacles_per_tile": int(args.worker_density),
                                        "trial": trial_idx,
                                        "trials": int(args.trials),
                                        "seed": int(args.worker_seed),
                                        "episode": ep + 1,
                                        "episodes": int(args.num_episodes),
                                        "step": step_count,
                                        "position": pos,
                                        "target": target,
                                        "trajectory": trajectory,
                                        "obstacles": preview_obstacles,
                                        "lidar_points": _sample_first_env_lidar_points(base_env),
                                        "success_rate": latest_success_rate,
                                    },
                                )
                            except Exception:
                                pass

                        if finished.all():
                            break
                        td = step_mdp(td)

                    # For envs that never finished, use current position
                    never_finished = torch.isnan(final_positions).any(dim=-1)
                    if never_finished.any():
                        current_pos = base_env.drone.pos.detach().clone().reshape(num_envs_eval, 3)
                        final_positions[never_finished] = current_pos[never_finished]

                    # Compute per-env completion_pct
                    final_dist = torch.norm(target_pos - final_positions, dim=-1)
                    completion_pct = torch.clamp(
                        (1.0 - final_dist / start_dist.clamp_min(1e-6)) * 100.0, 0.0, 100.0
                    )
                    success_mask = ep_success.bool()
                    sim_dt = float(getattr(base_env, "dt", 0.02))
                    arrival_times = arrival_steps.to(torch.float32) * sim_dt
                    episode_speeds_tensor = path_lengths / arrival_times.clamp_min(1e-6)

                    episode_success.extend(int(v) for v in ep_success.detach().cpu().tolist())
                    episode_returns.extend(float(v) for v in ep_returns.detach().cpu().tolist())
                    episode_lengths.extend(int(v) for v in finish_steps.detach().cpu().tolist())
                    episode_completion_pct.extend(float(v) for v in completion_pct.detach().cpu().tolist())
                    episode_arrival_times.extend(
                        float(v) for v in arrival_times[success_mask].detach().cpu().tolist()
                    )
                    episode_path_lengths.extend(
                        float(v) for v in path_lengths[success_mask].detach().cpu().tolist()
                    )
                    episode_speeds.extend(
                        float(v) for v in episode_speeds_tensor[success_mask].detach().cpu().tolist()
                    )

            success_count = int(sum(episode_success))
            episode_count = int(len(episode_success))
            mean_completion = float(np.mean(episode_completion_pct)) if episode_completion_pct else 0.0
            return {
                "obstacles_per_tile": int(args.worker_density),
                "trial": trial_idx,
                "seed": int(args.worker_seed),
                "success_rate": success_count / max(1, episode_count),
                "success_count": success_count,
                "episode_count": episode_count,
                "mean_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
                "mean_episode_len": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
                "mean_completion_pct": round(mean_completion, 1),
                "mean_arrival_time_s": round(float(np.mean(episode_arrival_times)), 3) if episode_arrival_times else float("nan"),
                "mean_path_length_m": round(float(np.mean(episode_path_lengths)), 3) if episode_path_lengths else float("nan"),
                "mean_speed_mps": round(float(np.mean(episode_speeds)), 3) if episode_speeds else float("nan"),
                "result": "ok",
                "duration_s": round(time.time() - trial_start, 3),
            }

        # ---- run all trials for this density in a single IsaacLab session ----
        all_rows = []
        for trial_idx in range(1, int(args.trials) + 1):
            try:
                row = _run_one_trial(trial_idx)
                all_rows.append(row)
                trial_result_path = Path(str(result_path).replace(f"_trial_{args.worker_trial}", f"_trial_{trial_idx}"))
                write_json(trial_result_path, row)
                print(json.dumps(row, indent=2))
            except Exception as exc:
                row = {
                    "obstacles_per_tile": int(args.worker_density),
                    "trial": trial_idx,
                    "seed": int(args.worker_seed),
                    "success_rate": 0.0,
                    "success_count": 0,
                    "episode_count": 0,
                    "mean_return": 0.0,
                    "mean_episode_len": 0.0,
                    "mean_completion_pct": 0.0,
                    "mean_arrival_time_s": float("nan"),
                    "mean_path_length_m": float("nan"),
                    "mean_speed_mps": float("nan"),
                    "result": f"error:{type(exc).__name__}:{exc}",
                    "duration_s": 0.0,
                }
                all_rows.append(row)
                trial_result_path = Path(str(result_path).replace(f"_trial_{args.worker_trial}", f"_trial_{trial_idx}"))
                write_json(trial_result_path, row)
                write_json(live_state_path, {**row, "phase": "error", "obstacles": preview_obstacles})
                print(f"[worker] trial {trial_idx} failed: {exc}")

        # Write the last trial result to the main result_path for compatibility
        if all_rows:
            write_json(result_path, all_rows[-1])
        state = read_json(live_state_path, {})
        last_row = all_rows[-1] if all_rows else {}
        state.update({"phase": "done", "result": "success" if last_row.get("success_rate", 0) > 0 else "no_success",
                       "success_rate": last_row.get("success_rate", 0)})
        state.setdefault("obstacles", preview_obstacles)
        write_json(live_state_path, state)
        print(f"[worker] density={args.worker_density} completed {len(all_rows)}/{args.trials} trials")
        return 0
    except Exception as exc:
        row = {
            "obstacles_per_tile": int(args.worker_density),
            "trial": int(args.worker_trial),
            "seed": int(args.worker_seed),
            "success_rate": 0.0,
            "success_count": 0,
            "episode_count": 0,
            "mean_return": 0.0,
            "mean_episode_len": 0.0,
            "mean_completion_pct": 0.0,
            "mean_arrival_time_s": float("nan"),
            "mean_path_length_m": float("nan"),
            "mean_speed_mps": float("nan"),
            "result": f"error:{type(exc).__name__}:{exc}",
            "duration_s": round(time.time() - start_time, 3),
        }
        write_json(result_path, row)
        write_json(live_state_path, {**row, "phase": "error", "obstacles": preview_obstacles})
        raise
    finally:
        if simulation_app is not None:
            simulation_app.close()


def controller(args, hydra_overrides):
    run_label = infer_run_label(args, hydra_overrides)
    output_dir = make_run_output_dir(args.output_dir, args.run_subdir, run_label=run_label)
    live_state = output_dir / "live_state.json"
    summary_path = output_dir / "density_sweep_summary.json"
    densities = make_int_range(args.obstacles_per_tile_min, args.obstacles_per_tile_max, args.obstacles_per_tile_step)
    total_trials = len(densities) * int(args.trials)
    rows = []
    write_results(output_dir, rows)
    write_json(live_state, {"phase": "idle", "trajectory": [], "lidar_points": [], "obstacles": []})

    server = None
    if not args.no_web:
        server = start_web_server(args.host, args.port, live_state, summary_path)
        print(f"[density sweep] web: http://127.0.0.1:{args.port}")

    print(
        f"[density sweep] plan: obstacles/tile={densities[0]}..{densities[-1]} "
        f"step={args.obstacles_per_tile_step} trials={args.trials} total={total_trials} "
        f"seed_mode=density base_seed={args.seed}"
    )
    if run_label:
        print(f"[density sweep] run label: {run_label}")
    print(f"[density sweep] output: {output_dir}")
    if args.dry_run:
        dry_obstacles = make_preview_obstacles(densities[0], int(args.seed)) if densities else []
        write_json(
            live_state,
            {"phase": "dry_run", "densities": densities, "trajectory": [], "lidar_points": [], "obstacles": dry_obstacles},
        )
        if server is not None:
            server.shutdown()
        return 0

    completed = 0
    try:
        for density_index, density in enumerate(densities):
            seed = int(args.seed) + density_index
            # Launch ONE worker per density – it runs all trials internally without restarting IsaacLab
            result_path = output_dir / f"worker_density_{density}_trial_1.json"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--worker-density",
                str(density),
                "--worker-trial",
                str(1),
                "--worker-seed",
                str(seed),
                "--trials",
                str(args.trials),
                "--eval-num-envs",
                str(args.eval_num_envs),
                "--num-episodes",
                str(args.num_episodes),
                "--max-steps",
                str(args.max_steps),
                "--web-update-interval",
                str(args.web_update_interval),
                "--obstacle-height-mode",
                args.obstacle_height_mode,
                "--live-state",
                str(live_state),
                "--worker-result",
                str(result_path),
            ]
            if args.checkpoint_path:
                cmd += ["--checkpoint-path", args.checkpoint_path]
            cmd += hydra_overrides
            print(
                f"[density sweep] density={density} ({density_index+1}/{len(densities)}) "
                f"seed={seed} trials={args.trials}"
            )
            proc = subprocess.run(cmd, cwd=str(OMNIDRONES_DIR))

            # Read back all trial results written by the worker
            for trial in range(1, int(args.trials) + 1):
                completed += 1
                trial_result_path = output_dir / f"worker_density_{density}_trial_{trial}.json"
                row = read_json(
                    trial_result_path,
                    {
                        "obstacles_per_tile": density,
                        "trial": trial,
                        "seed": seed,
                        "success_rate": 0.0,
                        "success_count": 0,
                        "episode_count": 0,
                        "mean_return": 0.0,
                        "mean_episode_len": 0.0,
                        "mean_completion_pct": 0.0,
                        "mean_arrival_time_s": float("nan"),
                        "mean_path_length_m": float("nan"),
                        "mean_speed_mps": float("nan"),
                        "result": f"worker_exit_{proc.returncode}",
                        "duration_s": 0.0,
                    },
                )
                if proc.returncode != 0 and str(row.get("result", "ok")) == "ok":
                    row["result"] = f"worker_exit_{proc.returncode}"
                rows.append(row)
                csv_path = write_results(output_dir, rows)
                print(
                    f"[density sweep] {completed}/{total_trials}: "
                    f"density={density} trial={trial}/{args.trials} "
                    f"success={float(row['success_rate']) * 100.0:.1f}% "
                    f"arrival={_finite_float(row.get('mean_arrival_time_s')):.2f}s "
                    f"path={_finite_float(row.get('mean_path_length_m')):.2f}m "
                    f"speed={_finite_float(row.get('mean_speed_mps')):.2f}m/s "
                    f"result={row['result']}"
                )
            if proc.returncode != 0 and bool(args.stop_on_error):
                write_json(
                    live_state,
                    {
                        "phase": "stopped_on_error",
                        "obstacles_per_tile": density,
                        "trial": 1,
                        "trials": int(args.trials),
                        "seed": seed,
                        "result": f"worker_exit_{proc.returncode}",
                        "trajectory": [],
                        "lidar_points": [],
                        "obstacles": make_preview_obstacles(density, seed),
                    },
                )
                print("[density sweep] stopped because worker failed; use --keep-going to continue after errors")
                return proc.returncode
    finally:
        if server is not None:
            server.shutdown()
    print(f"[density sweep] done: {output_dir}")
    return 0


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Evaluate an existing OmniDrones camera+LiDAR policy while sweeping only obstacle density.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--obstacles-per-tile-min", type=int, default=0)
    parser.add_argument("--obstacles-per-tile-max", type=int, default=40)
    parser.add_argument("--obstacles-per-tile-step", type=int, default=4)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-num-envs", type=int, default=10)
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--no-run-subdir",
        dest="run_subdir",
        action="store_false",
        help="Write directly into --output-dir without creating a per-run subdirectory",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8895)
    parser.add_argument("--no-web", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", dest="stop_on_error", action="store_false",
                        help="Continue the sweep after a worker error")
    parser.add_argument("--web-update-interval", type=int, default=10)
    parser.add_argument("--obstacle-height-mode", default="fixed", choices=["choice", "fixed"],
                        help="Obstacle height mode: choice (pillars+holes) or fixed (all pillars)")
    parser.set_defaults(stop_on_error=True, run_subdir=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-density", type=int, default=40, help=argparse.SUPPRESS)
    parser.add_argument("--worker-trial", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--worker-seed", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--live-state", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", default="", help=argparse.SUPPRESS)
    args, hydra_overrides = parser.parse_known_args(argv)
    return args, hydra_overrides


def main(argv=None):
    args, hydra_overrides = parse_args(sys.argv[1:] if argv is None else argv)
    if args.worker:
        if not args.live_state or not args.worker_result:
            raise ValueError("--worker requires --live-state and --worker-result")
        return run_worker(args, hydra_overrides)
    return controller(args, hydra_overrides)


if __name__ == "__main__":
    raise SystemExit(main())
