import argparse
import glob
import os
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


def _safe_step_count(valid_row: np.ndarray) -> int:
    count = int(np.sum(valid_row))
    return max(1, count)


def load_npz(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    xyz = data["xyz"]
    valid = data["valid"]
    done = data["done"] if "done" in data else np.zeros(valid.shape, dtype=bool)
    reward_total = data["reward_total"]
    death_reason_code = data["death_reason_code"] if "death_reason_code" in data else np.zeros((xyz.shape[0],), dtype=np.int32)
    death_reason_name = data["death_reason_name"] if "death_reason_name" in data else np.asarray(["unknown"] * xyz.shape[0], dtype=object)
    obstacle_points = data["obstacle_points"] if "obstacle_points" in data else np.full((xyz.shape[0], 0, 3), np.nan, dtype=np.float32)
    obstacle_env_points = data["obstacle_env_points"] if "obstacle_env_points" in data else np.full((0, 3), np.nan, dtype=np.float32)
    speed_mps = data["speed_mps"] if "speed_mps" in data else None
    if "sim_dt" in data:
        sim_dt = float(np.asarray(data["sim_dt"]).reshape(-1)[0])
    elif "dt" in data:
        sim_dt = float(np.asarray(data["dt"]).reshape(-1)[0])
    elif "control_dt" in data:
        sim_dt = float(np.asarray(data["control_dt"]).reshape(-1)[0])
    else:
        sim_dt = None

    env_ids = data["env_ids"]
    returns = data["returns"]
    success = data["success"]

    if speed_mps is None:
        speed_mps = np.full((xyz.shape[0], xyz.shape[1]), np.nan, dtype=np.float32)
    else:
        speed_mps = speed_mps.astype(np.float32)

    # Fallback: estimate speed from position differences when speed was not exported.
    if np.isnan(speed_mps).all() and sim_dt is not None and sim_dt > 1e-6:
        dxyz = np.diff(xyz, axis=1)
        speed_est = np.linalg.norm(dxyz, axis=-1) / float(sim_dt)
        speed_mps[:, 1:] = speed_est.astype(np.float32)
        speed_mps[:, 0] = speed_mps[:, 1] if xyz.shape[1] > 1 else 0.0

    factor_names = []
    factor_values = []
    for key in data.files:
        if key.startswith("factor__"):
            factor_names.append(key.replace("factor__", ""))
            factor_values.append(data[key])

    if len(factor_names) == 0:
        factor_mat = np.zeros((xyz.shape[0], xyz.shape[1], 0), dtype=np.float32)
    else:
        order = np.argsort(np.asarray(factor_names))
        factor_names = [factor_names[i] for i in order]
        factor_values = [factor_values[i] for i in order]
        factor_mat = np.stack(factor_values, axis=-1)

    return {
        "xyz": xyz,
        "valid": valid,
        "done": done,
        "reward_total": reward_total,
        "speed_mps": speed_mps,
        "death_reason_code": death_reason_code,
        "death_reason_name": death_reason_name,
        "obstacle_points": obstacle_points,
        "obstacle_env_points": obstacle_env_points,
        "sim_dt": sim_dt,
        "env_ids": env_ids,
        "returns": returns,
        "success": success,
        "factor_names": factor_names,
        "factor_mat": factor_mat,
    }


def _find_latest_npz(search_roots):
    candidates = []
    for root in search_roots:
        if not root:
            continue
        pattern = os.path.join(root, "**", "eval_top*_traj_step_*.npz")
        candidates.extend(glob.glob(pattern, recursive=True))

    if not candidates:
        return None

    candidates = sorted(candidates, key=os.path.getmtime, reverse=True)
    return candidates[0]


def _extract_cylinder_models(points_xyz: np.ndarray, cell_size: float = 0.8, min_points: int = 8, max_cylinders: int = 80):
    if points_xyz.size == 0:
        return []

    pts = points_xyz[np.isfinite(points_xyz).all(axis=1)]
    if pts.size == 0:
        return []

    # Remove ground points and keep vertical obstacle points.
    pts = pts[pts[:, 2] > 0.15]
    if pts.shape[0] < min_points:
        return []

    xy = pts[:, :2]
    grid = np.floor(xy / cell_size).astype(np.int32)
    buckets = {}
    for i, g in enumerate(grid):
        key = (int(g[0]), int(g[1]))
        buckets.setdefault(key, []).append(i)

    models = []
    for idxs in buckets.values():
        if len(idxs) < min_points:
            continue
        p = pts[np.asarray(idxs, dtype=np.int32)]
        center_xy = np.median(p[:, :2], axis=0)
        radial = np.linalg.norm(p[:, :2] - center_xy[None, :], axis=1)
        radius = float(np.clip(np.percentile(radial, 85), 0.15, 1.2))

        z0 = float(np.percentile(p[:, 2], 5))
        z1 = float(np.percentile(p[:, 2], 95))
        if (z1 - z0) < 0.25:
            continue

        models.append((float(center_xy[0]), float(center_xy[1]), radius, z0, z1, len(idxs)))

    models.sort(key=lambda x: x[-1], reverse=True)
    return models[:max_cylinders]


def _draw_cylinder(ax3d, cx: float, cy: float, r: float, z0: float, z1: float):
    theta = np.linspace(0.0, 2.0 * np.pi, 24)
    z = np.linspace(z0, z1, 4)
    tt, zz = np.meshgrid(theta, z)
    xx = cx + r * np.cos(tt)
    yy = cy + r * np.sin(tt)
    surf = ax3d.plot_surface(
        xx,
        yy,
        zz,
        color=(0.45, 0.45, 0.45),
        alpha=0.38,
        linewidth=0,
        antialiased=False,
        shade=True,
    )
    return surf


def _draw_goal_sphere(ax3d, cx: float, cy: float, cz: float, radius: float):
    u = np.linspace(0.0, 2.0 * np.pi, 40)
    v = np.linspace(0.0, np.pi, 20)
    uu, vv = np.meshgrid(u, v)
    xx = cx + radius * np.cos(uu) * np.sin(vv)
    yy = cy + radius * np.sin(uu) * np.sin(vv)
    zz = cz + radius * np.cos(vv)

    surf = ax3d.plot_surface(
        xx,
        yy,
        zz,
        color=(0.1, 0.7, 0.3),
        alpha=0.20,
        linewidth=0,
        antialiased=True,
        shade=False,
        zorder=1,
    )
    ax3d.plot(
        [cx],
        [cy],
        [cz],
        marker="x",
        markersize=8,
        markeredgewidth=2,
        color=(0.0, 0.45, 0.0),
    )
    return surf


def main():
    parser = argparse.ArgumentParser(description="Visualize drone trajectories with per-step reward factors.")
    parser.add_argument("--npz", default=None, help="Path to eval_topk trajectory npz file")
    parser.add_argument("--rank", type=int, default=0, help="Top-k rank index to view, 0 means best")
    parser.add_argument("--goal-x", type=float, default=0.0, help="Goal sphere center x")
    parser.add_argument("--goal-y", type=float, default=24.0, help="Goal sphere center y")
    parser.add_argument("--goal-z", type=float, default=2.0, help="Goal sphere center z")
    parser.add_argument("--goal-radius", type=float, default=6.0, help="Goal sphere radius")
    parser.add_argument("--step-dt", type=float, default=None, help="Seconds per trajectory step for auto-play")
    args = parser.parse_args()

    npz_path = args.npz
    if npz_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
        search_roots = [
            os.getcwd(),
            script_dir,
            workspace_root,
            os.path.join(workspace_root, "wandb"),
            os.path.join(workspace_root, "outputs"),
        ]
        npz_path = _find_latest_npz(search_roots)
        if npz_path is None:
            raise SystemExit(
                "No trajectory npz found. Run evaluation first, or pass --npz <path_to_eval_top*.npz>."
            )

    payload = load_npz(npz_path)
    xyz = payload["xyz"]
    valid = payload["valid"]
    done = payload["done"]
    reward_total = payload["reward_total"]
    speed_mps = payload["speed_mps"]
    death_reason_code = payload["death_reason_code"]
    death_reason_name = payload["death_reason_name"]
    obstacle_points = payload["obstacle_points"]
    obstacle_env_points = payload["obstacle_env_points"]
    sim_dt = payload["sim_dt"]
    env_ids = payload["env_ids"]
    returns = payload["returns"]
    success = payload["success"]
    factor_names = payload["factor_names"]
    factor_mat = payload["factor_mat"]

    if xyz.shape[0] == 0:
        raise RuntimeError("No trajectory data in npz.")

    top_k = xyz.shape[0]
    rank_init = int(np.clip(args.rank, 0, top_k - 1))
    goal_x = float(args.goal_x)
    goal_y = float(args.goal_y)
    goal_z = float(args.goal_z)
    goal_radius = max(0.0, float(args.goal_radius))
    if args.step_dt is not None:
        step_dt = max(1e-4, float(args.step_dt))
    elif sim_dt is not None:
        step_dt = max(1e-4, float(sim_dt))
    else:
        step_dt = 0.05

    fig = plt.figure(figsize=(13, 8))
    ax3d = fig.add_subplot(2, 2, (1, 3), projection="3d")
    ax_bar = fig.add_subplot(2, 2, 2)
    ax_txt = fig.add_subplot(2, 2, 4)
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.24, top=0.92, wspace=0.28, hspace=0.30)

    all_valid_xyz = xyz[valid]
    if all_valid_xyz.size == 0:
        all_valid_xyz = xyz.reshape(-1, 3)

    if goal_radius > 0.0:
        sphere_extent = np.asarray(
            [
                [goal_x - goal_radius, goal_y - goal_radius, goal_z - goal_radius],
                [goal_x + goal_radius, goal_y + goal_radius, goal_z + goal_radius],
            ],
            dtype=np.float32,
        )
        all_valid_xyz = np.concatenate([all_valid_xyz, sphere_extent], axis=0)

    range_pts = all_valid_xyz
    if obstacle_env_points.ndim == 2 and obstacle_env_points.shape[1] == 3 and obstacle_env_points.size > 0:
        env_obs = obstacle_env_points[np.isfinite(obstacle_env_points).all(axis=1)]
        if env_obs.size > 0:
            range_pts = np.concatenate([range_pts, env_obs], axis=0)
    elif obstacle_points.ndim == 3 and obstacle_points.size > 0:
        flat_obs = obstacle_points.reshape(-1, 3)
        flat_obs = flat_obs[np.isfinite(flat_obs).all(axis=1)]
        if flat_obs.size > 0:
            range_pts = np.concatenate([range_pts, flat_obs], axis=0)

    x_min, y_min, z_min = np.min(range_pts, axis=0)
    x_max, y_max, z_max = np.max(range_pts, axis=0)
    margin_xy = 0.8
    margin_z = 0.5
    x_min, x_max = x_min - margin_xy, x_max + margin_xy
    y_min, y_max = y_min - margin_xy, y_max + margin_xy
    z_min, z_max = z_min - margin_z, z_max + margin_z

    line, = ax3d.plot([], [], [], linewidth=1.4, alpha=0.85, label="trajectory")
    obstacle_scatter = ax3d.scatter([], [], [], s=4, c="gray", alpha=0.15, label="obstacles")
    if goal_radius > 0.0:
        _draw_goal_sphere(ax3d, goal_x, goal_y, goal_z, goal_radius)
    ax3d.scatter([goal_x], [goal_y], [goal_z], s=20, c="green", alpha=0.9, label="goal center")
    point = ax3d.scatter([], [], [], s=60, c="r", label="current")

    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.set_xlim(float(x_min), float(x_max))
    ax3d.set_ylim(float(y_min), float(y_max))
    ax3d.set_zlim(float(z_min), float(z_max))
    ax3d.set_box_aspect((float(x_max - x_min), float(y_max - y_min), float(z_max - z_min)))
    ax3d.set_proj_type("ortho")
    ax3d.legend(loc="upper right")

    if factor_names:
        bars = ax_bar.bar(range(len(factor_names)), np.zeros(len(factor_names), dtype=np.float32), color="tab:blue", alpha=0.8)
        ax_bar.set_xticks(range(len(factor_names)))
        ax_bar.set_xticklabels(factor_names, rotation=45, ha="right", fontsize=9)
    else:
        bars = []
        ax_bar.text(0.5, 0.5, "No reward factors found", ha="center", va="center", transform=ax_bar.transAxes)

    ax_bar.set_title("Per-step reward factors")
    ax_bar.set_ylabel("value")

    ax_txt.axis("off")
    info_text = ax_txt.text(0.01, 0.96, "", va="top", ha="left", fontsize=11)

    rank_slider_ax = fig.add_axes([0.12, 0.10, 0.76, 0.03])
    rank_slider = Slider(rank_slider_ax, "rank", 0, top_k - 1, valinit=rank_init, valstep=1)
    step_slider_ax = fig.add_axes([0.12, 0.05, 0.76, 0.03])
    step_slider = Slider(step_slider_ax, "step", 0, 1, valinit=0, valstep=1)
    play_button_ax = fig.add_axes([0.02, 0.05, 0.08, 0.08])
    play_button = Button(play_button_ax, "Play")

    state = {
        "rank": rank_init,
        "step": 0,
        "step_max": 0,
        "step_dt": step_dt,
        "playing": False,
        "play_t0": None,
        "play_start_step": 0,
        "suspend": False,
        "cylinder_cache": {},
        "cylinder_artists": [],
    }

    timer = fig.canvas.new_timer(interval=15)

    has_env_obstacles = (
        obstacle_env_points.ndim == 2
        and obstacle_env_points.shape[1] == 3
        and obstacle_env_points.shape[0] > 0
        and np.isfinite(obstacle_env_points).all()
    )

    def _get_traj(rank_idx):
        step_count = _safe_step_count(valid[rank_idx])
        traj_xyz = xyz[rank_idx, :step_count]
        traj_done = done[rank_idx, :step_count]
        traj_reward_total = reward_total[rank_idx, :step_count]
        traj_speed = speed_mps[rank_idx, :step_count]
        if factor_mat.shape[-1] > 0:
            traj_factors = factor_mat[rank_idx, :step_count]
        else:
            traj_factors = np.zeros((step_count, 0), dtype=np.float32)
        return step_count, traj_xyz, traj_done, traj_reward_total, traj_speed, traj_factors

    def _render(rank_idx, step_idx):
        step_count, traj_xyz, traj_done, traj_reward_total, traj_speed, traj_factors = _get_traj(rank_idx)
        step_idx = int(np.clip(step_idx, 0, step_count - 1))
        x_vals = traj_xyz[:, 0]
        y_vals = traj_xyz[:, 1]
        z_vals = traj_xyz[:, 2]
        x, y, z = traj_xyz[step_idx]

        line.set_data_3d(x_vals, y_vals, z_vals)
        point._offsets3d = ([x], [y], [z])

        for artist in state["cylinder_artists"]:
            try:
                artist.remove()
            except Exception:
                pass
        state["cylinder_artists"] = []

        if has_env_obstacles:
            obs_pts = obstacle_env_points
        else:
            obs_pts = obstacle_points[rank_idx] if obstacle_points.ndim == 3 else np.zeros((0, 3), dtype=np.float32)
        if obs_pts.size > 0:
            obs_mask = np.isfinite(obs_pts).all(axis=1)
            obs_pts = obs_pts[obs_mask]

        if rank_idx not in state["cylinder_cache"]:
            state["cylinder_cache"][rank_idx] = _extract_cylinder_models(obs_pts)
        models = state["cylinder_cache"][rank_idx]

        if len(models) > 0:
            for cx, cy, rad, z0, z1, _ in models:
                state["cylinder_artists"].append(_draw_cylinder(ax3d, cx, cy, rad, z0, z1))
            obstacle_scatter._offsets3d = ([], [], [])
        elif obs_pts.size > 0:
            # Fallback to points if cylinders cannot be reconstructed.
            obstacle_scatter._offsets3d = (obs_pts[:, 0], obs_pts[:, 1], obs_pts[:, 2])
        else:
            obstacle_scatter._offsets3d = ([], [], [])

        if bars:
            vals = traj_factors[step_idx]
            lo = float(np.nanmin(vals)) if vals.size > 0 else -1.0
            hi = float(np.nanmax(vals)) if vals.size > 0 else 1.0
            pad = max(1e-3, 0.1 * max(abs(lo), abs(hi), 1.0))
            ax_bar.set_ylim(lo - pad, hi + pad)
            for i, b in enumerate(bars):
                b.set_height(float(vals[i]))

        done_flag = bool(traj_done[step_idx])
        speed_now = float(traj_speed[step_idx]) if np.isfinite(traj_speed[step_idx]) else float("nan")
        speed_mean = float(np.nanmean(traj_speed)) if np.isfinite(traj_speed).any() else float("nan")
        text_lines = [
            f"rank={rank_idx}",
            f"env_id={int(env_ids[rank_idx])}",
            f"success={float(success[rank_idx]):.0f}",
            f"death_reason={str(death_reason_name[rank_idx])} (code={int(death_reason_code[rank_idx])})",
            f"obstacle_cylinders={len(models)}",
            f"episode_return={float(returns[rank_idx]):.3f}",
            f"step={step_idx}/{step_count - 1}",
            f"xyz=({x:.3f}, {y:.3f}, {z:.3f})",
            f"speed_now={speed_now:.3f} m/s",
            f"speed_mean={speed_mean:.3f} m/s",
            f"reward_total={float(traj_reward_total[step_idx]):.4f}",
            f"done={done_flag}",
        ]
        info_text.set_text("\n".join(text_lines))

        ax3d.set_title(
            f"Drone rank={rank_idx} env={int(env_ids[rank_idx])} step={step_idx}"
        )
        fig.canvas.draw_idle()

    def _sync_step_slider_for_rank(rank_idx):
        step_count, _, _, _, _, _ = _get_traj(rank_idx)
        new_step_max = max(0, step_count - 1)
        state["step_max"] = new_step_max
        step_slider.valmax = new_step_max
        step_slider.ax.set_xlim(step_slider.valmin, step_slider.valmax)
        state["step"] = int(min(state["step"], new_step_max))

    def _reset_play_clock():
        state["play_t0"] = perf_counter()
        state["play_start_step"] = int(state["step"])

    def _set_playing(enabled: bool):
        state["playing"] = bool(enabled)
        if state["playing"]:
            _reset_play_clock()
            play_button.label.set_text("Pause")
        else:
            play_button.label.set_text("Play")

    def _on_timer_tick():
        if not state["playing"]:
            return
        elapsed = perf_counter() - state["play_t0"]
        delta_steps = int(elapsed / state["step_dt"])
        target_step = int(min(state["step_max"], state["play_start_step"] + delta_steps))
        if target_step != state["step"]:
            step_slider.set_val(target_step)
        if target_step >= state["step_max"]:
            _set_playing(False)

    def on_rank_changed(rank_val):
        if state["suspend"]:
            return
        state["rank"] = int(rank_val)
        state["suspend"] = True
        _sync_step_slider_for_rank(state["rank"])
        step_slider.set_val(state["step"])
        state["suspend"] = False
        if state["playing"]:
            _reset_play_clock()
        _render(state["rank"], state["step"])

    def on_step_changed(step_val):
        if state["suspend"]:
            return
        state["step"] = int(step_val)
        if state["playing"]:
            _reset_play_clock()
        _render(state["rank"], state["step"])

    def on_play_clicked(_event):
        _set_playing(not state["playing"])

    _sync_step_slider_for_rank(rank_init)
    rank_slider.on_changed(on_rank_changed)
    step_slider.on_changed(on_step_changed)
    play_button.on_clicked(on_play_clicked)
    timer.add_callback(_on_timer_tick)
    timer.start()
    _render(rank_init, 0)

    print("Loaded file:", npz_path)
    print("Available top-k trajectories:", xyz.shape[0])
    print("Viewing rank:", rank_init, "env_id:", int(env_ids[rank_init]))
    print("Goal sphere center=(%.2f, %.2f, %.2f), radius=%.2f" % (goal_x, goal_y, goal_z, goal_radius))
    print("Playback step dt=%.4f s/step" % state["step_dt"])
    print("Controls: drag rank/step sliders for manual mode, click Play for real-time auto mode.")
    if has_env_obstacles:
        print("Obstacle source: environment mesh points from USD stage (ground-truth).")
    else:
        print("Obstacle source: exported obstacle_points fallback (env mesh points unavailable in this npz).")

    plt.show()


if __name__ == "__main__":
    main()
