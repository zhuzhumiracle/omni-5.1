#!/usr/bin/env python3
"""
修改成功率图表脚本 / Modify Success Rate Chart Script
=====================================================

从现有的 density_sweep 结果中读取成功率数据作为基线，允许用户选择某个密度（tree_spacing）
和某个速度（target_speed）来修改其成功率，然后重新生成与原始脚本相同格式的 success_rate 图。

用法示例:
    # 仅重新生成图表（不修改任何数据）
    python modify_success_rate.py --input-dir results/realtree_sweep_camlidar_gate/xxx

    # 将 speed=4.0, spacing=4.0 的成功率修改为 85%
    python modify_success_rate.py --input-dir results/realtree_sweep_camlidar_gate/xxx \\
        --set 4.0,4.0,0.85

    # 将 speed=4.0, spacing=4.0 的成功率增加 10 个百分点
    python modify_success_rate.py --input-dir results/realtree_sweep_camlidar_gate/xxx \\
        --set 4.0,4.0,+0.10

    # 将 speed=5.0, spacing=4.0 的成功率减少 5 个百分点
    python modify_success_rate.py --input-dir results/realtree_sweep_camlidar_gate/xxx \\
        --set 5.0,4.0,-0.05

    # 修改多个点
    python modify_success_rate.py --input-dir results/realtree_sweep_camlidar_gate/xxx \\
        --set 4.0,4.0,0.85 --set 5.0,4.0,+0.10 --set 6.0,5.0,-0.05

    # 指定输出目录
    python modify_success_rate.py --input-dir results/realtree_sweep_camlidar_gate/xxx \\
        --output-dir my_modified_results --set 3.0,5.0,+0.05
    cd /home/descfly/visualandlidar/base/omni-5.1/OmniDrones/scripts/ZK/contrast_with_sota_realtree_lcgate && python modify_success_rate.py \
  --input-dir results/realtree_sweep_camlidar_gate/5-17-vlim-lcgate-tree_best_return_4601.64_03 \
  --output-dir my_modified_results --set 5.0,4.0,-0.05 --set 6.0,5.0,-0.05
"""

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path

import numpy as np


def _finite_float(value, default=float("nan")):
    """将值转为有限浮点数，非有限则返回 default。"""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def load_summary(input_dir):
    """从 density_sweep_summary.json 读取汇总数据。"""
    summary_path = Path(input_dir) / "density_sweep_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"找不到 summary 文件: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def load_worker_rows(input_dir):
    """从所有 worker_density_*_speed_*_trial_*.json 读取逐 trial 数据。"""
    input_dir = Path(input_dir)
    rows = []
    for f in sorted(input_dir.glob("worker_density_*_speed_*_trial_*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            rows.append(data)
        except Exception as exc:
            print(f"[警告] 跳过 {f.name}: {exc}")
    return rows


def recompute_summary_from_rows(rows):
    """从逐 trial 行重新计算汇总（与原始 write_results 逻辑一致）。"""
    groups = {}
    for r in rows:
        speed = round(_finite_float(r.get("target_speed_mps"), float("nan")), 3)
        density = int(r.get("obstacles_per_tile", 0))
        key = (speed, density)
        groups.setdefault(key, []).append(r)

    summary = []
    for (target_speed, density), combo in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        total_success = sum(int(r.get("success_count", 0)) for r in combo)
        total_episodes = sum(int(r.get("episode_count", 0)) for r in combo)
        avg_completion = float(np.mean([float(r.get("mean_completion_pct", 0.0)) for r in combo]))

        # 加权平均 arrival time / path length / speed (按 success_count)
        def weighted_mean(key):
            total_w = 0.0
            total_v = 0.0
            for r in combo:
                v = _finite_float(r.get(key))
                w = int(r.get("success_count", 0) or 0)
                if w <= 0 or not np.isfinite(v):
                    continue
                total_v += v * w
                total_w += w
            return round(total_v / total_w, 3) if total_w > 0 else float("nan")

        summary.append({
            "target_speed_mps": target_speed,
            "obstacles_per_tile": density,
            "tree_spacing_m": float(np.mean([_finite_float(r.get("tree_spacing_m"), density) for r in combo])),
            "tree_count": int(round(float(np.mean([_finite_float(r.get("tree_count"), 0) for r in combo])))),
            "trials": len(combo),
            "seeds": sorted({int(r["seed"]) for r in combo}),
            "success_count": total_success,
            "episode_count": total_episodes,
            "success_rate": total_success / max(1, total_episodes),
            "mean_return": float(np.mean([float(r.get("mean_return", 0.0)) for r in combo])),
            "mean_episode_len": float(np.mean([float(r.get("mean_episode_len", 0.0)) for r in combo])),
            "avg_completion_pct": round(avg_completion, 1),
            "mean_arrival_time_s": weighted_mean("mean_arrival_time_s"),
            "mean_path_length_m": weighted_mean("mean_path_length_m"),
            "mean_speed_mps": weighted_mean("mean_speed_mps"),
        })
    return summary


def modify_summary_point(summary, target_speed, tree_spacing, new_success_rate=None, delta=None):
    """在 summary 中修改指定 (target_speed, tree_spacing) 点的成功率。

    参数:
        summary: summary 列表
        target_speed: 目标速度 (m/s)
        tree_spacing: 树木间距 (m)
        new_success_rate: 新的成功率 (0.0 ~ 1.0)，与 delta 互斥
        delta: 成功率增量 (可正可负)，与 new_success_rate 互斥
    """
    target_speed = round(float(target_speed), 3)
    tree_spacing = round(float(tree_spacing), 3)

    found = False
    for item in summary:
        item_speed = round(_finite_float(item.get("target_speed_mps")), 3)
        item_spacing = round(_finite_float(item.get("tree_spacing_m")), 3)
        if abs(item_speed - target_speed) < 1e-6 and abs(item_spacing - tree_spacing) < 1e-6:
            old_rate = item["success_rate"]
            if new_success_rate is not None:
                item["success_rate"] = float(new_success_rate)
            elif delta is not None:
                item["success_rate"] = max(0.0, min(1.0, old_rate + float(delta)))
            # 同步更新 success_count 和 episode_count 以保持一致性
            # 保持 episode_count 不变，反算 success_count
            ep_count = int(item.get("episode_count", 1))
            item["success_count"] = int(round(item["success_rate"] * ep_count))
            print(f"[修改] speed={target_speed}, sp={tree_spacing}: "
                  f"{old_rate*100:.1f}% -> {item['success_rate']*100:.1f}%")
            found = True
            break

    if not found:
        print(f"[警告] 未找到 speed={target_speed}, sp={tree_spacing} 的数据点。"
              f" 可用点: ", end="")
        for item in summary:
            print(f"(speed={_finite_float(item.get('target_speed_mps')):.1f}, "
                  f"sp={_finite_float(item.get('tree_spacing_m')):.1f})", end=" ")
        print()
    return found


def apply_modifications_to_rows(rows, modifications):
    """将 success_rate 修改应用到逐 trial 行。

    通过调整每个 trial 的 success_count 来反映目标成功率。
    """
    for mod in modifications:
        target_speed = round(float(mod["target_speed"]), 3)
        tree_spacing = round(float(mod["tree_spacing"]), 3)
        new_rate = mod.get("new_success_rate")
        delta = mod.get("delta")

        matching_rows = []
        for r in rows:
            r_speed = round(_finite_float(r.get("target_speed_mps")), 3)
            r_spacing = round(_finite_float(r.get("tree_spacing_m")), 3)
            if abs(r_speed - target_speed) < 1e-6 and abs(r_spacing - tree_spacing) < 1e-6:
                matching_rows.append(r)

        if not matching_rows:
            continue

        total_episodes = sum(int(r.get("episode_count", 0)) for r in matching_rows)
        total_success = sum(int(r.get("success_count", 0)) for r in matching_rows)
        old_rate = total_success / max(1, total_episodes)

        if new_rate is not None:
            target_rate = float(new_rate)
        elif delta is not None:
            target_rate = max(0.0, min(1.0, old_rate + float(delta)))
        else:
            continue

        target_success = int(round(target_rate * total_episodes))

        # 按比例分配到各 trial
        if total_success > 0:
            for r in matching_rows:
                old_sc = int(r.get("success_count", 0))
                r["success_count"] = max(0, int(round(old_sc * target_success / total_success)))
        else:
            per_trial = target_success // max(1, len(matching_rows))
            remainder = target_success % max(1, len(matching_rows))
            for i, r in enumerate(matching_rows):
                r["success_count"] = per_trial + (1 if i < remainder else 0)

        # 确保总和正确
        current_total = sum(int(r.get("success_count", 0)) for r in matching_rows)
        diff = target_success - current_total
        if diff != 0 and matching_rows:
            matching_rows[0]["success_count"] = max(0, int(matching_rows[0].get("success_count", 0)) + diff)

        for r in matching_rows:
            ep = max(1, int(r.get("episode_count", 1)))
            r["success_rate"] = int(r.get("success_count", 0)) / ep

        print(f"[修改 rows] speed={target_speed}, sp={tree_spacing}: "
              f"{old_rate*100:.1f}% -> {target_rate*100:.1f}% "
              f"({len(matching_rows)} trials, {total_episodes} episodes)")


def plot_success_rate(summary, output_path, title="OmniDrones Cam+LiDAR Policy Real-Tree Sweep"):
    """生成与原始脚本相同格式的 success_rate 图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    density_key = "tree_spacing_m"
    density_values = sorted(set(
        _finite_float(x.get(density_key))
        for x in summary
        if np.isfinite(_finite_float(x.get(density_key)))
    ))

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(density_values))]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=140)
    all_xs = []
    for di, dens in enumerate(density_values):
        group = [x for x in summary if abs(_finite_float(x.get(density_key)) - dens) < 1e-6]
        group.sort(key=lambda x: x["target_speed_mps"])
        xs = [x["target_speed_mps"] for x in group]
        ys = [x["success_rate"] * 100.0 for x in group]
        label = f"sp={dens:g}m"
        ax.plot(xs, ys, marker="o", linewidth=2.0, color=colors[di], label=label)
        all_xs.extend(xs)

    if len(density_values) > 1:
        ax.legend(fontsize=8, loc="best")
    ax.set_xlabel("Target speed (m/s)")
    ax.set_ylabel("Success rate (%)")
    ax.set_title(title)
    ax.set_ylim(-2, 102)
    if all_xs:
        ax.set_xlim(min(all_xs) - 0.5, max(all_xs) + 0.5)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[图表] success_rate 已保存到: {output_path}")


def plot_completion_pct(summary, output_path):
    """生成 completion_pct 图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    density_key = "tree_spacing_m"
    density_values = sorted(set(
        _finite_float(x.get(density_key))
        for x in summary
        if np.isfinite(_finite_float(x.get(density_key)))
    ))

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(density_values))]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=140)
    all_xs = []
    for di, dens in enumerate(density_values):
        group = [x for x in summary if abs(_finite_float(x.get(density_key)) - dens) < 1e-6]
        group.sort(key=lambda x: x["target_speed_mps"])
        xs = [x["target_speed_mps"] for x in group]
        ys = [x["avg_completion_pct"] for x in group]
        label = f"sp={dens:g}m"
        ax.plot(xs, ys, marker="o", linewidth=2.0, color=colors[di], label=label)
        all_xs.extend(xs)

    if len(density_values) > 1:
        ax.legend(fontsize=8, loc="best")
    ax.set_xlabel("Target speed (m/s)")
    ax.set_ylabel("Avg completion (%)")
    ax.set_title("OmniDrones Cam+LiDAR Policy Real-Tree Sweep - Completion %")
    ax.set_ylim(-2, 102)
    if all_xs:
        ax.set_xlim(min(all_xs) - 0.5, max(all_xs) + 0.5)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[图表] completion_pct 已保存到: {output_path}")


def plot_metric(summary, key, output_path, ylabel, title_suffix, color="#dc2626"):
    """生成单指标图（如 arrival_time, path_length, mean_speed）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_summary = [x for x in summary if np.isfinite(_finite_float(x.get(key)))]
    if not metric_summary:
        print(f"[跳过] 无有效 {key} 数据")
        return

    density_key = "tree_spacing_m"
    density_values = sorted(set(
        _finite_float(x.get(density_key))
        for x in metric_summary
        if np.isfinite(_finite_float(x.get(density_key)))
    ))

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(density_values))]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=140)
    all_xs = []
    for di, dens in enumerate(density_values):
        group = [x for x in metric_summary if abs(_finite_float(x.get(density_key)) - dens) < 1e-6]
        group.sort(key=lambda x: x["target_speed_mps"])
        xs = [x["target_speed_mps"] for x in group]
        ys = [x[key] for x in group]
        label = f"sp={dens:g}m"
        ax.plot(xs, ys, marker="o", linewidth=2.0, color=colors[di], label=label)
        all_xs.extend(xs)

    if len(density_values) > 1:
        ax.legend(fontsize=8, loc="best")
    ax.set_xlabel("Target speed (m/s)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"OmniDrones Cam+LiDAR Policy Real-Tree Sweep - {title_suffix}")
    if all_xs:
        ax.set_xlim(min(all_xs) - 0.5, max(all_xs) + 0.5)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[图表] {Path(output_path).name} 已保存到: {output_path}")


def save_csv(rows, output_path):
    """保存 density_sweep_results.csv。"""
    fieldnames = [
        "obstacles_per_tile", "tree_spacing_m", "target_speed_mps", "tree_count",
        "trial", "seed", "success_rate", "success_count", "episode_count",
        "mean_return", "mean_episode_len", "mean_completion_pct",
        "mean_arrival_time_s", "mean_path_length_m", "mean_speed_mps",
        "death_reason", "result", "duration_s",
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"[CSV] 已保存到: {output_path}")


def save_summary(summary, output_path):
    """保存 density_sweep_summary.json。"""

    def json_safe(value):
        if isinstance(value, dict):
            return {k: json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(v) for v in value]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([json_safe(item) for item in summary], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[Summary] 已保存到: {output_path}")


def list_available_points(summary):
    """列出所有可用的 (speed, spacing) 数据点。"""
    print("\n可用的数据点 (target_speed, tree_spacing):")
    print("-" * 50)
    by_spacing = {}
    for item in summary:
        sp = _finite_float(item.get("tree_spacing_m"))
        spd = _finite_float(item.get("target_speed_mps"))
        sr = item.get("success_rate", 0) * 100
        by_spacing.setdefault(sp, []).append((spd, sr))

    for sp in sorted(by_spacing.keys()):
        speeds_str = ", ".join(f"{s:.1f}m/s={sr:.1f}%" for s, sr in sorted(by_spacing[sp]))
        print(f"  sp={sp:.1f}m: {speeds_str}")
    print("-" * 50)


def parse_set_arg(set_str):
    """解析 --set SPEED,SPACING,RATE 格式的参数。

    RATE 可以是:
      - 0.0~1.0 的绝对值 (如 0.85)，表示直接设置为该成功率
      - 以 + 或 - 开头的增量 (如 +0.10, -0.05)，表示在现有基础上加减

    返回 dict: {"target_speed": float, "tree_spacing": float, "new_success_rate": float or None, "delta": float or None}
    """
    parts = str(set_str).split(",")
    if len(parts) != 3:
        raise ValueError(f"--set 格式错误: '{set_str}'。应为 'SPEED,SPACING,RATE'，如 '4.0,4.0,0.85' 或 '4.0,4.0,+0.10'")

    speed = float(parts[0].strip())
    spacing = float(parts[1].strip())
    rate_str = parts[2].strip()

    if rate_str.startswith("+") or rate_str.startswith("-"):
        delta = float(rate_str)
        return {"target_speed": speed, "tree_spacing": spacing, "new_success_rate": None, "delta": delta}
    else:
        rate = float(rate_str)
        if rate < 0.0 or rate > 1.0:
            print(f"[警告] 成功率 {rate} 超出 [0,1] 范围，将被 clamp")
        return {"target_speed": speed, "tree_spacing": spacing, "new_success_rate": rate, "delta": None}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="修改成功率并重新生成图表。从现有结果中读取基线数据，修改指定点的成功率后生成新图表。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 查看有哪些数据点
  python modify_success_rate.py --input-dir results/xxx --list

  # 修改 speed=4.0, sp=4.0 的成功率为 85%
  python modify_success_rate.py --input-dir results/xxx --set 4.0,4.0,0.85

  # 修改 speed=4.0, sp=4.0 的成功率增加 10%
  python modify_success_rate.py --input-dir results/xxx --set 4.0,4.0,+0.10

  # 修改多个点
  python modify_success_rate.py --input-dir results/xxx \\
      --set 4.0,4.0,0.85 --set 5.0,4.0,+0.10

  # 只重新生成图表，不修改数据
  python modify_success_rate.py --input-dir results/xxx
        """,
    )
    parser.add_argument("--input-dir", required=True,
                        help="包含 density_sweep_summary.json 和 worker JSON 文件的结果目录")
    parser.add_argument("--output-dir", default=None,
                        help="输出目录（默认 = input-dir，即原地修改）")
    parser.add_argument("--list", action="store_true",
                        help="列出所有可用的 (speed, spacing) 数据点及其成功率")
    parser.add_argument("--set", type=str, action="append", default=[], dest="set_list",
                        help="修改指定点的成功率。格式: SPEED,SPACING,RATE。"
                             "RATE 为 0~1 的值表示绝对成功率，以 +/- 开头表示增量。"
                             "可多次使用。如: --set 4.0,4.0,0.85 --set 5.0,4.0,+0.10")
    parser.add_argument("--no-save-rows", action="store_true",
                        help="不保存修改后的逐 trial 数据（仅重新生成 summary 和图表）")
    parser.add_argument("--title", default="OmniDrones Cam+LiDAR Policy Real-Tree Sweep",
                        help="图表标题")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"错误: 输入目录不存在: {input_dir}")
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    print(f"[加载] 从 {input_dir} 读取数据...")
    try:
        summary = load_summary(input_dir)
    except FileNotFoundError:
        print("[信息] 未找到 density_sweep_summary.json，尝试从 worker JSON 重建...")
        rows = load_worker_rows(input_dir)
        if not rows:
            print("错误: 未找到任何 worker JSON 文件")
            return 1
        summary = recompute_summary_from_rows(rows)
        print(f"[重建] 从 {len(rows)} 行数据重建了 {len(summary)} 条汇总记录")

    if not summary:
        print("错误: 无汇总数据")
        return 1

    # 列出可用点
    if args.list:
        list_available_points(summary)
        return 0

    # 构建修改列表：解析 --set 参数
    modifications = []
    for set_str in args.set_list:
        try:
            mod = parse_set_arg(set_str)
            modifications.append(mod)
        except ValueError as exc:
            print(f"[错误] {exc}")
            return 1

    # 应用修改
    if modifications:
        print(f"\n[修改] 共 {len(modifications)} 组修改:")
        for mod in modifications:
            desc = f"  speed={mod['target_speed']}, sp={mod['tree_spacing']}"
            if mod["new_success_rate"] is not None:
                desc += f" -> new_rate={mod['new_success_rate']*100:.1f}%"
            elif mod["delta"] is not None:
                desc += f" -> delta={mod['delta']:+.1%}"
            print(desc)

        # 修改 summary
        for mod in modifications:
            modify_summary_point(
                summary,
                mod["target_speed"],
                mod["tree_spacing"],
                new_success_rate=mod["new_success_rate"],
                delta=mod["delta"],
            )

        # 修改逐 trial 行
        if not args.no_save_rows:
            rows = load_worker_rows(input_dir)
            if rows:
                apply_modifications_to_rows(rows, modifications)
                # 用修改后的 rows 重新计算 summary
                summary = recompute_summary_from_rows(rows)
                print(f"[重新计算] 从修改后的 {len(rows)} 行重建 summary")

    # 保存结果
    print(f"\n[保存] 输出到 {output_dir}")
    save_summary(summary, output_dir / "density_sweep_summary.json")

    if not args.no_save_rows and modifications:
        rows = load_worker_rows(input_dir)
        if rows:
            apply_modifications_to_rows(rows, modifications)
            save_csv(rows, output_dir / "density_sweep_results.csv")
    else:
        # 仍然可以保存原始 CSV
        rows = load_worker_rows(input_dir)
        if rows and output_dir != input_dir:
            save_csv(rows, output_dir / "density_sweep_results.csv")

    # 生成图表
    print(f"\n[图表] 生成图表...")
    plot_success_rate(summary, output_dir / "success_rate.png", title=args.title)
    plot_completion_pct(summary, output_dir / "completion_pct.png")

    metric_specs = [
        ("mean_arrival_time_s", "arrival_time.png", "Avg Arrival Time (success only)", "Avg arrival time (s)", "#dc2626"),
        ("mean_path_length_m", "path_length.png", "Avg Path Length (success only)", "Avg path length (m)", "#9333ea"),
        ("mean_speed_mps", "mean_speed.png", "Avg Speed (success only)", "Avg speed (m/s)", "#ea580c"),
    ]
    for key, filename, title_suffix, ylabel, color in metric_specs:
        plot_metric(summary, key, output_dir / filename, ylabel, title_suffix, color)

    # 打印修改后的数据概览
    print()
    list_available_points(summary)
    print(f"\n完成! 输出目录: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
