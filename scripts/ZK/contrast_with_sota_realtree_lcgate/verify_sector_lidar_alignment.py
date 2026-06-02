#!/usr/bin/env python3
"""
Camera‑Sector‑LiDAR Alignment Verification Script (v2)

Verifies spatial alignment between depth camera, 2D sector gates, and LiDAR KU map.

Key fixes vs v1:
  - Camera target now [0.0, -2.0, 0.95] matching forest_lc_gate.yaml body -Y nose axis.
  - KU pitch mapping validated against environment code:
    env uses  pitch_bin = ((pitch + π/2) / π) * num_pitch_bins  → [-90°, 90°] to [0, 39]
  - Added pixel‑to‑ray projection: camera pixel → yaw/pitch → KU row/col
  - Dry-run and live modes both use consistent config

Modes:
  --dry-run    Compute sector masks from config params (no Isaac Sim needed)
  (default)    Start Isaac Sim headless, read real USD camera geometry

Usage:
  python verify_sector_lidar_alignment.py --dry-run --camera-risk-layout 3x3
  DISPLAY=:2 CUDA_VISIBLE_DEVICES=1 python verify_sector_lidar_alignment.py --camera-risk-layout 3x3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ZK_DIR = SCRIPT_DIR.parent
OMNIDRONES_DIR = ZK_DIR.parent.parent
CFG_DIR = OMNIDRONES_DIR / "cfg"

# ---- Config matching cfg/task/forest_lc_gate.yaml ----
# Camera tilted upward ~23.4° so its vertical FoV lands within LiDAR [-7°, 52°]
_DEFAULT_CAM_POS = [0.0, -0.22, 0.18]
_DEFAULT_CAM_TARGET = [0.0, -2.0, 0.95]
_DEFAULT_FOCAL_MM = 12.0
_DEFAULT_H_APERTURE_MM = 20.955
_DEFAULT_DEPTH_W = 160
_DEFAULT_DEPTH_H = 96
_DEFAULT_LIDAR_VFOV_MIN = -7.0
_DEFAULT_LIDAR_VFOV_MAX = 52.0


# ============================================================
# Sector mask builder (mirrors DualStreamBackbone._build_sector_2d_masks)
# ============================================================

def build_sector_masks(
    ku_h: int = 40,
    ku_w: int = 80,
    camera_h_fov_rad: float | None = None,
    fov_pitch_min: float = -math.pi / 2,
    fov_pitch_max: float = math.pi / 2,
    num_rows: int = 3,
    num_cols: int = 3,
    camera_yaw_center_rad: float = -math.pi / 2,
    lidar_pitch_min: float = -math.pi / 2,
    lidar_pitch_max: float = math.pi / 2,
) -> tuple[list[np.ndarray], dict]:
    """
    Build 2D boolean masks [ku_h, ku_w] for each (row, col) sector.

    LiDAR KU grid binning (matching env code):
      yaw_bin  = ((yaw_remapped_to_0_2π) / 2π) * ku_w   → 0..ku_w-1
      pitch_bin= ((pitch + π/2) / π) * ku_h               → 0..ku_h-1  (i.e. [-90°,+90°] → [0,39])
    """
    if camera_h_fov_rad is None:
        camera_h_fov_rad = 2.0 * math.atan(_DEFAULT_DEPTH_W / (2.0 * 91.6249))
    h_fov = float(camera_h_fov_rad)
    p_min, p_max = float(fov_pitch_min), float(fov_pitch_max)

    # Yaw: 0→2π mapped to 0→ku_w-1.  For FOV masks we use the [-π, π] wrapped version.
    col_yaws = (np.arange(ku_w, dtype=np.float64) + 0.5) / ku_w * 2.0 * math.pi
    col_yaws_w = (col_yaws + math.pi) % (2.0 * math.pi) - math.pi

    # Pitch: [-π/2, π/2] (or lidar range) → [0, ku_h-1]
    p_span = lidar_pitch_max - lidar_pitch_min
    row_pitches = (np.arange(ku_h, dtype=np.float64) + 0.5) / ku_h * p_span + lidar_pitch_min

    yaw_grid = col_yaws_w[np.newaxis, :].repeat(ku_h, axis=0)    # [H,W]
    rel_yaw_grid = (yaw_grid - float(camera_yaw_center_rad) + math.pi) % (2.0 * math.pi) - math.pi
    pitch_grid = row_pitches[:, np.newaxis].repeat(ku_w, axis=1)  # [H,W]

    fov_mask = (
        (rel_yaw_grid >= -h_fov / 2.0) & (rel_yaw_grid <= h_fov / 2.0)
        & (pitch_grid >= p_min) & (pitch_grid <= p_max)
    )

    masks, sectors = [], []
    for ri in range(num_rows):
        pl = p_min + ri * (p_max - p_min) / num_rows
        pr = p_min + (ri + 1) * (p_max - p_min) / num_rows
        for ci in range(num_cols):
            yl = -h_fov / 2.0 + ci * h_fov / num_cols
            yr = -h_fov / 2.0 + (ci + 1) * h_fov / num_cols
            m = fov_mask & (rel_yaw_grid >= yl) & (rel_yaw_grid <= yr) & (pitch_grid >= pl) & (pitch_grid <= pr)
            masks.append(m)
            sectors.append({
                "row": ri, "col": ci,
                "pitch_deg": (math.degrees(pl), math.degrees(pr)),
                "yaw_deg": (math.degrees(yl), math.degrees(yr)),
                "pixel_count": int(m.sum()),
            })

    meta = {
        "ku_h": ku_h, "ku_w": ku_w,
        "num_rows": num_rows, "num_cols": num_cols,
        "camera_h_fov_rad": h_fov,
        "camera_h_fov_deg": math.degrees(h_fov),
        "camera_yaw_center_deg": math.degrees(float(camera_yaw_center_rad)),
        "fov_pitch_min_deg": math.degrees(p_min),
        "fov_pitch_max_deg": math.degrees(p_max),
        "fov_total_pixels": int(fov_mask.sum()),
        "fov_pct_of_sphere": float(fov_mask.sum()) / (ku_h * ku_w) * 100,
        "lidar_pitch_min_deg": math.degrees(lidar_pitch_min),
        "lidar_pitch_max_deg": math.degrees(lidar_pitch_max),
        "row_pitches_deg": [math.degrees(float(v)) for v in row_pitches],
        "col_yaws_deg": [math.degrees(float(v)) for v in col_yaws_w],
        "sectors": sectors,
    }
    return masks, meta


# ============================================================
# Camera extrinsic rotation (from pos + target, row-vector convention)
# ============================================================

def _normalize(v, eps=1e-12):
    n = float(np.linalg.norm(v))
    if n <= eps: raise ValueError("zero vector")
    return v / n

def camera_rotation_from_view(camera_pos, target_pos, up=None):
    """Build row-vector rotation matrix: rows are [right, up, -forward]."""
    if up is None: up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    fwd = _normalize(target_pos - camera_pos)
    up = _normalize(up)
    r = np.cross(fwd, up)
    if np.linalg.norm(r) <= 1e-9:
        r = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
        if np.linalg.norm(r) <= 1e-9: raise ValueError("degenerate")
    r = _normalize(r)
    u = _normalize(np.cross(r, fwd))
    return np.stack([r, u, -fwd], axis=0)  # row-vector: 3×3


# ============================================================
# Pixel-to-ray → KU row/col projection
# ============================================================

def project_pixels_to_ku(
    fx: float, fy: float, cx: float, cy: float,
    camera_h_fov_rad: float,
    fov_pitch_min: float, fov_pitch_max: float,
    ku_h: int = 40, ku_w: int = 80,
    lidar_pitch_min: float = -math.pi / 2,
    lidar_pitch_max: float = math.pi / 2,
    camera_yaw_center_rad: float = -math.pi / 2,
    num_test_rows: int = 3, num_test_cols: int = 3,
    depth_w: int = 160, depth_h: int = 96,
    cam_rot_body: np.ndarray | None = None,  # 3×3 row-vector: camera→body
) -> dict:
    """
    Back-project camera pixels to 3D rays, rotate to body frame,
    and map to LiDAR KU (row, col).

    Step 1: ray_cam = [(u-cx)/fx,  (cy-v)/fy,  -1.0]   (matches env's _precompute_camera_ray_dirs)
    Step 2: ray_body = ray_cam @ cam_rot_body            (row-vector convention)
    Step 3: yaw_body = atan2(ray_body[1], ray_body[0])   [body XY plane]
            pitch_body = atan2(ray_body[2], sqrt(x²+y²))
    Step 4: KU bin from body-frame (yaw, pitch)
    """
    if cam_rot_body is None:
        cam_rot_body = np.eye(3, dtype=np.float64)

    test_us = np.linspace(0, depth_w - 1, num_test_cols + 2)[1:-1]
    test_vs = np.linspace(0, depth_h - 1, num_test_rows + 2)[1:-1]

    results = []
    for v in test_vs:
        for u in test_us:
            # Camera-frame ray direction (matches env: dx=(u-cx)/fx, dy=(cy-v)/fy, dz=-1)
            rx_c = (float(u) - cx) / fx
            ry_c = (cy - float(v)) / fy   # env convention: cy - v
            rz_c = -1.0  # forward in USD camera convention
            ray_cam = np.array([rx_c, ry_c, rz_c], dtype=np.float64)

            # Rotate to body frame: ray_body = ray_cam @ R  (row-vector)
            ray_body = ray_cam @ cam_rot_body  # [3] @ [3,3] → [3]

            # Body-frame yaw/pitch
            yaw_body = float(np.arctan2(ray_body[1], ray_body[0]))  # atan2(y, x)
            horiz = np.sqrt(ray_body[0]**2 + ray_body[1]**2)
            pitch_body = float(np.arctan2(ray_body[2], horiz))

            # KU bin indices (matching env code)
            yw = yaw_body % (2.0 * math.pi)
            yb = int(np.clip(yw / (2.0 * math.pi) * ku_w, 0, ku_w - 1))
            p_span = lidar_pitch_max - lidar_pitch_min
            pb = int(np.clip((pitch_body - lidar_pitch_min) / p_span * ku_h, 0, ku_h - 1))

            # Which sector (based on body-frame angles)
            rel_yaw = (yaw_body - camera_yaw_center_rad + math.pi) % (2.0 * math.pi) - math.pi
            sc = int(np.clip((rel_yaw - (-camera_h_fov_rad / 2)) / camera_h_fov_rad * num_test_cols,
                             0, num_test_cols - 1)) \
                if -camera_h_fov_rad / 2 <= rel_yaw <= camera_h_fov_rad / 2 else -1
            sr = int(np.clip((pitch_body - fov_pitch_min) / (fov_pitch_max - fov_pitch_min) * num_test_rows,
                             0, num_test_rows - 1)) \
                if fov_pitch_min <= pitch_body <= fov_pitch_max else -1

            results.append({
                "u_px": int(u), "v_px": int(v),
                "yaw_deg": round(math.degrees(yaw_body), 3),
                "pitch_deg": round(math.degrees(pitch_body), 3),
                "ku_row": pb, "ku_col": yb,
                "sector": f"({sr},{sc})" if sr >= 0 and sc >= 0 else "outside",
            })

    return {"num_test_rows": num_test_rows, "num_test_cols": num_test_cols,
            "pixel_results": results}


# ============================================================
# ASCII heatmap
# ============================================================

_SCH = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def render_ascii(masks, meta):
    ku_h, ku_w = meta["ku_h"], meta["ku_w"]
    label = np.full((ku_h, ku_w), -1, dtype=np.int32)
    for i, m in enumerate(masks):
        label[m] = i

    lines = [
        f"LiDAR KU grid ({ku_h}×{ku_w}) — sector overlay",
        f"Legend: . = outside FOV, 0-{min(len(masks)-1,35)} = sector idx",
        f"Camera yaw={meta['camera_yaw_center_deg']:.1f}° hFoV={meta['camera_h_fov_deg']:.1f}°  "
        f"pitch=[{meta['fov_pitch_min_deg']:.1f}°, {meta['fov_pitch_max_deg']:.1f}°]",
        f"LiDAR pitch=[{meta['lidar_pitch_min_deg']:.1f}°, {meta['lidar_pitch_max_deg']:.1f}°]",
        f"FOV: {meta['fov_total_pixels']}/{ku_h*ku_w} px ({meta['fov_pct_of_sphere']:.1f}%)",
        "─" * 80,
    ]
    rs = max(1, ku_h // 20)
    cs = max(1, ku_w // 40)
    for r in range(0, ku_h, rs):
        chars = []
        for c in range(0, ku_w, cs):
            blk = label[r:r+rs, c:c+cs]
            if (blk == -1).all():
                chars.append(".")
            else:
                vv = blk[blk >= 0]
                chars.append(_SCH[min(int(np.argmax(np.bincount(vv))), len(_SCH)-1)])
        lines.append(f"{float(meta['row_pitches_deg'][r]):+6.1f}° {''.join(chars)}")
    lines.append("─" * 80)
    return "\n".join(lines)


# ============================================================
# Matplotlib
# ============================================================

_SCOL = ["#e6194b","#3cb44b","#ffe119","#4363d8","#f58231",
         "#911eb4","#42d4f4","#f032e6","#bfef45","#fabed4",
         "#469990","#dcbeff","#9a6324","#fffac8","#800000",
         "#aaffc3","#808000","#ffd8b1","#000075","#a9a9a9"]

def save_fig(masks, meta, pixel_proj, path, dpi=150):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] no matplotlib, skip figure")
        return False

    ku_h, ku_w = meta["ku_h"], meta["ku_w"]

    # ---- Build coloured image, then ROLL so yaw 0→360 matches KU column order ----
    img = np.ones((ku_h, ku_w, 3), np.float32) * 0.15
    for i, m in enumerate(masks):
        ch = _SCOL[i % len(_SCOL)]
        img[m] = [int(ch[1:3],16)/255, int(ch[3:5],16)/255, int(ch[5:7],16)/255]
    any_fov = np.zeros((ku_h, ku_w), bool)
    for m in masks: any_fov |= m
    img[~any_fov] *= 0.3

    # KU column 0 = yaw 0°, column ku_w-1 = yaw ~360°.  Currently yaw_wrapped in masks
    # is in [-π, π]; the KU image cols go 0→2π.  Roll so centre column becomes "straight ahead".
    roll_by = ku_w // 2  # bring the 180°→360° half to the right side
    img = np.roll(img, roll_by, axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7),
                                      gridspec_kw={"width_ratios": [2, 1]})
    fig.subplots_adjust(wspace=0.05)

    pext = [meta["lidar_pitch_min_deg"], meta["lidar_pitch_max_deg"]]
    ax1.imshow(img, origin="lower", aspect="auto", extent=[-180, 180, pext[0], pext[1]])
    ax1.set_xlabel("Yaw (deg)"); ax1.set_ylabel("Pitch (deg)")
    ax1.set_title(f"Sectors ({meta['num_rows']}×{meta['num_cols']}) on LiDAR KU\n"
                  f"hFoV={meta['camera_h_fov_deg']:.1f}°  "
                  f"pitch=[{meta['fov_pitch_min_deg']:.1f}°,{meta['fov_pitch_max_deg']:.1f}°]\n"
                  f"(yaw 0° at centre, KU cols 0→360°)")

    # Determine where the FOV centre sits in the rolled image
    # Before roll: centre yaw=0° maps to col ku_w/2
    # After rolling by ku_w/2, it maps to col 0 → x= -180° in extent
    # We can instead directly draw boundaries in the extent coordinate system.
    # The sector yaw boundaries in meta["sectors"][*]["yaw_deg"] are in [-180, 180].
    # After the roll, a yaw of y maps to rolled_col = (ku_w/2 + y/360*ku_w) % ku_w,
    # which in extent translates to x = (rolled_col/ku_w)*360 - 180.
    h_deg = meta["camera_h_fov_deg"]
    for ri in range(meta["num_rows"] + 1):
        py = meta["fov_pitch_min_deg"] + ri * (meta["fov_pitch_max_deg"] - meta["fov_pitch_min_deg"]) / meta["num_rows"]
        ax1.axhline(py, color="white", lw=0.8, ls="--", alpha=0.6)

    # Draw sector yaw boundaries in the rolled coordinate
    for ci in range(meta["num_cols"] + 1):
        yw_orig = -h_deg / 2 + ci * h_deg / meta["num_cols"]  # [-180, 180] yaw
        # Map to rolled extent x
        x_rolled = yw_orig  # after roll by half, the mapping stays the same conceptually
        ax1.axvline(x_rolled, color="white", lw=0.8, ls="--", alpha=0.6)

    if pixel_proj:
        for px in pixel_proj.get("pixel_results", []):
            # yaw_deg is in [-180, 180]; after image roll it's still in [-180, 180] extent
            yw = px["yaw_deg"]
            pt = px["pitch_deg"]
            if -h_deg / 2 <= yw <= h_deg / 2 and meta["fov_pitch_min_deg"] <= pt <= meta["fov_pitch_max_deg"]:
                ax1.plot(yw, pt, "w+", ms=8, mew=1.5)
                ax1.annotate(f"({px['u_px']},{px['v_px']})", (yw, pt),
                             textcoords="offset points", xytext=(8, 8), fontsize=7, color="white",
                             bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))

    ax2.axis("off")
    txt = ["Sector details:", ""]
    txt.append(f"{'Idx':>3s} {'R':>1s} {'C':>1s} {'Pitch range':>18s} {'Yaw range':>18s} {'Px':>5s}")
    txt.append("-" * 60)
    for i, s in enumerate(meta["sectors"]):
        txt.append(f"{i:3d} {s['row']:1d} {s['col']:1d} "
                   f"[{s['pitch_deg'][0]:+6.1f},{s['pitch_deg'][1]:+6.1f}] "
                   f"[{s['yaw_deg'][0]:+6.1f},{s['yaw_deg'][1]:+6.1f}] {s['pixel_count']:5d}")
    txt += ["", "Pixel → KU projection:"]
    txt.append(f"{'u_px':>5s} {'v_px':>5s} {'yaw°':>7s} {'pitch°':>7s} {'KU_r':>4s} {'KU_c':>4s} {'sector':>7s}")
    txt.append("-" * 48)
    if pixel_proj:
        for px in pixel_proj["pixel_results"]:
            txt.append(f"{px['u_px']:5d} {px['v_px']:5d} "
                       f"{px['yaw_deg']:7.2f} {px['pitch_deg']:7.2f} "
                       f"{px['ku_row']:4d} {px['ku_col']:4d} {px['sector']:>7s}")
    txt += ["", f"FOV pixels: {meta['fov_total_pixels']} ({meta['fov_pct_of_sphere']:.1f}%)"]
    ax2.text(0.02, 0.98, "\n".join(txt), transform=ax2.transAxes,
             fontfamily="monospace", fontsize=7.5, va="top")
    plt.tight_layout(); plt.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close()
    print(f"[OK] Figure → {path}")
    return True


# ============================================================
# Core: dry-run
# ============================================================

def run_dry(args):
    nr, nc = args.layout["rows"], args.layout["cols"]
    dw, dh = args.depth_w, args.depth_h
    fl, ha = args.focal_length, args.h_aperture
    va = ha * dh / max(1, dw) if args.v_aperture <= 0 else args.v_aperture
    fx = dw * fl / ha;  fy = dh * fl / va
    cx = 0.5 * dw;      cy = 0.5 * dh
    hfov = 2 * math.atan(dw / (2 * fx));  vfov = 2 * math.atan(dh / (2 * fy))

    lp_min = math.radians(args.lidar_vfov_min); lp_max = math.radians(args.lidar_vfov_max)
    cp = np.array(args.camera_pos); ct = np.array(args.camera_target)
    ax = ct - cp; axy = float(np.hypot(ax[0], ax[1]))
    if np.linalg.norm(ax) <= 1e-9: print("ERROR cam_pos==cam_target"); sys.exit(1)
    cyaw = math.atan2(float(ax[1]), float(ax[0]))
    cpitch = math.atan2(float(ax[2]), axy)
    cp_min = cpitch - vfov/2; cp_max = cpitch + vfov/2
    fp_min = max(cp_min, lp_min); fp_max = min(cp_max, lp_max)
    if fp_max <= fp_min:
        print(f"ERROR no overlap! cam=[{math.degrees(cp_min):.1f},{math.degrees(cp_max):.1f}]°  "
              f"lidar=[{args.lidar_vfov_min},{args.lidar_vfov_max}]°"); sys.exit(1)

    ku_pmin = -math.pi/2; ku_pmax = math.pi/2
    cam_rot = camera_rotation_from_view(cp, ct)  # row-vector

    print("\n" + "=" * 70)
    print("Camera‑Sector‑LiDAR Alignment (dry-run)")
    print("=" * 70)
    print(f"  Cam pos={args.camera_pos}  target={args.camera_target}  yaw={math.degrees(cyaw):.2f}° pitch={math.degrees(cpitch):.2f}°")
    print(f"  Cam rotation (row-vector):")
    for i, name in enumerate(["right", "up", "-fwd"]):
        print(f"    {name}: [{cam_rot[i,0]:.4f}, {cam_rot[i,1]:.4f}, {cam_rot[i,2]:.4f}]")
    print(f"  Depth {dw}×{dh}  focal={fl}mm  fx={fx:.2f} fy={fy:.2f}")
    print(f"  FoV h={math.degrees(hfov):.2f}° v={math.degrees(vfov):.2f}°")
    print(f"  Cam vFoV [{math.degrees(cp_min):.1f}°, {math.degrees(cp_max):.1f}°]")
    print(f"  LiDAR pitch [{args.lidar_vfov_min}°, {args.lidar_vfov_max}°]")
    print(f"  Overlap [{math.degrees(fp_min):.1f}°, {math.degrees(fp_max):.1f}°]")
    print(f"  KU pitch map [{math.degrees(ku_pmin):.0f}°, {math.degrees(ku_pmax):.0f}°]")
    print(f"  Sectors {nr}×{nc}={nr*nc}")

    masks, meta = build_sector_masks(ku_h=40, ku_w=80, camera_h_fov_rad=hfov,
                                      fov_pitch_min=fp_min, fov_pitch_max=fp_max,
                                      num_rows=nr, num_cols=nc,
                                      camera_yaw_center_rad=cyaw,
                                      lidar_pitch_min=ku_pmin, lidar_pitch_max=ku_pmax)

    print(f"\n  Sectors:")
    print(f"  {'Idx':>3s} {'R':>2s} {'C':>2s} {'Pitch range':>18s} {'Yaw range':>18s} {'Px':>6s}")
    print(f"  {'-'*60}")
    for i, s in enumerate(meta["sectors"]):
        print(f"  {i:3d} {s['row']:2d} {s['col']:2d} "
              f"[{s['pitch_deg'][0]:+6.1f},{s['pitch_deg'][1]:+6.1f}]  "
              f"[{s['yaw_deg'][0]:+6.1f},{s['yaw_deg'][1]:+6.1f}]  {s['pixel_count']:6d}")

    print("\n" + render_ascii(masks, meta))

    pproj = project_pixels_to_ku(fx=fx, fy=fy, cx=cx, cy=cy, camera_h_fov_rad=hfov,
                                  fov_pitch_min=fp_min, fov_pitch_max=fp_max,
                                  ku_h=40, ku_w=80, lidar_pitch_min=ku_pmin, lidar_pitch_max=ku_pmax,
                                  camera_yaw_center_rad=cyaw,
                                  num_test_rows=nr, num_test_cols=nc, depth_w=dw, depth_h=dh,
                                  cam_rot_body=cam_rot)
    print(f"\n  Pixel → KU projection:")
    print(f"  {'u_px':>5s} {'v_px':>5s} {'yaw°':>8s} {'pitch°':>8s} {'KU_r':>5s} {'KU_c':>5s} {'sector':>8s}")
    print(f"  {'-'*55}")
    for px in pproj["pixel_results"]:
        print(f"  {px['u_px']:5d} {px['v_px']:5d} {px['yaw_deg']:8.2f} {px['pitch_deg']:8.2f} "
              f"{px['ku_row']:5d} {px['ku_col']:5d} {px['sector']:>8s}")

    outd = Path(args.output_dir); outd.mkdir(parents=True, exist_ok=True)
    rep = {"config": {"depth_w": dw, "depth_h": dh, "focal_mm": fl, "h_aperture_mm": ha,
                       "v_aperture_mm": va, "fx_px": fx, "fy_px": fy, "cx_px": cx, "cy_px": cy,
                       "camera_h_fov_deg": math.degrees(hfov), "camera_v_fov_deg": math.degrees(vfov),
                       "camera_pos": args.camera_pos, "camera_target": args.camera_target,
                       "camera_pitch_deg": math.degrees(cpitch),
                       "lidar_vfov": [args.lidar_vfov_min, args.lidar_vfov_max],
                       "effective_overlap_deg": [math.degrees(fp_min), math.degrees(fp_max)]},
           "sector_meta": {k: v for k, v in meta.items() if k not in ("yaw_grid","pitch_grid","col_yaws_deg","row_pitches_deg")},
           "pixel_projection": pproj}
    rp = outd / "alignment_report.json"
    with open(rp, "w") as f: json.dump(rep, f, indent=2, default=str)
    print(f"\n[OK] JSON → {rp}")
    save_fig(masks, meta, pproj, str(outd / "sector_lidar_alignment.png"))

    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    ok1 = fp_max > fp_min
    print(f"  [1] Pitch overlap: {'✓' if ok1 else '✗ FAIL'}")
    ok2 = all(s["pixel_count"] > 0 for s in meta["sectors"])
    print(f"  [2] All sectors nonempty: {'✓' if ok2 else '✗'}")
    pc = True
    for p in pproj["pixel_results"]:
        if p["sector"] == "outside":
            print(f"       ✗ pixel ({p['u_px']},{p['v_px']}) → {p['yaw_deg']:.2f}°/{p['pitch_deg']:.2f}° → outside!")
            pc = False
    print(f"  [3] Test pixels in FOV: {'✓' if pc else '✗ FAIL'}")
    print(f"  Output dir: {outd}")


# ============================================================
# Core: live Isaac Sim
# ============================================================

def compose_live_cfg(cfg_ov):
    """Compose the real OmniDrones Hydra config used by forest_lc_gate."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    nr, nc = cfg_ov["nr"], cfg_ov["nc"]
    task_name = cfg_ov.get("task_name", "forest_lc_gate")
    active_gpu = int(cfg_ov.get("active_gpu", 1))
    num_envs = int(cfg_ov.get("num_envs", 1))
    overrides = [
        f"task={task_name}",
        "headless=true",
        f"task.env.num_envs={num_envs}",
        "task.env.max_episode_length=100",
        f"task.camera_risk_num_rows={nr}",
        f"task.camera_risk_num_cols={nc}",
        "task.use_camera_risk_observation=true",
        "task.show_depth_preview_window=false",
        "task.depth_debug_checks=false",
        "task.save_depth_debug_outputs=false",
        "task.sim.enable_viewport=false",
        "task.sim.enable_replicator=true",
        f"task.sim.active_gpu={active_gpu}",
        "task.sim.physics_gpu=0",
        "task.sim.device=cuda:0",
    ]

    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(config_name="train", overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


def run_live(cfg_ov):
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print("[WARN] Clearing CUDA_VISIBLE_DEVICES"); del os.environ["CUDA_VISIBLE_DEVICES"]

    # NOTE: All Isaac Sim / Omniverse imports MUST happen AFTER init_simulation_app()
    nr, nc = cfg_ov["nr"], cfg_ov["nc"]
    cfg = compose_live_cfg(cfg_ov)

    sim_app = None
    try:
        # ---- Delayed imports: must be after SimulationApp is alive ----
        from omni_drones import init_simulation_app
        sim_app = init_simulation_app(cfg)

        from omni_drones.envs.isaac_env import IsaacEnv
        import importlib
        try: importlib.import_module(f"omni_drones.envs.single.{cfg.task.name.lower()}")
        except ModuleNotFoundError: pass
        benv = IsaacEnv.REGISTRY[cfg.task.name](cfg, headless=True)

        sys.path.insert(0, str(ZK_DIR))
        from train_canlidargate_trees import _read_depth_camera_geometry_from_stage
        cg = _read_depth_camera_geometry_from_stage(benv)

        print("\n" + "=" * 70)
        print("Camera Geometry from USD")
        print("=" * 70)
        for k, v in cg.items():
            if k == "intrinsic_matrix":
                print(f"  {k}: {v[0]} / {v[1]} / {v[2]}")
            elif isinstance(v, (list, tuple)) and len(v) <= 6:
                print(f"  {k}: {[round(x,4) if isinstance(x,float) else x for x in v]}")
            else:
                print(f"  {k}: {v}")

        fx = cg["intrinsic_matrix"][0][0]; fy = cg["intrinsic_matrix"][1][1]
        cx = cg["intrinsic_matrix"][0][2]; cy = cg["intrinsic_matrix"][1][2]
        depth_res = cfg.task.get("depth_resolution", [_DEFAULT_DEPTH_H, _DEFAULT_DEPTH_W])
        dh, dw = int(depth_res[0]), int(depth_res[1])
        hfov = 2 * math.atan(dw / (2 * fx)); vfov = 2 * math.atan(dh / (2 * fy))

        lidar_vfov = cfg.task.get("lidar_vfov", [_DEFAULT_LIDAR_VFOV_MIN, _DEFAULT_LIDAR_VFOV_MAX])
        lp_min = math.radians(float(lidar_vfov[0])); lp_max = math.radians(float(lidar_vfov[1]))
        ku_pmin = -math.pi/2; ku_pmax = math.pi/2
        cp = np.array(cfg.task.get("depth_camera_pos", _DEFAULT_CAM_POS), dtype=np.float64)
        ct = np.array(cfg.task.get("depth_camera_target", _DEFAULT_CAM_TARGET), dtype=np.float64)
        ax = ct - cp
        cyaw = math.atan2(float(ax[1]), float(ax[0]))
        cpitch = math.atan2(float(ax[2]), float(np.hypot(ax[0], ax[1])))
        cp_min = cpitch - vfov/2; cp_max = cpitch + vfov/2
        fp_min = max(cp_min, lp_min); fp_max = min(cp_max, lp_max)

        print(f"\n  fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
        print(f"  hFoV={math.degrees(hfov):.2f}° vFoV={math.degrees(vfov):.2f}°")
        print(f"  Cam yaw={math.degrees(cyaw):.2f}° pitch={math.degrees(cpitch):.2f}° overlap=[{math.degrees(fp_min):.1f}°,{math.degrees(fp_max):.1f}°]")

        # Extract camera→body rotation from USD geometry for pixel projection
        cam_rot_live = np.array(cg.get("camera_rot_base", [[1,0,0],[0,1,0],[0,0,1]]),
                                 dtype=np.float64)
        # _read_depth_camera_geometry_from_stage returns row-vector convention,
        # same as what project_pixels_to_ku expects.

        masks, meta = build_sector_masks(ku_h=40, ku_w=80, camera_h_fov_rad=hfov,
                                          fov_pitch_min=fp_min, fov_pitch_max=fp_max,
                                          num_rows=nr, num_cols=nc,
                                          camera_yaw_center_rad=cyaw,
                                          lidar_pitch_min=ku_pmin, lidar_pitch_max=ku_pmax)
        print("\n" + render_ascii(masks, meta))

        pproj = project_pixels_to_ku(fx=fx, fy=fy, cx=cx, cy=cy, camera_h_fov_rad=hfov,
                                      fov_pitch_min=fp_min, fov_pitch_max=fp_max,
                                      ku_h=40, ku_w=80, lidar_pitch_min=ku_pmin, lidar_pitch_max=ku_pmax,
                                      camera_yaw_center_rad=cyaw,
                                      num_test_rows=nr, num_test_cols=nc, depth_w=dw, depth_h=dh,
                                      cam_rot_body=cam_rot_live)
        print(f"\n  Pixel → KU:")
        print(f"  {'u_px':>5s} {'v_px':>5s} {'yaw°':>8s} {'pitch°':>8s} {'KU_r':>5s} {'KU_c':>5s} {'sector':>8s}")
        for px in pproj["pixel_results"]:
            print(f"  {px['u_px']:5d} {px['v_px']:5d} {px['yaw_deg']:8.2f} {px['pitch_deg']:8.2f} "
                  f"{px['ku_row']:5d} {px['ku_col']:5d} {px['sector']:>8s}")

        outd = Path(cfg_ov.get("output_dir", SCRIPT_DIR / "alignment_check"))
        outd.mkdir(parents=True, exist_ok=True)
        rep = {"camera_geometry": {k: v for k, v in cg.items() if k != "camera_rot_base"},
               "fov": {"h_deg": math.degrees(hfov), "v_deg": math.degrees(vfov),
                       "overlap_deg": [math.degrees(fp_min), math.degrees(fp_max)]},
               "sector_meta": {k: v for k, v in meta.items() if k not in ("yaw_grid","pitch_grid","col_yaws_deg","row_pitches_deg")},
               "pixel_projection": pproj}
        rp = outd / "alignment_report.json"
        with open(rp, "w") as f: json.dump(rep, f, indent=2, default=str)
        print(f"\n[OK] JSON → {rp}")
        save_fig(masks, meta, pproj, str(outd / "sector_lidar_alignment.png"))

        re = cg.get("row_err_deg", 999); ce = cg.get("col_err_deg", 999)
        be = min(re, ce)
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"  Rotation convention: {cg.get('rotation_convention','N/A')}")
        print(f"  Rotation error: row={re:.3f}° col={ce:.3f}° best={be:.3f}° {'✓' if be<=5 else '⚠ >5°'}")
        print(f"  Output dir: {outd}")
    finally:
        if sim_app is not None: sim_app.close()


# ============================================================
# CLI
# ============================================================

def main():
    p = argparse.ArgumentParser(description="Camera-sector-LiDAR alignment (v2)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--camera-risk-layout", default="3x3")
    p.add_argument("--output-dir", default=str(SCRIPT_DIR / "alignment_check"))
    p.add_argument("--depth-w", type=int, default=_DEFAULT_DEPTH_W)
    p.add_argument("--depth-h", type=int, default=_DEFAULT_DEPTH_H)
    p.add_argument("--focal-length", type=float, default=_DEFAULT_FOCAL_MM)
    p.add_argument("--h-aperture", type=float, default=_DEFAULT_H_APERTURE_MM)
    p.add_argument("--v-aperture", type=float, default=0.0)
    p.add_argument("--camera-pos", type=float, nargs=3, default=_DEFAULT_CAM_POS)
    p.add_argument("--camera-target", type=float, nargs=3, default=_DEFAULT_CAM_TARGET)
    p.add_argument("--lidar-vfov-min", type=float, default=_DEFAULT_LIDAR_VFOV_MIN)
    p.add_argument("--lidar-vfov-max", type=float, default=_DEFAULT_LIDAR_VFOV_MAX)
    p.add_argument("--active-gpu", type=int, default=1)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--task-name", default="forest_lc_gate")
    a = p.parse_args()

    ly = a.camera_risk_layout.strip().lower()
    if "x" in ly:
        nr, nc = int(ly.split("x")[0]), int(ly.split("x")[1])
    elif ly.isdigit():
        nr, nc = 1, int(ly)
    else:
        p.error(f"Bad layout: {a.camera_risk_layout}")
    a.layout = {"rows": nr, "cols": nc}

    if a.dry_run:
        run_dry(a)
    else:
        run_live({"nr": nr, "nc": nc, "active_gpu": a.active_gpu,
                  "num_envs": a.num_envs,
                  "task_name": a.task_name, "output_dir": a.output_dir})

if __name__ == "__main__":
    main()
