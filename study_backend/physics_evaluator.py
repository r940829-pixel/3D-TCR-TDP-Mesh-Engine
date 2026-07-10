# ==============================================================================
#  HIGH-FIDELITY TRACKING PHYSICS SOLVER & PUBLICATION VISUALIZER (COMPLETE)
# ==============================================================================
#  Author: Zhuang Huaijie
#  Description: Dynamic Grid-Interpolated Lorentz Solver. Compares cell-resolved 
#               trajectories against continuous ground truth to reveal drift errors.
#  License: Academic Research Use Only
# ==============================================================================

import json
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['mathtext.fontset'] = 'cm'
rcParams['axes.linewidth'] = 1.0
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

def load_mesh_file(filename):
    """Reads and parses serialized structured mesh JSON data."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[❌ ERROR] Target mesh database '{filename}' not found.")
        return None

def compute_mesh_quality_and_ortho(mesh_data):
    """Evaluates geometric grid orthogonality metric scores with fixed tensor indexing."""
    if mesh_data is None: return 0.0
    vertices = mesh_data["vertices"]
    grid_shape = mesh_data["grid_shape"]
    X, Y, Z = np.zeros(grid_shape), np.zeros(grid_shape), np.zeros(grid_shape)
    for node in vertices:
        i, j, k = node["index"]
        X[i, j, k], Y[i, j, k], Z[i, j, k] = node["pos"]
        
    ortho_deviations = []
    for i in range(grid_shape[0] - 1):
        for j in range(grid_shape[1] - 1):
            for k in range(grid_shape[2]):
                v_r = np.array([X[i+1, j, k] - X[i, j, k], Y[i+1, j, k] - Y[i, j, k], Z[i+1, j, k] - Z[i, j, k]])
                
                v_th = np.array([X[i, j+1, k] - X[i, j, k], Y[i, j+1, k] - Y[i, j, k], Z[i, j+1, k] - Z[i, j, k]])
                
                norm_r, norm_th = np.linalg.norm(v_r), np.linalg.norm(v_th)
                if norm_r > 1e-5 and norm_th > 1e-5:
                    ortho_deviations.append(np.abs(np.dot(v_r, v_th) / (norm_r * norm_th)))
                    
    return max(0.0, 100.0 * (1.0 - np.mean(ortho_deviations))) if ortho_deviations else 0.0

def solve_lorentz_trajectory(mesh_data, is_ground_truth=False, timesteps=1200, dt=4e-3):
    """
    True Numerical Engine: Simulates electron flight path. 
    Evaluates the real discrete grid nodes to execute genuine physical 
    field interpolation, exposing localized numerical dissipation caused by grid dilution.
    """
    q_over_m = -2.5  
    pos = np.array([3.8, 0.0, 0.1])  
    vel = np.array([-0.3, 3.5, 0.02]) 
    
    trajectory = []
    node_positions = np.array([n["pos"] for n in mesh_data["vertices"]]) if mesh_data else None
    
    def get_analytic_E(p):
        r_mag = np.linalg.norm(p)
        if r_mag < 1e-2: r_mag = 1e-2
        return -3.0 * p / (r_mag**2.8)

    def get_interpolated_E(p):
        if is_ground_truth or node_positions is None:
            return get_analytic_E(p)
        
        sq_dists = np.sum((node_positions - p)**2, axis=1)
        nearest_indices = np.argsort(sq_dists)[:4]  
        
        weights = 1.0 / (np.sqrt(sq_dists[nearest_indices]) + 1e-6)
        weights /= np.sum(weights)
        
        E_interp = np.zeros(3)
        for idx, w in zip(nearest_indices, weights):
            E_interp += w * get_analytic_E(node_positions[idx])
            
        return E_interp

    B_field = np.array([0.0, 0.0, 1.5])

    for _ in range(timesteps):
        trajectory.append(pos.copy())
        
        # --- RK4 Step 1 ---
        E1 = get_interpolated_E(pos)
        k1_pos = vel.copy()
        k1_vel = q_over_m * (E1 + np.cross(vel, B_field))
        
        # --- RK4 Step 2 ---
        pos_k2 = pos + 0.5 * dt * k1_pos
        vel_k2 = vel + 0.5 * dt * k1_vel
        E2 = get_interpolated_E(pos_k2)
        k2_pos = vel_k2.copy()
        k2_vel = q_over_m * (E2 + np.cross(vel_k2, B_field))
        
        # --- RK4 Step 3 ---
        pos_k3 = pos + 0.5 * dt * k2_pos
        vel_k3 = vel + 0.5 * dt * k2_vel
        E3 = get_interpolated_E(pos_k3)
        k3_pos = vel_k3.copy()
        k3_vel = q_over_m * (E3 + np.cross(vel_k3, B_field))
        
        # --- RK4 Step 4 ---
        pos_k4 = pos + dt * k3_pos
        vel_k4 = vel + dt * k3_vel
        E4 = get_interpolated_E(pos_k4)
        k4_pos = vel_k4.copy()
        k4_vel = q_over_m * (E4 + np.cross(vel_k4, B_field))
        
        # --- Unified Accumulation Step ---
        pos += (dt / 6.0) * (k1_pos + 2.0 * k2_pos + 2.0 * k3_pos + k4_pos)
        vel += (dt / 6.0) * (k1_vel + 2.0 * k2_vel + 2.0 * k3_vel + k4_vel)
        
    return np.array(trajectory)

def plot_journal_visualization(trajectories, cached_meshes, ground_truth):
    """Generates comparative plots showing genuine grid-induced path drift with microinsets."""
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#0B0F19")
    colors = ["#00FFCC", "#FF0055", "#38BDF8"]
    titles = ["(a) Pure 3D TCR Engine", "(b) Hybrid TDP System", "(c) Pure Laplace PDE"]
    
    for idx, name in enumerate(cached_meshes.keys()):
        ax = axes[idx]
        ax.set_facecolor("#0B0F19")
        mesh_data = cached_meshes[name]
        traj_grid = trajectories[name]
        
        if mesh_data:
            grid_shape = mesh_data["grid_shape"]
            X, Y = np.zeros(grid_shape), np.zeros(grid_shape)
            for n in mesh_data["vertices"]:
                i, j, k = n["index"]
                X[i, j, k], Y[i, j, k] = n["pos"][0], n["pos"][1]
            for i in range(grid_shape[0]): ax.plot(X[i, :, 0], Y[i, :, 0], color="#1E293B", alpha=0.25, linewidth=0.5)
            for j in range(grid_shape[1]): ax.plot(X[:, j, 0], Y[:, j, 0], color="#1E293B", alpha=0.25, linewidth=0.5)
                
        ax.plot(ground_truth[:, 0], ground_truth[:, 1], color="#64748B", linestyle="--", linewidth=1.2, label="Ground Truth")
        ax.plot(traj_grid[:, 0], traj_grid[:, 1], color=colors[idx], linewidth=1.8, label="Grid-Resolved Path")
        ax.scatter(traj_grid[0,0], traj_grid[0,1], color="#FFFFFF", s=25, zorder=5)
        
        ax.set_title(titles[idx], color="#FFFFFF", fontsize=11, fontweight="bold", pad=8)
        ax.axis("equal")
        ax.set_xlim([-4.5, 4.5])
        ax.set_ylim([-4.5, 4.5])
        ax.grid(True, color="#1E293B", linestyle=":", linewidth=0.5)
        
        ax_inset = ax.inset_axes([0.05, 0.05, 0.38, 0.38], facecolor="#0F172A")
        if mesh_data:
            for i in range(grid_shape[0]): ax_inset.plot(X[i, :, 0], Y[i, :, 0], color="#475569", alpha=0.3, linewidth=0.4)
            for j in range(grid_shape[1]): ax_inset.plot(X[:, j, 0], Y[:, j, 0], color="#475569", alpha=0.3, linewidth=0.4)
        ax_inset.plot(ground_truth[:, 0], ground_truth[:, 1], color="#64748B", linestyle="--", linewidth=1.0)
        ax_inset.plot(traj_grid[:, 0], traj_grid[:, 1], color=colors[idx], linewidth=1.5)
        ax_inset.set_xlim([-0.6, 0.6])
        ax_inset.set_ylim([-0.6, 0.6])
        ax_inset.axis("equal")
        ax_inset.get_xaxis().set_visible(False)
        ax_inset.get_yaxis().set_visible(False)
        for s in ax_inset.spines.values(): s.set_edgecolor(colors[idx])
        ax.indicate_inset_zoom(ax_inset, edgecolor="#64748B", alpha=0.2, linewidth=0.5)
        ax.legend(loc="upper right", frameon=True, facecolor="#0B0F19", edgecolor="#334155", fontsize=8)

    plt.suptitle("CONFORMAL PHYSICS-GEOMETRIC TRAJECTORY QUANTIZATION COMPARISON\n"
                 "True Grid-Interpolated Trajectories with Core Singularity Zoom-In Insets", color="#FFFFFF", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig("journal_trajectory_comparison.png", dpi=300, facecolor=fig.get_facecolor())
    print("[🖼️ VISUAL] High-resolution drift comparison plot saved.")
    plt.show()

def run_evaluation_pipeline():
    print("=" * 70)
    print("    RUNNING CORE-FOCUSED INTERPOLATED PHYSICS VALIDATION SUITE")
    print("=" * 70)
    
    cases = [
        {"name": "Case A: Pure Analytical 3D TCR Engine", "mesh_file": "tcr_mesh_journal.json", "log_file": "tcr_final.txt", "complexity": "O(N) - Discrete Node Field Search"},
        {"name": "Case B: Hybrid TDP System", "mesh_file": "tdp_mesh_journal.json", "log_file": "tdp_final.txt", "complexity": "O(N^3) - Iterative Source-Driven"},
        {"name": "Case C: Pure Non-Source Laplace Elliptic PDE", "mesh_file": "pde_mesh_journal.json", "log_file": "pde_final.txt", "complexity": "O(N^3) - Homogeneous Baseline"}
    ]
    
    ground_truth = solve_lorentz_trajectory(mesh_data=None, is_ground_truth=True)
    cached_meshes, trajectories = {}, {}
    
    for case in cases:
        mesh_data = load_mesh_file(case["mesh_file"])
        cached_meshes[case["name"]] = mesh_data
        
        traj_grid = solve_lorentz_trajectory(mesh_data, is_ground_truth=False)
        trajectories[case["name"]] = traj_grid
        
        core_drift_accum, count = 0.0, 0
        for p_t, p_g in zip(traj_grid, ground_truth):
            r_gt = np.linalg.norm(p_g)
            if r_gt < 3.5:  
                core_drift_accum += np.linalg.norm(p_t - p_g)
                count += 1
                
        mean_core_drift = core_drift_accum / max(1, count)
        fit_score = 100.0 * np.exp(-2.5 * mean_core_drift)
        ortho_score = compute_mesh_quality_and_ortho(mesh_data)
        
        report_content = f"""========================================================================
INTERNATIONAL PUBLICATION METRIC REPORT: COMPANION DATA VERIFICATION
========================================================================
Evaluated Framework      : {case['name']}
Associated Cache Database: {case['mesh_file']}
------------------------------------------------------------------------
[METRIC 1: GEOMETRIC MESH QUALITY]
 - Mean Grid Orthogonality Score     : {ortho_score:.4f} / 100.0000 Points
 - Geometric Shear Deformation Level : {100.0 - ortho_score:.4f}%

[METRIC 2: CORE PHYSICAL SINGULARITY ADAPTATION INTEGRITY (DRIFT TEST)]
 - Localized Core Trajectory Drift   : {mean_core_drift:.6f} Absolute Euclidean Error
 - Singularity Layer Resolution Score: {fit_score:.4f} / 100.0000 Points

[METRIC 3: COMPUTATIONAL TIME OVERHEAD & COMPLEXITY]
 - Algorithm Computational Complexity: {case['complexity']}
----------------------------------------------------------------========
"""
        with open(case["log_file"], "w", encoding="utf-8") as f: f.write(report_content)
        print(f"[💾 EXPORTED] Report logged to '{case['log_file']}'")
        
    plot_journal_visualization(trajectories, cached_meshes, ground_truth)

if __name__ == "__main__":
    run_evaluation_pipeline()
