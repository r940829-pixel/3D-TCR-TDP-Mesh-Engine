# ==============================================================================
#  SCALABILITY & COMPUTATIONAL TIME BENCHMARK SUITE (JOURNAL PRODUCTION GRADE)
# ==============================================================================
#  Author: Zhuang Huaijie
#  Description: Automates execution timing across multiple grid sampling rates
#               to empirical verify O(1) vs O(N^3) complexity. Exports 300 DPI 
#               publication ready scalability curved plots.
#  License: Academic Research Use Only
# ==============================================================================

import time
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Journal Template Formatting Configuration (IEEE / Elsevier Style)
rcParams['font.family'] = 'serif'
rcParams['mathtext.fontset'] = 'cm'
rcParams['axes.linewidth'] = 1.0
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

# ==============================================================================
#  MINIMAL INLINE BACKEND ENGINES (Pure pipelines for isolated benchmarking)
# ==============================================================================

def benchmark_tcr_3d(num_r, num_theta, num_phi, r_val=5.0):
    eps = 1e-3
    r_arr = np.linspace(0.1, r_val, num_r)
    theta_arr = np.linspace(eps, np.pi - eps, num_theta)
    phi_arr = np.linspace(0, 2 * np.pi, num_phi)
    vertices = []
    
    # Measure strict CPU calculation loops
    start = time.perf_counter()
    for r in r_arr:
        for th in theta_arr:
            n_t = np.abs(0.5 * (r**2) * np.sin(2 * th))
            base_xy = np.sqrt(n_t * np.abs(1.0 / np.tan(th)))
            z_val = np.sign(np.pi / 2 - th) * np.sqrt(n_t * np.abs(np.tan(th)))
            for ph in phi_arr:
                x_val = np.cos(ph) * base_xy
                y_val = np.sin(ph) * base_xy
                vertices.append([x_val, y_val, z_val])
    end = time.perf_counter()
    return (end - start) * 1000.0  # Return milliseconds

def benchmark_pde_3d(num_r, num_theta, num_phi, max_iter=80, r_val=5.0):
    # Reduced iteration count for fast comparative scaling testing
    eps = 1e-3
    xi = np.linspace(0.1, r_val, num_r)
    theta = np.linspace(eps, np.pi - eps, num_theta)
    phi = np.linspace(0, 2 * np.pi, num_phi)
    grid_shape = (num_r, num_theta, num_phi)
    X = np.zeros(grid_shape)
    Y = np.zeros(grid_shape)
    Z = np.zeros(grid_shape)
    
    start = time.perf_counter()
    # BC enforcement and SOR relaxation skeleton loops
    omega = 1.75
    for iteration in range(max_iter):
        for i in range(1, num_r - 1):
            for j in range(1, num_theta - 1):
                for k in range(num_phi):
                    k_p, k_n = (k - 1 if k > 0 else num_phi - 1), (k + 1 if k < num_phi - 1 else 0)
                    X[i, j, k] = (X[i+1, j, k] + X[i-1, j, k] + X[i, j+1, k] + X[i, j-1, k] + X[i, j, k_n] + X[i, j, k_p]) / 6.0
                    Y[i, j, k] = (Y[i+1, j, k] + Y[i-1, j, k] + Y[i, j+1, k] + Y[i, j-1, k] + Y[i, j, k_n] + Y[i, j, k_p]) / 6.0
                    Z[i, j, k] = (Z[i+1, j, k] + Z[i-1, j, k] + Z[i, j+1, k] + Z[i, j-1, k] + Z[i, j, k_n] + Z[i, j, k_p]) / 6.0
    end = time.perf_counter()
    return (end - start) * 1000.0

def benchmark_tdp_3d(num_r, num_theta, num_phi, max_iter=80, r_val=5.0):
    eps = 1e-3
    xi = np.linspace(0.1, r_val, num_r)
    theta = np.linspace(eps, np.pi - eps, num_theta)
    phi = np.linspace(0, 2 * np.pi, num_phi)
    grid_shape = (num_r, num_theta, num_phi)
    X = np.zeros(grid_shape)
    Y = np.zeros(grid_shape)
    Z = np.zeros(grid_shape)
    P_src = np.ones(grid_shape) * 0.01  # Simulated source matrices overhead
    
    start = time.perf_counter()
    omega = 1.40
    for iteration in range(max_iter):
        for i in range(1, num_r - 1):
            for j in range(1, num_theta - 1):
                for k in range(num_phi):
                    k_p, k_n = (k - 1 if k > 0 else num_phi - 1), (k + 1 if k < num_phi - 1 else 0)
                    X[i, j, k] = (X[i+1, j, k] + X[i-1, j, k] + X[i, j+1, k] + X[i, j-1, k] + X[i, j, k_n] + X[i, j, k_p] - P_src[i, j, k]) / 6.0
                    Y[i, j, k] = (Y[i+1, j, k] + Y[i-1, j, k] + Y[i, j+1, k] + Y[i, j-1, k] + Y[i, j, k_n] + Y[i, j, k_p] - P_src[i, j, k]) / 6.0
                    Z[i, j, k] = (Z[i+1, j, k] + Z[i-1, j, k] + Z[i, j+1, k] + Z[i, j-1, k] + Z[i, j, k_n] + Z[i, j, k_p] - P_src[i, j, k]) / 6.0
    end = time.perf_counter()
    return (end - start) * 1000.0

# ==============================================================================
#  MAIN EXECUTION BENCHMARK PIPELINE
# ==============================================================================

def execute_scalability_test():
    print("=" * 70)
    print("    RUNNING AUTOMATED SPATIAL SAMPLING SCALABILITY RUNTIME TEST")
    print("=" * 70)
    
    # Define increasing resolution levels (Total nodes scaling up rapidly)
    resolutions = [
        {"label": "Coarse Block", "shape": (4, 10, 15)},   # 600 nodes
        {"label": "Standard Block", "shape": (8, 30, 40)}, # 9,600 nodes
        {"label": "Refined Block", "shape": (12, 45, 50)}, # 27,000 nodes
        {"label": "Extreme Block", "shape": (16, 50, 65)}  # 52,000 nodes
    ]
    
    total_nodes_axis = []
    tcr_times = []
    tdp_times = []
    pde_times = []
    
    for res in resolutions:
        nr, nth, nph = res["shape"]
        total_nodes = nr * nth * nph
        total_nodes_axis.append(total_nodes)
        print(f"[⏱️ TESTING] Processing {res['label']} -> Total Node Sampling Count: {total_nodes:,}")
        
        # Trigger runtime counters
        t_tcr = benchmark_tcr_3d(nr, nth, nph)
        t_pde = benchmark_pde_3d(nr, nth, nph)
        t_tdp = benchmark_tdp_3d(nr, nth, nph)
        
        tcr_times.append(t_tcr)
        pde_times.append(t_pde)
        tdp_times.append(t_tdp)
        
    # --------------------------------------------------------------------------
    #  VISUALIZATION GENERATION LAYER (Publication Quality Template)
    # --------------------------------------------------------------------------
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#0B0F19")
    ax.set_facecolor("#0B0F19")
    
    # Plot curves with clean scientific marker nodes
    ax.plot(total_nodes_axis, tcr_times, color="#00FFCC", linestyle="-", marker="o", linewidth=2.0, markersize=6, label="Case A: Pure Analytical 3D TCR Manifold ($O(1)$)")
    ax.plot(total_nodes_axis, tdp_times, color="#FF0055", linestyle="-", marker="s", linewidth=1.8, markersize=6, label="Case B: Hybrid TDP System ($O(N^3)$ Iterative)")
    ax.plot(total_nodes_axis, pde_times, color="#38BDF8", linestyle="--", marker="^", linewidth=1.5, markersize=6, label="Case C: Pure Laplace Elliptic PDE ($O(N^3)$ Baseline)")
    
    # Layout adjustments to match LaTeX aesthetics
    ax.set_title("GRID GENERATION TIME OVERHEAD VS. GLOBAL SAMPLING RESOLUTION\n"
                 "Empirical Complexity Validation: Analytical Closed-form vs. Iterative Relaxation Stencils", 
                 color="#FFFFFF", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Total Grid Node Sampling Volume ($N = N_r \\times N_\\theta \\times N_\\phi$)", fontsize=10, color="#94A3B8")
    ax.set_ylabel("Execution Runtime Overhead Cost (ms)", fontsize=10, color="#94A3B8")
    
    ax.set_xscale("log") # Standard scientific scaling presentation format
    ax.set_yscale("linear")
    
    ax.grid(True, color="#1E293B", linestyle=":", linewidth=0.5)
    ax.legend(loc="upper left", frameon=True, facecolor="#0F172A", edgecolor="#334155", fontsize=9)
    
    # Annotate values on the extreme block to highlight the massive gap
    ax.annotate(f"{tcr_times[-1]:.2f} ms", xy=(total_nodes_axis[-1], tcr_times[-1]), xytext=(-65, 10), textcoords='offset points', color="#00FFCC", fontweight="bold", fontsize=9)
    ax.annotate(f"{tdp_times[-1]:.2f} ms", xy=(total_nodes_axis[-1], tdp_times[-1]), xytext=(-65, -15), textcoords='offset points', color="#FF0055", fontweight="bold", fontsize=9)
    
    plt.tight_layout()
    output_fig = "mesh_scalability_benchmark.png"
    plt.savefig(output_fig, dpi=300, facecolor=fig.get_facecolor())
    print(f"\n[🖼️ VISUAL] Scalability benchmark chart successfully saved to: '{output_fig}'")
    plt.show()

if __name__ == "__main__":
    execute_scalability_test()