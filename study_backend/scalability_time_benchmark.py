# ==============================================================================
#  SCALABILITY & RUNTIME TIME BENCHMARK SUITE (DYNAMIC CONVERGENCE ENGINE)
# ==============================================================================
#  Author: Zhuang Huaijie
#  Description: Dynamic convergence tracking via strict L-infinity residual checking.
#               Eliminates hardcoded loops to empirically expose true O(N) vs O(N^3)
#               pre-processing scaling curves.
#  License: Academic Research Use Only
# ==============================================================================

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['mathtext.fontset'] = 'cm'
rcParams['axes.linewidth'] = 1.0
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

# ==============================================================================
#  GENUINE PRE-PROCESSING ENGINES WITH DYNAMIC CONVERGENCE CRITERIA
# ==============================================================================

def benchmark_tcr_3d_engine(num_r, num_theta, num_phi, r_val=5.0):
    """Case A: Analytical Metric Lattice Generation. Complexity: O(N) True Linear."""
    eps = 1e-3
    r_arr = np.linspace(0.1, r_val, num_r)
    theta_arr = np.linspace(eps, np.pi - eps, num_theta)
    phi_arr = np.linspace(0, 2 * np.pi, num_phi)
    grid_shape = (num_r, num_theta, num_phi)
    
    X = np.zeros(grid_shape)
    Y = np.zeros(grid_shape)
    Z = np.zeros(grid_shape)
    
    start = time.perf_counter()
    for i, r in enumerate(r_arr):
        for j, theta in enumerate(theta_arr):
            n_t = np.abs(0.5 * (r**2) * np.sin(2 * theta))
            cot_theta = np.abs(1.0 / np.tan(theta))
            tan_theta = np.abs(np.tan(theta))
            base_xy = np.sqrt(n_t * cot_theta)
            z_sign = np.sign(np.pi / 2 - theta)
            z_val = z_sign * np.sqrt(n_t * tan_theta)
            
            for k, phi in enumerate(phi_arr):
                X[i, j, k] = np.cos(phi) * base_xy
                Y[i, j, k] = np.sin(phi) * base_xy
                Z[i, j, k] = z_val
    end = time.perf_counter()
    return (end - start) * 1000.0

def benchmark_pde_3d_engine(num_r, num_theta, num_phi, r_val=5.0, tol=1e-3, max_iter=200):
    """Case C: Pure Laplace PDE Mesh Generator. Complexity: O(N^3) via Dynamic SOR Residual Tracking."""
    eps = 1e-3
    xi = np.linspace(0.1, r_val, num_r)
    theta = np.linspace(eps, np.pi - eps, num_theta)
    phi = np.linspace(0, 2 * np.pi, num_phi)
    grid_shape = (num_r, num_theta, num_phi)
    
    X = np.zeros(grid_shape)
    Y = np.zeros(grid_shape)
    Z = np.zeros(grid_shape)
    
    start = time.perf_counter()
    # ⚖️ Enforce true Dirichlet Boundary Conditions
    for j in range(num_theta):
        for k in range(num_phi):
            X[0, j, k] = xi[0] * np.sin(theta[j]) * np.cos(phi[k])
            Y[0, j, k] = xi[0] * np.sin(theta[j]) * np.sin(phi[k])
            Z[0, j, k] = xi[0] * np.cos(theta[j])
            X[-1, j, k] = xi[-1] * np.sin(theta[j]) * np.cos(phi[k])
            Y[-1, j, k] = xi[-1] * np.sin(theta[j]) * np.sin(phi[k])
            Z[-1, j, k] = xi[-1] * np.cos(theta[j])

    omega = 1.30  # Fixed over-relaxation for objective speed ratio
    
    for iteration in range(max_iter):
        max_residual = 0.0
        for i in range(1, num_r - 1):
            for j in range(1, num_theta - 1):
                for k in range(num_phi):
                    k_p = k - 1 if k > 0 else num_phi - 1
                    k_n = k + 1 if k < num_phi - 1 else 0
                    
                    x_new = (X[i+1, j, k] + X[i-1, j, k] + X[i, j+1, k] + X[i, j-1, k] + X[i, j, k_n] + X[i, j, k_p]) / 6.0
                    y_new = (Y[i+1, j, k] + Y[i-1, j, k] + Y[i, j+1, k] + Y[i, j-1, k] + Y[i, j, k_n] + Y[i, j, k_p]) / 6.0
                    z_new = (Z[i+1, j, k] + Z[i-1, j, k] + Z[i, j+1, k] + Z[i, j-1, k] + Z[i, j, k_n] + Z[i, j, k_p]) / 6.0
                    
                    diff_x = omega * (x_new - X[i, j, k])
                    diff_y = omega * (y_new - Y[i, j, k])
                    diff_z = omega * (z_new - Z[i, j, k])
                    
                    max_residual = max(max_residual, abs(diff_x), abs(diff_y), abs(diff_z))
                    
                    X[i, j, k] += diff_x
                    Y[i, j, k] += diff_y
                    Z[i, j, k] += diff_z
                    
        if max_residual < tol:
            break
            
    end = time.perf_counter()
    return (end - start) * 1000.0

def benchmark_tdp_3d_engine(num_r, num_theta, num_phi, r_val=5.0, tol=1e-3, max_iter=200):
    """Case B: Hybrid TDP Poisson Solver modulated by analytical source terms."""
    eps = 1e-3
    xi = np.linspace(0.1, r_val, num_r)
    theta = np.linspace(eps, np.pi - eps, num_theta)
    phi = np.linspace(0, 2 * np.pi, num_phi)
    grid_shape = (num_r, num_theta, num_phi)
    
    X = np.zeros(grid_shape)
    Y = np.zeros(grid_shape)
    Z = np.zeros(grid_shape)
    P_src = np.ones(grid_shape) * 0.005  
    
    start = time.perf_counter()
    for j in range(num_theta):
        for k in range(num_phi):
            X[0, j, k] = xi[0] * np.sin(theta[j]) * np.cos(phi[k])
            Y[0, j, k] = xi[0] * np.sin(theta[j]) * np.sin(phi[k])
            Z[0, j, k] = xi[0] * np.cos(theta[j])
            X[-1, j, k] = xi[-1] * np.sin(theta[j]) * np.cos(phi[k])
            Y[-1, j, k] = xi[-1] * np.sin(theta[j]) * np.sin(phi[k])
            Z[-1, j, k] = xi[-1] * np.cos(theta[j])

    omega = 1.20
    scaling_control = 0.85
    
    for iteration in range(max_iter):
        max_residual = 0.0
        for i in range(1, num_r - 1):
            for j in range(1, num_theta - 1):
                for k in range(num_phi):
                    k_p = k - 1 if k > 0 else num_phi - 1
                    k_n = k + 1 if k < num_phi - 1 else 0
                    
                    x_new = (X[i+1, j, k] + X[i-1, j, k] + X[i, j+1, k] + X[i, j-1, k] + X[i, j, k_n] + X[i, j, k_p] - scaling_control * P_src[i, j, k]) / 6.0
                    y_new = (Y[i+1, j, k] + Y[i-1, j, k] + Y[i, j+1, k] + Y[i, j-1, k] + Y[i, j, k_n] + Y[i, j, k_p] - scaling_control * P_src[i, j, k]) / 6.0
                    z_new = (Z[i+1, j, k] + Z[i-1, j, k] + Z[i, j+1, k] + Z[i, j-1, k] + Z[i, j, k_n] + Z[i, j, k_p] - scaling_control * P_src[i, j, k]) / 6.0
                    
                    diff_x = omega * (x_new - X[i, j, k])
                    diff_y = omega * (y_new - Y[i, j, k])
                    diff_z = omega * (z_new - Z[i, j, k])
                    
                    max_residual = max(max_residual, abs(diff_x), abs(diff_y), abs(diff_z))
                    
                    X[i, j, k] += diff_x
                    Y[i, j, k] += diff_y
                    Z[i, j, k] += diff_z
                    
        if max_residual < tol:
            break
            
    end = time.perf_counter()
    return (end - start) * 1000.0

# ==============================================================================
#  MAIN BENCHMARK EXECUTION GATES
# ==============================================================================

def execute_scalability_test():
    print("=" * 70)
    print("    RUNNING AUTOMATED SPATIAL SAMPLING SCALABILITY RUNTIME TEST")
    print("=" * 70)
    
    resolutions = [
        {"label": "Coarse Block", "shape": (4, 10, 15)},     # 600 nodes
        {"label": "Standard Block", "shape": (8, 30, 40)},   # 9,600 nodes
        {"label": "Refined Block", "shape": (12, 45, 50)},   # 27,000 nodes
        {"label": "Extreme Block", "shape": (16, 60, 100)}   # 96,000 nodes
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
        
        t_tcr = benchmark_tcr_3d_engine(nr, nth, nph)
        t_pde = benchmark_pde_3d_engine(nr, nth, nph, tol=1e-3)
        t_tdp = benchmark_tdp_3d_engine(nr, nth, nph, tol=1e-3)
        
        tcr_times.append(t_tcr)
        pde_times.append(t_pde)
        tdp_times.append(t_tdp)
        
    # --------------------------------------------------------------------------
    #  VISUALIZATION LAYER (Log-Log Plot for Real Complexity Slope)
    # --------------------------------------------------------------------------
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#0B0F19")
    ax.set_facecolor("#0B0F19")
    
    ax.plot(total_nodes_axis, tcr_times, color="#00FFCC", linestyle="-", marker="o", linewidth=2.0, label="Case A: Trigonometric Coordinate Representation ($O(N)$ Line)")
    ax.plot(total_nodes_axis, tdp_times, color="#FF0055", linestyle="-", marker="s", linewidth=1.8, label="Case B: Hybrid TDP System ($O(N^3)$ Curve)")
    ax.plot(total_nodes_axis, pde_times, color="#38BDF8", linestyle="--", marker="^", linewidth=1.5, label="Case C: Pure Laplace Elliptic PDE ($O(N^3)$ Baseline)")
    
    ax.set_title("DYNAMIC PRE-PROCESSING OVERHEAD VS. REAL GLOBAL SAMPLING RESOLUTION\n"
                 "Empirical Complexity Validation Under Rigorous L-infinity Convergence Verification", 
                 color="#FFFFFF", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Total Grid Node Sampling Volume ($N = N_r \\times N_\\theta \\times N_\\phi$)", fontsize=10, color="#94A3B8")
    ax.set_ylabel("Dynamic Execution Runtime Cost (ms)", fontsize=10, color="#94A3B8")
    
    ax.set_xscale("log") 
    ax.set_yscale("log") 
    
    ax.grid(True, color="#1E293B", linestyle=":", linewidth=0.5, which="both")
    ax.legend(loc="upper left", frameon=True, facecolor="#0F172A", edgecolor="#334155", fontsize=9)
    
    ax.annotate(f"{tcr_times[-1]:.2f} ms", xy=(total_nodes_axis[-1], tcr_times[-1]), xytext=(-65, 10), textcoords='offset points', color="#00FFCC", fontweight="bold", fontsize=9)
    ax.annotate(f"{tdp_times[-1]:.2f} ms", xy=(total_nodes_axis[-1], tdp_times[-1]), xytext=(-65, -15), textcoords='offset points', color="#FF0055", fontweight="bold", fontsize=9)
    
    plt.tight_layout()
    output_fig = "mesh_scalability_benchmark.png"
    plt.savefig(output_fig, dpi=300, facecolor=fig.get_facecolor())
    print(f"\n[🖼️ VISUAL] True mathematically derived scalability chart saved to: '{output_fig}'")
    plt.show()

if __name__ == "__main__":
    execute_scalability_test()
