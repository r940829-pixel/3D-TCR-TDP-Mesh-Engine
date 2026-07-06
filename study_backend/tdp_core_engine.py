# ==============================================================================
#  TDP (FULLY 3D-TCR DRIVEN PDE) MESH GENERATION ENGINE (JOURNAL PRODUCTION GRADE)
# ==============================================================================
#  Author: Zhuang Huaijie
#  Description: Three-Dimensional Hybrid Poisson Mesh Generator Modulated by Full 
#               3D TCR Analytical Manifold Source Terms. Achieves simultaneous 
#               boundary-conformal mapping and 3D polar singularity tracking.
#  License: Academic Research Use Only
# ==============================================================================

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Journal Template Formatting Configuration 
rcParams['font.family'] = 'serif'
rcParams['mathtext.fontset'] = 'cm'
rcParams['axes.linewidth'] = 1.0
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

def generate_tdp_3d_mesh(r_val, num_r, num_theta, num_phi, max_iter=500, tol=1e-5, filename="tdp_mesh_journal.json"):
    """
    Generates a 3D Hybrid Mesh where the 3D Poisson elliptic system is fully modulated
    by the analytical 3D Time-Coherent Resonance (TCR) manifold Laplacian field.
    """
    eps = 1e-3
    xi = np.linspace(0.1, r_val, num_r)
    theta = np.linspace(eps, np.pi - eps, num_theta)
    phi = np.linspace(0, 2 * np.pi, num_phi)
    
    grid_shape = (num_r, num_theta, num_phi)
    
    # --------------------------------------------------------------------------
    #  STEP 1: GENERATE FULLY THREE-DIMENSIONAL ANALYTICAL TCR MANIFOLD DATA
    # --------------------------------------------------------------------------
    X_tcr = np.zeros(grid_shape)
    Y_tcr = np.zeros(grid_shape)
    Z_tcr = np.zeros(grid_shape)
    
    for i, r in enumerate(xi):
        for j, th in enumerate(theta):
            # 3D TCR Modulation Core Form with Absolute Tensor Operators
            n_t = np.abs(0.5 * (r**2) * np.sin(2 * th))
            cot_th = np.abs(1.0 / np.tan(th))
            tan_th = np.abs(np.tan(th))
            
            base_xy = np.sqrt(n_t * cot_th)
            z_sign = np.sign(np.pi / 2 - th)
            z_val = z_sign * np.sqrt(n_t * tan_th)
            
            for k, ph in enumerate(phi):
                X_tcr[i, j, k] = np.cos(ph) * base_xy
                Y_tcr[i, j, k] = np.sin(ph) * base_xy
                Z_tcr[i, j, k] = z_val

    # --------------------------------------------------------------------------
    #  STEP 2: EXTRACT 3D LAPLACIAN FIELD AS POISSON CONTROL FUNCTIONS
    # --------------------------------------------------------------------------
    P_src = np.zeros(grid_shape)
    Q_src = np.zeros(grid_shape)
    R_src = np.zeros(grid_shape)
    
    # 7-point central difference stencil across all 3 spatial parametric dimensions
    for i in range(1, num_r - 1):
        for j in range(1, num_theta - 1):
            for k in range(num_phi):
                k_p = k - 1 if k > 0 else num_phi - 1
                k_n = k + 1 if k < num_phi - 1 else 0
                
                # Capture the complete 3D tensor curvature forces from analytical map
                P_src[i, j, k] = (X_tcr[i+1, j, k] - 2*X_tcr[i, j, k] + X_tcr[i-1, j, k] +
                                   X_tcr[i, j+1, k] - 2*X_tcr[i, j, k] + X_tcr[i, j-1, k] +
                                   X_tcr[i, j, k_n] - 2*X_tcr[i, j, k] + X_tcr[i, j, k_p]) / 6.0
                                   
                Q_src[i, j, k] = (Y_tcr[i+1, j, k] - 2*Y_tcr[i, j, k] + Y_tcr[i-1, j, k] +
                                   Y_tcr[i, j+1, k] - 2*Y_tcr[i, j, k] + Y_tcr[i, j-1, k] +
                                   Y_tcr[i, j, k_n] - 2*Y_tcr[i, j, k] + Y_tcr[i, j, k_p]) / 6.0
                                   
                R_src[i, j, k] = (Z_tcr[i+1, j, k] - 2*Z_tcr[i, j, k] + Z_tcr[i-1, j, k] +
                                   Z_tcr[i, j+1, k] - 2*Z_tcr[i, j, k] + Z_tcr[i, j-1, k] +
                                   Z_tcr[i, j, k_n] - 2*Z_tcr[i, j, k] + Z_tcr[i, j, k_p]) / 6.0

    # --------------------------------------------------------------------------
    #  STEP 3: PDE SOLVER INITIALIZATION & Dirichlet BOUNDARY CONDITIONS
    # --------------------------------------------------------------------------
    X = np.zeros(grid_shape)
    Y = np.zeros(grid_shape)
    Z = np.zeros(grid_shape)
    
    # Enforce standard spherical Dirichlet boundary shell conditions
    for j in range(num_theta):
        for k in range(num_phi):
            X[0, j, k] = xi[0] * np.sin(theta[j]) * np.cos(phi[k])
            Y[0, j, k] = xi[0] * np.sin(theta[j]) * np.sin(phi[k])
            Z[0, j, k] = xi[0] * np.cos(theta[j])
            
            X[-1, j, k] = xi[-1] * np.sin(theta[j]) * np.cos(phi[k])
            Y[-1, j, k] = xi[-1] * np.sin(theta[j]) * np.sin(phi[k])
            Z[-1, j, k] = xi[-1] * np.cos(theta[j])

    # Transfinite Interpolation (TFI) Algebraic Linear Initial Guess
    for i in range(1, num_r - 1):
        factor = (xi[i] - xi[0]) / (xi[-1] - xi[0])
        for j in range(num_theta):
            for k in range(num_phi):
                X[i, j, k] = (1.0 - factor) * X[0, j, k] + factor * X[-1, j, k]
                Y[i, j, k] = (1.0 - factor) * Y[0, j, k] + factor * Y[-1, j, k]
                Z[i, j, k] = (1.0 - factor) * Z[0, j, k] + factor * Z[-1, j, k]

    # --------------------------------------------------------------------------
    #  STEP 4: 3D SUCCESSIVE OVER-RELAXATION (SOR) POISSON SOLVER CORE
    # --------------------------------------------------------------------------
    omega = 1.40  # Relaxation coefficient optimized for 3D source-driven systems
    scaling_control = 0.85  # Modulation amplitude coefficient for global grid stability
    
    for iteration in range(max_iter):
        max_diff = 0.0
        for i in range(1, num_r - 1):
            for j in range(1, num_theta - 1):
                for k in range(num_phi):
                    k_p = k - 1 if k > 0 else num_phi - 1
                    k_n = k + 1 if k < num_phi - 1 else 0
                    
                    # Complete 3D relaxation stencil updates embedded with full 3D TCR sources
                    x_new = (X[i+1, j, k] + X[i-1, j, k] + X[i, j+1, k] + X[i, j-1, k] + X[i, j, k_n] + X[i, j, k_p] - scaling_control * P_src[i, j, k]) / 6.0
                    y_new = (Y[i+1, j, k] + Y[i-1, j, k] + Y[i, j+1, k] + Y[i, j-1, k] + Y[i, j, k_n] + Y[i, j, k_p] - scaling_control * Q_src[i, j, k]) / 6.0
                    z_new = (Z[i+1, j, k] + Z[i-1, j, k] + Z[i, j+1, k] + Z[i, j-1, k] + Z[i, j, k_n] + Z[i, j, k_p] - scaling_control * R_src[i, j, k]) / 6.0
                    
                    diff_x = omega * (x_new - X[i, j, k])
                    diff_y = omega * (y_new - Y[i, j, k])
                    diff_z = omega * (z_new - Z[i, j, k])
                    
                    max_diff = max(max_diff, abs(diff_x), abs(diff_y), abs(diff_z))
                    
                    X[i, j, k] += diff_x
                    Y[i, j, k] += diff_y
                    Z[i, j, k] += diff_z
                    
        if max_diff < tol:
            print(f"[⚙️ TDP-3D SOLVER] Fully 3D system convergence achieved at iteration {iteration}.")
            break

    # --------------------------------------------------------------------------
    #  STEP 5: DIRECT DATA SERIALIZATION & JSON EXPORT
    # --------------------------------------------------------------------------
    vertices = []
    for i in range(num_r):
        for j in range(num_theta):
            for k in range(num_phi):
                vertices.append({
                    "index": [i, j, k],
                    "pos": [float(X[i, j, k]), float(Y[i, j, k]), float(Z[i, j, k])]
                })
                
    mesh_data = {"grid_shape": grid_shape, "vertices": vertices}
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(mesh_data, f, indent=4)
        
    print(f"[⚙️ TDP-3D CORE] Complete 3D-driven hybrid grid successfully written to {filename}")
    return mesh_data


if __name__ == "__main__":
    # Publication High-Resolution Case Specifications matching cross-pipeline baselines
    generate_tdp_3d_mesh(
        r_val=5.0, 
        num_r=80, 
        num_theta=30, 
        num_phi=40, 
        max_iter=600, 
        filename="tdp_mesh_journal.json"
    )