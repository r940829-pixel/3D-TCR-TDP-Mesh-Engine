# ==============================================================================
#  PURE PDE MESH GENERATION ENGINE FOR COMPUTATIONAL PHYSICS (JOURNAL PRODUCTION)
# ==============================================================================
#  Author: Zhuang Huaijie
#  Description: Three-Dimensional Pure Elliptic PDE (Laplace) Mesh Generator.
#               Serves as the baseline control group (Ablation A) resolving 
#               homogeneous boundary value problems via Successive Over-Relaxation.
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

def generate_pure_pde_3d_mesh(r_val, num_r, num_theta, num_phi, max_iter=500, tol=1e-5, filename="pde_mesh_journal.json"):
    """
    Generates a 3D Pure Elliptic PDE structured spherical mesh via Successive 
    Over-Relaxation (SOR) and exports the coordinates directly into a JSON file.
    
    Mathematical Foundation:
        Solves Laplace equations (d^2/d_xi^2 + d^2/d_eta^2 + d^2/d_zeta^2) [X, Y, Z] = 0
        satisfying Dirichlet boundary conditions on the target sphere without 
        source-term modulation.
        
    Parameters:
        r_val (float): Maximum radial boundary dimension.
        num_r (int): Radial grid resolution layers.
        num_theta (int): Polar/Latitude angle resolution layers [0, pi].
        num_phi (int): Azimuthal/Longitude angle resolution layers [0, 2*pi].
        max_iter (int): Maximum convergence iterations for the SOR solver.
        tol (float): L-infinity convergence tolerance threshold.
        filename (str): Target path for explicit output file storage.
        
    Returns:
        dict: The serialized structured mesh dictionary containing topology and vertices.
    """
    eps = 1e-3
    xi = np.linspace(0.1, r_val, num_r)
    theta = np.linspace(eps, np.pi - eps, num_theta)
    phi = np.linspace(0, 2 * np.pi, num_phi)
    
    # Grid shape array dimensions
    grid_shape = (num_r, num_theta, num_phi)
    
    # Initialize 3D field arrays for numerical relaxation
    X = np.zeros(grid_shape)
    Y = np.zeros(grid_shape)
    Z = np.zeros(grid_shape)
    
    # 1. Enforce Dirichlet Boundary Conditions (Standard Spherical Boundary)
    for j in range(num_theta):
        for k in range(num_phi):
            # Inner boundary core (r = min)
            r_min = xi[0]
            X[0, j, k] = r_min * np.sin(theta[j]) * np.cos(phi[k])
            Y[0, j, k] = r_min * np.sin(theta[j]) * np.sin(phi[k])
            Z[0, j, k] = r_min * np.cos(theta[j])
            
            # Outer boundary shell (r = max)
            r_max = xi[-1]
            X[-1, j, k] = r_max * np.sin(theta[j]) * np.cos(phi[k])
            Y[-1, j, k] = r_max * np.sin(theta[j]) * np.sin(phi[k])
            Z[-1, j, k] = r_max * np.cos(theta[j])

    # Linear boundary blending for polar axes (j=0 and j=num_theta-1)
    for i in range(1, num_r - 1):
        for k in range(num_phi):
            r_curr = xi[i]
            Z[i, 0, k] = r_curr * np.cos(theta[0])
            Z[i, -1, k] = r_curr * np.cos(theta[-1])

    # 2. Linear Algebraic Initialization (Transfinite Interpolation Guess)
    for i in range(1, num_r - 1):
        factor = (xi[i] - xi[0]) / (xi[-1] - xi[0])
        for j in range(1, num_theta - 1):
            for k in range(num_phi):
                X[i, j, k] = (1.0 - factor) * X[0, j, k] + factor * X[-1, j, k]
                Y[i, j, k] = (1.0 - factor) * Y[0, j, k] + factor * Y[-1, j, k]
                Z[i, j, k] = (1.0 - factor) * Z[0, j, k] + factor * Z[-1, j, k]

    # 3. 3D Successive Over-Relaxation (SOR) Laplace Solver Core
    omega = 1.75  # Optimal acceleration parameter for structured spherical blocks
    
    for iteration in range(max_iter):
        max_diff = 0.0
        
        # Periodic boundary handling for azimuthal angle phi (k-loop wrapping)
        for i in range(1, num_r - 1):
            for j in range(1, num_theta - 1):
                for k in range(num_phi):
                    k_prev = k - 1 if k > 0 else num_phi - 1
                    k_next = k + 1 if k < num_phi - 1 else 0
                    
                    # 7-point 3D Laplacian finite-difference stencil averaging
                    x_new = (X[i+1, j, k] + X[i-1, j, k] + 
                             X[i, j+1, k] + X[i, j-1, k] + 
                             X[i, j, k_next] + X[i, j, k_prev]) / 6.0
                             
                    y_new = (Y[i+1, j, k] + Y[i-1, j, k] + 
                             Y[i, j+1, k] + Y[i, j-1, k] + 
                             Y[i, j, k_next] + Y[i, j, k_prev]) / 6.0
                             
                    z_new = (Z[i+1, j, k] + Z[i-1, j, k] + 
                             Z[i, j+1, k] + Z[i, j-1, k] + 
                             Z[i, j, k_next] + Z[i, j, k_prev]) / 6.0
                    
                    # Apply relaxation updates and record residual field norm
                    diff_x = omega * (x_new - X[i, j, k])
                    diff_y = omega * (y_new - Y[i, j, k])
                    diff_z = omega * (z_new - Z[i, j, k])
                    
                    max_diff = max(max_diff, abs(diff_x), abs(diff_y), abs(diff_z))
                    
                    X[i, j, k] += diff_x
                    Y[i, j, k] += diff_y
                    Z[i, j, k] += diff_z
                    
        if max_diff < tol:
            print(f"[⚙️ PDE SOLVER] Convergence achieved at iteration {iteration}.")
            break

    # 4. Serialize Structured Nodes directly into JSON Dataset
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
        
    print(f"[⚙️ PDE CORE] Baseline mesh successfully written to {filename}")
    return mesh_data

if __name__ == "__main__":
    # Matches exact resolution of tcr_core_engine.py for precise ablation comparison
    generate_pure_pde_3d_mesh(
        r_val=5.0, 
        num_r=80, 
        num_theta=30, 
        num_phi=40, 
        max_iter=600, 
        filename="pde_mesh_journal.json"
    )