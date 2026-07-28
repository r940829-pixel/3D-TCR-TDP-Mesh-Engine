import json
import time
import warnings
import numpy as np
import sympy as sp

def generate_tdp_3d_mesh(system_input, tdp_config):
    
    var_names = tdp_config.get("var_names", ('x', 'y', 'z'))
    domain_bounds = tdp_config.get("domain_bounds", (-10.0, 10.0))
    num_r = tdp_config["num_r"]
    num_theta = tdp_config["num_theta"]
    num_phi = tdp_config["num_phi"]
    r_max = tdp_config.get("r_val", 5.0)
    max_iter = tdp_config.get("max_iter", 600)
    tol = tdp_config.get("tol", 1e-5)
    omega = tdp_config.get("omega", 1.75)
    alpha = tdp_config.get("scaling_control", 0.05)  
    out_file = tdp_config.get("output_filename", "tdp_mesh.json")

    print(f"[⚙️ TDP-3D CORE] Step 1: Synthesizing Monge Surface Area Sources...")

    
    vars_sym = [sp.Symbol(name, real=True) for name in var_names]
    if isinstance(system_input, (list, tuple)):
        eq_syms = [sp.sympify(eq_str, locals={'pi': np.pi}) for eq_str in system_input]
        f_scalar_sym = sum(eq**2 for eq in eq_syms)
    else:
        f_scalar_sym = sp.sympify(system_input, locals={'pi': np.pi})

    grad_syms = [sp.diff(f_scalar_sym, v) for v in vars_sym]
    grad_norm_sq_sym = sum(g**2 for g in grad_syms)
    laplacian_sym = sum(sp.diff(f_scalar_sym, v, 2) for v in vars_sym)

    grad_norm_eval = sp.lambdify(vars_sym, sp.sqrt(grad_norm_sq_sym), modules=['numpy'])
    laplacian_eval = sp.lambdify(vars_sym, laplacian_sym, modules=['numpy'])

    
    r_arr = np.linspace(0.1, r_max, num_r)
    theta_arr = np.linspace(1e-3, np.pi - 1e-3, num_theta)
    phi_arr = np.linspace(0, 2 * np.pi, num_phi, endpoint=False)

    
    X = np.zeros((num_r, num_theta, num_phi), dtype=np.float64)
    Y = np.zeros((num_r, num_theta, num_phi), dtype=np.float64)
    Z = np.zeros((num_r, num_theta, num_phi), dtype=np.float64)

    
    for i, r in enumerate(r_arr):
        for j, th in enumerate(theta_arr):
            for k, ph in enumerate(phi_arr):
                X[i, j, k] = r * np.sin(th) * np.cos(ph)
                Y[i, j, k] = r * np.sin(th) * np.sin(ph)
                Z[i, j, k] = r * np.cos(th)

    
    print(f"[⚙️ TDP-3D CORE] Step 2: Extracting Monge Tensor Source Fields...")
    W_density = np.zeros((num_r, num_theta, num_phi), dtype=np.float64)
    for i, r in enumerate(r_arr):
        for j, th in enumerate(theta_arr):
            for k, ph in enumerate(phi_arr):
                x_p, y_p, z_p = X[i, j, k], Y[i, j, k], Z[i, j, k]
                g_v = float(grad_norm_eval(x_p, y_p, z_p))
                l_v = float(laplacian_eval(x_p, y_p, z_p))
                W_density[i, j, k] = np.sqrt(1.0 + g_v**2 + 0.1 * np.abs(l_v))

    
    P_src, Q_src, R_src = np.gradient(W_density)
    P_src *= alpha
    Q_src *= alpha
    R_src *= alpha

    
    print(f"[⚙️ TDP-3D SOLVER] Executing Poisson SOR Solver (Tol={tol:.1e}, Max_Iter={max_iter})...")
    
    dr = r_arr[1] - r_arr[0]
    dth = theta_arr[1] - theta_arr[0]
    dph = phi_arr[1] - phi_arr[0]

    converged = False
    for it in range(1, max_iter + 1):
        max_res = 0.0

        
        for i in range(1, num_r - 1):
            for j in range(1, num_theta - 1):
                for k in range(num_phi):
                    k_prev = (k - 1) % num_phi
                    k_next = (k + 1) % num_phi

                    
                    for Grid, Src in [(X, P_src), (Y, Q_src), (Z, R_src)]:
                        val_old = Grid[i, j, k]
                        
                        lap_val = (
                            (Grid[i+1, j, k] + Grid[i-1, j, k]) / (dr**2) +
                            (Grid[i, j+1, k] + Grid[i, j-1, k]) / (dth**2) +
                            (Grid[i, j, k_next] + Grid[i, j, k_prev]) / (dph**2)
                        )
                        denom = 2.0 / (dr**2) + 2.0 / (dth**2) + 2.0 / (dph**2)
                        
                        val_new = (lap_val - Src[i, j, k]) / denom
                        val_relaxed = (1.0 - omega) * val_old + omega * val_new
                        
                        Grid[i, j, k] = val_relaxed
                        max_res = max(max_res, abs(val_relaxed - val_old))

        if max_res < tol:
            converged = True
            print(f"[⚙️ TDP-3D SOLVER] SUCCESS: Converged at iteration {it} with residual {max_res:.4e}")
            break

    if not converged:
        warning_msg = (
            f"[⚠️ TDP-3D SOLVER FAILED] Reached max iteration ({max_iter}) "
            f"without converging to tol ({tol:.1e}). Final residual: {max_res:.4e}"
        )
        warnings.warn(warning_msg, RuntimeWarning)
        print(f"[⚠️ TDP-3D SOLVER] Reached max iteration ({max_iter}) with final residual {max_res:.4e}")
    
    vertices = []
    grid_shape = [num_r, num_theta, num_phi]

    for i in range(num_r):
        for j in range(num_theta):
            for k in range(num_phi):
                vertices.append({
                    "index": [i, j, k],
                    "pos": [float(X[i, j, k]), float(Y[i, j, k]), float(Z[i, j, k])]
                })

    mesh_data = {
        "metadata": {
            "solver": "Hybrid TDP-3D Monge-Poisson Core Engine",
            "converged": converged,
            "final_residual": float(max_res),
            "grid_shape": grid_shape
        },
        "grid_shape": grid_shape,
        "vertices": vertices
    }

    print(f"[⚙️ TDP-3D CORE] Complete 3D-driven hybrid grid successfully written to {out_file}")
    return mesh_data
