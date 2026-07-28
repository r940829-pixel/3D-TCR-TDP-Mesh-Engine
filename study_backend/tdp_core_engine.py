import json
import warnings
import numpy as np
import sympy as sp
from matplotlib import rcParams

# Journal Template Formatting Configuration 
rcParams['font.family'] = 'serif'
rcParams['mathtext.fontset'] = 'cm'
rcParams['axes.linewidth'] = 1.0
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

# ==============================================================================
#  BLOCK 1: INPUT & SYSTEM SPECIFICATION BLOCK (ALIGN WITH TCR CORE)
# ==============================================================================

def get_coupled_system_specification():
    
    p_a, p_b, p_c = 10.0, 28.0, 8.0 / 3.0

    system_equations = (
        f"{p_a} * (y - x)",
        f"x * ({p_b} - z) - y",
        f"x * y - {p_c} * z"
    )

    mesh_config = {
        "var_names": ('x', 'y', 'z'),
        "domain_bounds": (-10.0, 10.0),
        "num_r": 80,                     
        "num_theta": 30,                 
        "num_phi": 40,                  
        "r_val": 5.0,
        "max_iter": 600,
        "tol": 1e-5,
        "omega": 1.75,                   
        "scaling_control": 0.15,         
        "output_filename": "tdp_mesh_journal.json"
    }

    return system_equations, mesh_config


# ==============================================================================
#  BLOCK 2: HYBRID TDP ENGINE (3D FIELD-DRIVEN TCR + POISSON SOR SOLVER)
# ==============================================================================

def generate_tdp_3d_mesh(system_input, config):
    
    var_names = config["var_names"]
    domain_bounds = config["domain_bounds"]
    num_r = config["num_r"]
    num_theta = config["num_theta"]
    num_phi = config["num_phi"]
    r_val = config["r_val"]
    max_iter = config["max_iter"]
    tol = config["tol"]
    omega = config["omega"]
    scaling_control = config["scaling_control"]
    filename = config["output_filename"]

    grid_shape = (num_r, num_theta, num_phi)

    # --------------------------------------------------------------------------
    #  STEP 1: GENERATE 3D FIELD-DRIVEN TCR MANIFOLD (ALIGNED WITH TCR ENGINE)
    # --------------------------------------------------------------------------
    print("[⚙️ TDP-3D CORE] Step 1: Synthesizing 3D Field-Driven TCR Source Manifold...")
    
    
    vars_sym = [sp.Symbol(name, real=True) for name in var_names]
    if isinstance(system_input, (list, tuple)):
        eq_syms = [sp.sympify(eq_str, locals={'pi': np.pi}) for eq_str in system_input]
        f_scalar_sym = sum(eq**2 for eq in eq_syms) 
    else:
        f_scalar_sym = sp.sympify(system_input, locals={'pi': np.pi})

    grad_syms = [sp.diff(f_scalar_sym, v) for v in vars_sym]
    grad_norm_sq_sym = sum(g**2 for g in grad_syms)
    laplacian_sym = sum(sp.diff(f_scalar_sym, v, 2) for v in vars_sym)

    f_eval = sp.lambdify(vars_sym, f_scalar_sym, modules=['numpy'])
    grad_norm_eval = sp.lambdify(vars_sym, sp.sqrt(grad_norm_sq_sym), modules=['numpy'])
    laplacian_eval = sp.lambdify(vars_sym, laplacian_sym, modules=['numpy'])

    r_arr = np.linspace(0.1, domain_bounds[1], num_r)
    phi_arr = np.linspace(0, 2 * np.pi, num_phi, endpoint=False) 

    n_vals = []
    theta_vals = []

    for r in r_arr:
        x_s, y_s, z_s = r / np.sqrt(3), r / np.sqrt(3), r / np.sqrt(3)
        f_val = float(f_eval(x_s, y_s, z_s))
        grad_val = float(grad_norm_eval(x_s, y_s, z_s))  
        lap_val = float(laplacian_eval(x_s, y_s, z_s))    

        s1 = np.sign(lap_val)  
        s2 = np.sign(grad_val - 1.0)

        
        if s1 > 0 and s2 > 0:
            region_weight = 1.25
        elif s1 > 0 and s2 < 0:
            region_weight = 0.85
        elif s1 == 0 or abs(grad_val) < 1e-3:
            region_weight = 1.00
        elif s1 < 0 and s2 > 0:
            region_weight = 1.40
        else:
            region_weight = 0.60

        n_f = np.abs(0.5 * (f_val**2)) * region_weight
        n_vals.append(n_f)

        
        rho = lap_val / (np.sqrt(grad_val**2 + lap_val**2) + 1e-6)
        theta_val = (np.pi / 4.0) + (np.pi / 8.0) * rho
        theta_vals.append(theta_val)

   
    X_tcr = np.zeros(grid_shape)
    Y_tcr = np.zeros(grid_shape)
    Z_tcr = np.zeros(grid_shape)

    for i in range(num_r):
        n_t = n_vals[i]
        theta_center = theta_vals[i]
        t_min, t_max = 1e-3, (np.pi / 2.0) - 1e-3
        scale_factor = theta_center / (np.pi / 4.0)
        upper_bound = min(t_max, t_max * scale_factor)
        theta_local = np.linspace(t_min, upper_bound, num_theta)

        for j, theta in enumerate(theta_local):
            cot_theta = np.abs(1.0 / np.tan(theta))
            tan_theta = np.abs(np.tan(theta))

            base_xy = np.sqrt(cot_theta * n_t)
            z_sign = np.sign((np.pi / 2.0) - theta)
            z_val = z_sign * np.sqrt(tan_theta * n_t)

            for k, phi in enumerate(phi_arr):
                X_tcr[i, j, k] = np.cos(phi) * base_xy
                Y_tcr[i, j, k] = np.sin(phi) * base_xy
                Z_tcr[i, j, k] = z_val

    # --------------------------------------------------------------------------
    #  STEP 2: EXTRACT 3D LAPLACIAN SOURCE TERMS WITH METRIC WEIGHTS
    # --------------------------------------------------------------------------
    print("[⚙️ TDP-3D CORE] Step 2: Extracting 3D Tensor Source Fields (P, Q, R)...")
    
    
    d_r = (r_arr[-1] - r_arr[0]) / (num_r - 1)
    d_theta = (np.pi - 2e-3) / (num_theta - 1)
    d_phi = 2 * np.pi / num_phi

    w_r = 1.0 / (d_r ** 2)
    w_theta = 1.0 / (d_theta ** 2)
    w_phi = 1.0 / (d_phi ** 2)
    denom = 2.0 * (w_r + w_theta + w_phi)

    P_src = np.zeros(grid_shape)
    Q_src = np.zeros(grid_shape)
    R_src = np.zeros(grid_shape)

    for i in range(1, num_r - 1):
        for j in range(1, num_theta - 1):
            for k in range(num_phi):
                k_p = (k - 1) % num_phi
                k_n = (k + 1) % num_phi

                
                P_src[i, j, k] = (w_r * (X_tcr[i+1, j, k] - 2*X_tcr[i, j, k] + X_tcr[i-1, j, k]) +
                                  w_theta * (X_tcr[i, j+1, k] - 2*X_tcr[i, j, k] + X_tcr[i, j-1, k]) +
                                  w_phi * (X_tcr[i, j, k_n] - 2*X_tcr[i, j, k] + X_tcr[i, j, k_p]))

                Q_src[i, j, k] = (w_r * (Y_tcr[i+1, j, k] - 2*Y_tcr[i, j, k] + Y_tcr[i-1, j, k]) +
                                  w_theta * (Y_tcr[i, j+1, k] - 2*Y_tcr[i, j, k] + Y_tcr[i, j-1, k]) +
                                  w_phi * (Y_tcr[i, j, k_n] - 2*Y_tcr[i, j, k] + Y_tcr[i, j, k_p]))

                R_src[i, j, k] = (w_r * (Z_tcr[i+1, j, k] - 2*Z_tcr[i, j, k] + Z_tcr[i-1, j, k]) +
                                  w_theta * (Z_tcr[i, j+1, k] - 2*Z_tcr[i, j, k] + Z_tcr[i, j-1, k]) +
                                  w_phi * (Z_tcr[i, j, k_n] - 2*Z_tcr[i, j, k] + Z_tcr[i, j, k_p]))

    # --------------------------------------------------------------------------
    #  STEP 3: PDE INITIALIZATION & DIRICHLET BOUNDARY CONDITIONS
    # --------------------------------------------------------------------------
    X = np.zeros(grid_shape)
    Y = np.zeros(grid_shape)
    Z = np.zeros(grid_shape)

    theta_boundary = np.linspace(1e-3, np.pi - 1e-3, num_theta)

    
    for j in range(num_theta):
        for k in range(num_phi):
            X[0, j, k] = r_arr[0] * np.sin(theta_boundary[j]) * np.cos(phi_arr[k])
            Y[0, j, k] = r_arr[0] * np.sin(theta_boundary[j]) * np.sin(phi_arr[k])
            Z[0, j, k] = r_arr[0] * np.cos(theta_boundary[j])

            X[-1, j, k] = r_arr[-1] * np.sin(theta_boundary[j]) * np.cos(phi_arr[k])
            Y[-1, j, k] = r_arr[-1] * np.sin(theta_boundary[j]) * np.sin(phi_arr[k])
            Z[-1, j, k] = r_arr[-1] * np.cos(theta_boundary[j])

    
    for i in range(1, num_r - 1):
        factor = (r_arr[i] - r_arr[0]) / (r_arr[-1] - r_arr[0])
        for j in range(num_theta):
            for k in range(num_phi):
                X[i, j, k] = (1.0 - factor) * X[0, j, k] + factor * X[-1, j, k]
                Y[i, j, k] = (1.0 - factor) * Y[0, j, k] + factor * Y[-1, j, k]
                Z[i, j, k] = (1.0 - factor) * Z[0, j, k] + factor * Z[-1, j, k]

    # --------------------------------------------------------------------------
    #  STEP 4: 3D POISSON SOR SOLVER WITH RESIDUAL DIAGNOSTICS
    # --------------------------------------------------------------------------
    print(f"[⚙️ TDP-3D SOLVER] Executing Poisson SOR Solver (Tol={tol}, Max_Iter={max_iter})...")
    residual_history = []
    converged = False

    for iteration in range(max_iter):
        max_diff = 0.0
        for i in range(1, num_r - 1):
            for j in range(1, num_theta - 1):
                for k in range(num_phi):
                    k_p = (k - 1) % num_phi
                    k_n = (k + 1) % num_phi

                    
                    x_new = (w_r * (X[i+1, j, k] + X[i-1, j, k]) +
                             w_theta * (X[i, j+1, k] + X[i, j-1, k]) +
                             w_phi * (X[i, j, k_n] + X[i, j, k_p]) - scaling_control * P_src[i, j, k]) / denom

                    y_new = (w_r * (Y[i+1, j, k] + Y[i-1, j, k]) +
                             w_theta * (Y[i, j+1, k] + Y[i, j-1, k]) +
                             w_phi * (Y[i, j, k_n] + Y[i, j, k_p]) - scaling_control * Q_src[i, j, k]) / denom

                    z_new = (w_r * (Z[i+1, j, k] + Z[i-1, j, k]) +
                             w_theta * (Z[i, j+1, k] + Z[i, j-1, k]) +
                             w_phi * (Z[i, j, k_n] + Z[i, j, k_p]) - scaling_control * R_src[i, j, k]) / denom

                    diff_x = omega * (x_new - X[i, j, k])
                    diff_y = omega * (y_new - Y[i, j, k])
                    diff_z = omega * (z_new - Z[i, j, k])

                    max_diff = max(max_diff, abs(diff_x), abs(diff_y), abs(diff_z))

                    X[i, j, k] += diff_x
                    Y[i, j, k] += diff_y
                    Z[i, j, k] += diff_z

        residual_history.append(float(max_diff))

        if max_diff < tol:
            converged = True
            print(f"[⚙️ TDP-3D SOLVER] SUCCESS: Converged at iteration {iteration} with residual {max_diff:.4e}")
            break

   
    if not converged:
        warnings.warn(
            f"[⚠️ TDP SOLVER FAILED] Engine reached max_iter ({max_iter}) without converging to tol ({tol}). "
            f"Final residual: {max_diff:.4e}",
            RuntimeWarning
        )

    # --------------------------------------------------------------------------
    #  STEP 5: DATA SERIALIZATION & JSON EXPORT
    # --------------------------------------------------------------------------
    vertices = []
    for i in range(num_r):
        for j in range(num_theta):
            for k in range(num_phi):
                vertices.append({
                    "index": [i, j, k],
                    "pos": [float(X[i, j, k]), float(Y[i, j, k]), float(Z[i, j, k])]
                })

    mesh_data = {
        "metadata": {
            "solver": "Hybrid TDP-3D (TCR-Modulated Poisson)",
            "converged": converged,
            "final_iteration": len(residual_history) - 1 if residual_history else 0,
            "tolerance": tol,
            "max_iterations": max_iter,
            "omega": omega,
            "scaling_control": scaling_control,
            "final_residual": residual_history[-1] if residual_history else max_diff,
            "resolution": list(grid_shape)
        },
        "grid_shape": list(grid_shape),
        "residual_history": residual_history,
        "vertices": vertices
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(mesh_data, f, indent=4)

    print(f"[⚙️ TDP-3D CORE] Complete 3D-driven hybrid grid successfully written to {filename}\n")
    return mesh_data


# ==============================================================================
#  BLOCK 3: MAIN EXECUTOR
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("   TDP (FULLY 3D-TCR DRIVEN PDE) MESH GENERATION PIPELINE")
    print("=" * 70)

    
    sys_eqs, sys_config = get_coupled_system_specification()
    
    
    mesh_results = generate_tdp_3d_mesh(sys_eqs, sys_config)
