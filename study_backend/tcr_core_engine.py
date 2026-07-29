import json
import warnings
import numpy as np
import sympy as sp

# ==============================================================================
#  SECTION 1: SYSTEM DEFINITION & CONFIGURATION
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
        "num_theta_pts": 30,
        "num_phi": 40,
        "r_val": 5.0,
        "max_iter": 100,
        "tol": 1e-5,
        "output_filename": "tcr_mesh_journal.json"
    }

    return system_equations, mesh_config


# ==============================================================================
#  SECTION 2: TCR ENGINE WITH CHAOTIC METRIC INTEGRATION
# ==============================================================================

def execute_tcr_manifold_engine(system_input=None, config=None):
    
    if system_input is None or config is None:
        sys_eqs, sys_config = get_coupled_system_specification()
        system_input = sys_eqs if system_input is None else system_input
        config = sys_config if config is None else config

    num_r = config.get("num_r", 80)
    num_theta = config.get("num_theta_pts", 30)
    num_phi = config.get("num_phi", 40)
    r_val = config.get("r_val", 5.0)
    max_iter = config.get("max_iter", 100)
    tol = config.get("tol", 1e-5)

    # --------------------------------------------------------------------------
    #  Step 1
    # --------------------------------------------------------------------------
    vars_sym = [sp.Symbol(name, real=True) for name in ('x', 'y', 'z')]
    if isinstance(system_input, (list, tuple)):
        eq_syms = [sp.sympify(eq_str, locals={'pi': np.pi}) for eq_str in system_input]
    else:
        eq_syms = [sp.sympify(system_input, locals={'pi': np.pi})]

    # J = dF_i / dx_j
    J_sym = sp.Matrix([[sp.diff(f, v) for v in vars_sym] for f in eq_syms])
    
    #  det(I + J^T * J)
    I_mat = sp.eye(len(vars_sym))
    g_det_sym = (I_mat + J_sym.T * J_sym).det()
    w_chaotic_sym = sp.sqrt(sp.Abs(g_det_sym))

    f_scalar_sym = sum(eq**2 for eq in eq_syms)
    grad_syms = [sp.diff(f_scalar_sym, v) for v in vars_sym]
    grad_norm_sq_sym = sum(g**2 for g in grad_syms)

    w_chaotic_eval = sp.lambdify(vars_sym, w_chaotic_sym, modules=['numpy'])
    grad_norm_eval = sp.lambdify(vars_sym, sp.sqrt(grad_norm_sq_sym), modules=['numpy'])

    # --------------------------------------------------------------------------
    #  Step 2: (0 < theta < pi/2)
    # --------------------------------------------------------------------------
    eps = 1e-3
    r_arr = np.linspace(0.1, r_val, num_r)
    theta_base_arr = np.linspace(eps, (np.pi / 2.0) - eps, num_theta)
    phi_arr = np.linspace(0, 2 * np.pi, num_phi, endpoint=False)

    dr = r_arr[1] - r_arr[0]
    grid_shape = (num_r, num_theta, num_phi)

    # --------------------------------------------------------------------------
    #  Step 3:  n(r, theta)
    # --------------------------------------------------------------------------
    print(f"[TCR Engine] Evaluating chaotic manifold surface integral ({num_r}x{num_theta}x{num_phi})...")
    
    n_field = np.zeros(grid_shape, dtype=np.float64)
    g_val_matrix = np.zeros(grid_shape, dtype=np.float64)

    for j in range(num_theta):
        th_b = theta_base_arr[j]
        for k in range(num_phi):
            ph_c = phi_arr[k]
            n_integral = 0.0

            for i in range(num_r):
                r_c = r_arr[i]
                x_ref = r_c * np.sin(th_b) * np.cos(ph_c)
                y_ref = r_c * np.sin(th_b) * np.sin(ph_c)
                z_ref = r_c * np.cos(th_b)

                
                w_val = float(w_chaotic_eval(x_ref, y_ref, z_ref))
                g_v = float(grad_norm_eval(x_ref, y_ref, z_ref))
                g_val_matrix[i, j, k] = g_v

                if i > 0:
                    n_integral += w_val * dr

                n_field[i, j, k] = n_integral

    # --------------------------------------------------------------------------
    #  Step 4
    # --------------------------------------------------------------------------
    print(f"[TCR Engine] Solving theta mapping iteration (Tol={tol:.1e})...")

    alpha_scale = 0.15
    theta_mapped = np.tile(theta_base_arr[None, :, None], (num_r, 1, num_phi))

    residual_history = []
    converged = False

    for iteration in range(1, max_iter + 1):
        d_theta = alpha_scale * np.arctan(n_field / (1.0 + g_val_matrix))
        theta_new = np.clip(theta_base_arr[None, :, None] + d_theta, eps, (np.pi / 2.0) - eps)

        res = float(np.max(np.abs(theta_new - theta_mapped)))
        residual_history.append(res)

        theta_mapped = theta_new

        if res < tol:
            converged = True
            print(f"[TCR Engine] Manifold iteration converged at step {iteration}, final residual: {res:.4e}")
            break

    if not converged:
        warn_msg = f"[TCR Engine] Max iteration ({max_iter}) reached, final residual: {residual_history[-1]:.4e}"
        warnings.warn(warn_msg, RuntimeWarning)
        print(warn_msg)

    # --------------------------------------------------------------------------
    #  Step 5
    # --------------------------------------------------------------------------
    X = np.zeros(grid_shape, dtype=np.float64)
    Y = np.zeros(grid_shape, dtype=np.float64)
    Z = np.zeros(grid_shape, dtype=np.float64)

    for i in range(num_r):
        for j in range(num_theta):
            for k in range(num_phi):
                th_m = theta_mapped[i, j, k]
                ph_c = phi_arr[k]
                n_v = n_field[i, j, k]

                tan_th = np.tan(th_m)
                cot_th = 1.0 / max(1e-12, tan_th)

                # TCR 
                X[i, j, k] = np.cos(ph_c) * np.sqrt(np.abs(n_v * cot_th))
                Y[i, j, k] = np.sin(ph_c) * np.sqrt(np.abs(n_v * cot_th))
                
                sign_z = np.sign((np.pi / 2.0) - th_m)
                if sign_z == 0:
                    sign_z = 1.0
                Z[i, j, k] = sign_z * np.sqrt(np.abs(n_v * tan_th))

    # --------------------------------------------------------------------------
    #  Step 6
    # --------------------------------------------------------------------------
    vertices = []
    node_map = {}
    flat_counter = 0

    for i in range(num_r):
        for j in range(num_theta):
            for k in range(num_phi):
                vertices.append({
                    "index": [i, j, k],
                    "pos": [float(X[i, j, k]), float(Y[i, j, k]), float(Z[i, j, k])]
                })
                node_map[(i, j, k)] = flat_counter
                flat_counter += 1

    elements_hex8 = []
    for i in range(num_r - 1):
        for j in range(num_theta - 1):
            for k in range(num_phi):
                k_next = (k + 1) % num_phi

                n0 = node_map[(i,   j,   k)]
                n1 = node_map[(i+1, j,   k)]
                n2 = node_map[(i+1, j+1, k)]
                n3 = node_map[(i,   j+1, k)]

                n4 = node_map[(i,   j,   k_next)]
                n5 = node_map[(i+1, j,   k_next)]
                n6 = node_map[(i+1, j+1, k_next)]
                n7 = node_map[(i,   j+1, k_next)]

                elements_hex8.append([n0, n1, n2, n3, n4, n5, n6, n7])

    mesh_data = {
        "metadata": {
            "solver": "Chaotic Manifold 3D TCR Engine",
            "converged": converged,
            "final_iteration": len(residual_history),
            "final_residual": residual_history[-1],
            "residual_history": residual_history,
            "grid_shape": [num_r, num_theta, num_phi],
            "angle_bounds": "0 < theta < pi/2"
        },
        "grid_shape": [num_r, num_theta, num_phi],
        "vertices": vertices
    }

    print(f"[TCR Engine] Grid generation completed successfully.")
    return mesh_data


# ==============================================================================
#  MAIN EXECUTOR
# ==============================================================================

if __name__ == "__main__":
    sys_eqs, sys_config = get_coupled_system_specification()
    mesh_output = execute_tcr_manifold_engine(sys_eqs, sys_config)
    filename = sys_config["output_filename"]
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(mesh_output, f, indent=4)
    print(f"[TCR Engine] Output saved to {filename}")
