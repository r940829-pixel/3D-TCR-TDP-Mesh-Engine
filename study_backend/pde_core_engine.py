import json
import warnings
import numpy as np
import sympy as sp

# ==============================================================================
#  BLOCK 1: INPUT & SYSTEM SPECIFICATION BLOCK
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
        "output_filename": "pde_mesh_journal.json"
    }

    return system_equations, mesh_config


# ==============================================================================
#  BLOCK 2: CHAOTIC-FIELD GUIDED ELLIPTIC PDE SOLVER
# ==============================================================================

def generate_pure_pde_3d_mesh(
    r_val=5.0, 
    num_r=80, 
    num_theta=30, 
    num_phi=40, 
    max_iter=600, 
    tol=1e-5, 
    omega=1.75,
    filename="pde_mesh_journal.json",
    system_input=None
):
    
    eps = 1e-3
    grid_shape = (num_r, num_theta, num_phi)

    
    if system_input is None:
        sys_eqs, _ = get_coupled_system_specification()
        system_input = sys_eqs

    vars_sym = [sp.Symbol(name, real=True) for name in ('x', 'y', 'z')]
    if isinstance(system_input, (list, tuple)):
        eq_syms = [sp.sympify(eq_str, locals={'pi': np.pi}) for eq_str in system_input]
        f_scalar_sym = sum(eq**2 for eq in eq_syms)
    else:
        f_scalar_sym = sp.sympify(system_input, locals={'pi': np.pi})

    grad_syms = [sp.diff(f_scalar_sym, v) for v in vars_sym]
    grad_norm_sq_sym = sum(g**2 for g in grad_syms)
    
    f_eval = sp.lambdify(vars_sym, f_scalar_sym, modules=['numpy'])
    grad_norm_eval = sp.lambdify(vars_sym, sp.sqrt(grad_norm_sq_sym), modules=['numpy'])

    
    xi = np.linspace(0.1, r_val, num_r)
    theta = np.linspace(eps, np.pi - eps, num_theta)
    phi = np.linspace(0, 2 * np.pi, num_phi, endpoint=False) 

    d_r = (xi[-1] - xi[0]) / (num_r - 1)
    d_theta = (theta[-1] - theta[0]) / (num_theta - 1)
    d_phi = 2 * np.pi / num_phi  

    w_r = 1.0 / (d_r ** 2)
    w_theta = 1.0 / (d_theta ** 2)
    w_phi = 1.0 / (d_phi ** 2)
    denom = 2.0 * (w_r + w_theta + w_phi)

    X = np.zeros(grid_shape, dtype=np.float64)
    Y = np.zeros(grid_shape, dtype=np.float64)
    Z = np.zeros(grid_shape, dtype=np.float64)

    
    for j in range(num_theta):
        for k in range(num_phi):
            
            X[0, j, k] = xi[0] * np.sin(theta[j]) * np.cos(phi[k])
            Y[0, j, k] = xi[0] * np.sin(theta[j]) * np.sin(phi[k])
            Z[0, j, k] = xi[0] * np.cos(theta[j])

            X[-1, j, k] = xi[-1] * np.sin(theta[j]) * np.cos(phi[k])
            Y[-1, j, k] = xi[-1] * np.sin(theta[j]) * np.sin(phi[k])
            Z[-1, j, k] = xi[-1] * np.cos(theta[j])

    for i in range(1, num_r - 1):
        for k in range(num_phi):
            X[i, 0, k] = xi[i] * np.sin(theta[0]) * np.cos(phi[k])
            Y[i, 0, k] = xi[i] * np.sin(theta[0]) * np.sin(phi[k])
            Z[i, 0, k] = xi[i] * np.cos(theta[0])

            X[i, -1, k] = xi[i] * np.sin(theta[-1]) * np.cos(phi[k])
            Y[i, -1, k] = xi[i] * np.sin(theta[-1]) * np.sin(phi[k])
            Z[i, -1, k] = xi[i] * np.cos(theta[-1])

    
    for i in range(1, num_r - 1):
        factor = (xi[i] - xi[0]) / (xi[-1] - xi[0])
        for j in range(1, num_theta - 1):
            for k in range(num_phi):
                X[i, j, k] = (1.0 - factor) * X[0, j, k] + factor * X[-1, j, k]
                Y[i, j, k] = (1.0 - factor) * Y[0, j, k] + factor * Y[-1, j, k]
                Z[i, j, k] = (1.0 - factor) * Z[0, j, k] + factor * Z[-1, j, k]

    
    residual_history = []
    converged = False

    print(f"[⚙️ PDE SOLVER] Starting Chaotic-Guided SOR Solver (Tol={tol:.1e}, Max_Iter={max_iter})...")
    print(f"[⚙️ METRIC WEIGHTS] w_r={w_r:.3e}, w_theta={w_theta:.3e}, w_phi={w_phi:.3e}")

    for iteration in range(1, max_iter + 1):
        max_diff = 0.0

        for i in range(1, num_r - 1):
            for j in range(1, num_theta - 1):
                for k in range(num_phi):
                    k_prev = (k - 1) % num_phi
                    k_next = (k + 1) % num_phi

                    
                    x_lap = (w_r * (X[i+1, j, k] + X[i-1, j, k]) +
                             w_theta * (X[i, j+1, k] + X[i, j-1, k]) +
                             w_phi * (X[i, j, k_next] + X[i, j, k_prev])) / denom

                    y_lap = (w_r * (Y[i+1, j, k] + Y[i-1, j, k]) +
                             w_theta * (Y[i, j+1, k] + Y[i, j-1, k]) +
                             w_phi * (Y[i, j, k_next] + Y[i, j, k_prev])) / denom

                    z_lap = (w_r * (Z[i+1, j, k] + Z[i-1, j, k]) +
                             w_theta * (Z[i, j+1, k] + Z[i, j-1, k]) +
                             w_phi * (Z[i, j, k_next] + Z[i, j, k_prev])) / denom

                    diff_x = omega * (x_lap - X[i, j, k])
                    diff_y = omega * (y_lap - Y[i, j, k])
                    diff_z = omega * (z_lap - Z[i, j, k])

                    max_diff = max(max_diff, abs(diff_x), abs(diff_y), abs(diff_z))

                    X[i, j, k] += diff_x
                    Y[i, j, k] += diff_y
                    Z[i, j, k] += diff_z

        residual_history.append(float(max_diff))

        if max_diff < tol:
            converged = True
            print(f"[⚙️ PDE SOLVER] SUCCESS: Converged at iteration {iteration} with residual {max_diff:.4e}")
            break

   
    if not converged:
        warn_msg = (
            f"[⚠️ PDE SOLVER FAILED] Reached max_iter ({max_iter}) without converging to tol ({tol:.1e}). "
            f"Final residual: {max_diff:.4e}"
        )
        warnings.warn(warn_msg, RuntimeWarning)
        print(warn_msg)

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
            "solver": "Chaotic-Guided Pure Elliptic PDE (Laplace-SOR)",
            "converged": converged,
            "final_iteration": len(residual_history),
            "tolerance": tol,
            "max_iterations": max_iter,
            "omega": omega,
            "metric_steps": {"d_r": d_r, "d_theta": d_theta, "d_phi": d_phi}
        },
        "grid_shape": list(grid_shape),
        "residual_history": residual_history,
        "vertices": vertices
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(mesh_data, f, indent=4)

    print(f"[⚙️ PDE CORE] Mesh output written to {filename}")
    return mesh_data


# ==============================================================================
#  BLOCK 3: MAIN EXECUTOR
# ==============================================================================

if __name__ == "__main__":
    sys_eqs, sys_config = get_coupled_system_specification()
    generate_pure_pde_3d_mesh(
        r_val=sys_config["r_val"], 
        num_r=sys_config["num_r"], 
        num_theta=sys_config["num_theta"], 
        num_phi=sys_config["num_phi"], 
        max_iter=sys_config["max_iter"],
        tol=sys_config["tol"],
        filename=sys_config["output_filename"],
        system_input=sys_eqs
    )
