import json
import warnings
import numpy as np

# ==============================================================================
#  BLOCK 1: INPUT & SYSTEM SPECIFICATION BLOCK
# ==============================================================================

def get_coupled_system_specification():
    
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
    return None, mesh_config


# ==============================================================================
#  BLOCK 2: STANDARD ELLIPTIC PDE GRID GENERATOR (TTM METHOD)
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

    
    r_arr = np.linspace(0.1, r_val, num_r)
    theta_arr = np.linspace(eps, np.pi - eps, num_theta)
    phi_arr = np.linspace(0, 2 * np.pi, num_phi, endpoint=False) 

    dr = r_arr[1] - r_arr[0]
    dth = theta_arr[1] - theta_arr[0]
    dph = phi_arr[1] - phi_arr[0]

    
    w_r = 1.0 / (dr ** 2)
    w_th = 1.0 / (dth ** 2)
    w_ph = 1.0 / (dph ** 2)
    denom = 2.0 * (w_r + w_th + w_ph)

    X = np.zeros(grid_shape, dtype=np.float64)
    Y = np.zeros(grid_shape, dtype=np.float64)
    Z = np.zeros(grid_shape, dtype=np.float64)

    
    for j in range(num_theta):
        for k in range(num_phi):
            
            X[0, j, k] = r_arr[0] * np.sin(theta_arr[j]) * np.cos(phi_arr[k])
            Y[0, j, k] = r_arr[0] * np.sin(theta_arr[j]) * np.sin(phi_arr[k])
            Z[0, j, k] = r_arr[0] * np.cos(theta_arr[j])

            
            X[-1, j, k] = r_arr[-1] * np.sin(theta_arr[j]) * np.cos(phi_arr[k])
            Y[-1, j, k] = r_arr[-1] * np.sin(theta_arr[j]) * np.sin(phi_arr[k])
            Z[-1, j, k] = r_arr[-1] * np.cos(theta_arr[j])

    for i in range(1, num_r - 1):
        for k in range(num_phi):
            
            X[i, 0, k] = r_arr[i] * np.sin(theta_arr[0]) * np.cos(phi_arr[k])
            Y[i, 0, k] = r_arr[i] * np.sin(theta_arr[0]) * np.sin(phi_arr[k])
            Z[i, 0, k] = r_arr[i] * np.cos(theta_arr[0])

            X[i, -1, k] = r_arr[i] * np.sin(theta_arr[-1]) * np.cos(phi_arr[k])
            Y[i, -1, k] = r_arr[i] * np.sin(theta_arr[-1]) * np.sin(phi_arr[k])
            Z[i, -1, k] = r_arr[i] * np.cos(theta_arr[-1])

    
    for i in range(1, num_r - 1):
        factor = (r_arr[i] - r_arr[0]) / (r_arr[-1] - r_arr[0])
        for j in range(1, num_theta - 1):
            for k in range(num_phi):
                X[i, j, k] = (1.0 - factor) * X[0, j, k] + factor * X[-1, j, k]
                Y[i, j, k] = (1.0 - factor) * Y[0, j, k] + factor * Y[-1, j, k]
                Z[i, j, k] = (1.0 - factor) * Z[0, j, k] + factor * Z[-1, j, k]

    
    residual_history = []
    converged = False

    print(f"[⚙️ STANDARD PDE] Starting Standard Elliptic TTM SOR Solver (Tol={tol:.1e}, Max_Iter={max_iter})...")
    print(f"[⚙️ METRIC WEIGHTS] w_r={w_r:.3e}, w_theta={w_th:.3e}, w_phi={w_ph:.3e}")

    for iteration in range(1, max_iter + 1):
        max_diff = 0.0

        for i in range(1, num_r - 1):
            for j in range(1, num_theta - 1):
                for k in range(num_phi):
                    k_prev = (k - 1) % num_phi  
                    k_next = (k + 1) % num_phi

                    x_lap = (w_r * (X[i+1, j, k] + X[i-1, j, k]) +
                             w_th * (X[i, j+1, k] + X[i, j-1, k]) +
                             w_ph * (X[i, j, k_next] + X[i, j, k_prev])) / denom

                    y_lap = (w_r * (Y[i+1, j, k] + Y[i-1, j, k]) +
                             w_th * (Y[i, j+1, k] + Y[i, j-1, k]) +
                             w_ph * (Y[i, j, k_next] + Y[i, j, k_prev])) / denom

                    z_lap = (w_r * (Z[i+1, j, k] + Z[i-1, j, k]) +
                             w_th * (Z[i, j+1, k] + Z[i, j-1, k]) +
                             w_ph * (Z[i, j, k_next] + Z[i, j, k_prev])) / denom

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
            print(f"[⚙️ STANDARD PDE] SUCCESS: Converged at iteration {iteration} with residual {max_diff:.4e}")
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
            "solver": "Standard Elliptic Grid Generator (TTM Laplace-SOR)",
            "converged": converged,
            "final_iteration": len(residual_history),
            "tolerance": tol,
            "max_iterations": max_iter,
            "omega": omega,
            "metric_steps": {"d_r": dr, "d_theta": dth, "d_phi": dph}
        },
        "grid_shape": list(grid_shape),
        "residual_history": residual_history,
        "vertices": vertices
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(mesh_data, f, indent=4)

    print(f"[⚙️ PDE CORE] Standard mesh output written to {filename}")
    return mesh_data


# ==============================================================================
#  BLOCK 3: MAIN EXECUTOR
# ==============================================================================

if __name__ == "__main__":
    _, sys_config = get_coupled_system_specification()
    generate_pure_pde_3d_mesh(
        r_val=sys_config["r_val"], 
        num_r=sys_config["num_r"], 
        num_theta=sys_config["num_theta"], 
        num_phi=sys_config["num_phi"], 
        max_iter=sys_config["max_iter"],
        tol=sys_config["tol"],
        filename=sys_config["output_filename"]
    )
