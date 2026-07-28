import json
import warnings
import numpy as np

def generate_pure_pde_3d_mesh(
    r_val=5.0, 
    num_r=80, 
    num_theta=30, 
    num_phi=40, 
    max_iter=600, 
    tol=1e-5, 
    omega=1.75,
    filename="pde_mesh_journal.json"
):

    eps = 1e-3
    
    
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

    grid_shape = (num_r, num_theta, num_phi)
    X = np.zeros(grid_shape)
    Y = np.zeros(grid_shape)
    Z = np.zeros(grid_shape)

    
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

    print(f"[⚙️ PDE SOLVER] Starting SOR Solver (Tol={tol}, Max_Iter={max_iter})...")
    print(f"[⚙️ METRIC WEIGHTS] w_r={w_r:.3e}, w_theta={w_theta:.3e}, w_phi={w_phi:.3e}")

    for iteration in range(max_iter):
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
        warnings.warn(
            f"[⚠️ PDE SOLVER FAILED] Reached max_iter ({max_iter}) without converging to tol ({tol}). "
            f"Final residual: {max_diff:.4e}",
            RuntimeWarning
        )

   
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
            "solver": "Pure Elliptic PDE (Laplace-SOR)",
            "converged": converged,
            "final_iteration": len(residual_history) - 1,
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

if __name__ == "__main__":
    generate_pure_pde_3d_mesh(
        r_val=5.0, 
        num_r=80, 
        num_theta=30, 
        num_phi=40, 
        max_iter=600,
        tol=1e-5,
        filename="pde_mesh_journal.json"
    )
