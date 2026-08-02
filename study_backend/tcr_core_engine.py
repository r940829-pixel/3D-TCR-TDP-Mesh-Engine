import os
import json
import numpy as np

# ==============================================================================
#  SECTION 1: SYSTEM DEFINITION & CONFIGURATION
# ==============================================================================

def get_coupled_system_specification():
    
    system_equations = ()  
    mesh_config = {
        "var_names": ('x', 'y', 'z'),
        "num_theta_pts": 30,
        "num_phi": 40,
        "theta_min_deg": 1e-3,
        "theta_max_deg": np.pi - 1e-3,
        "n_val": 1.0,  
        "output_filename": "tcr_mesh_journal.json"
    }

    return system_equations, mesh_config


# ==============================================================================
#  SECTION 2: PURE HYPERBOLIC TCR ENGINE (NO R-SCALE, CONSTANT N = 1.0)
# ==============================================================================

def execute_tcr_manifold_engine(system_input=None, config=None):
    
    if config is None:
        _, config = get_coupled_system_specification()

    num_theta = config.get("num_theta_pts", 30)
    num_phi = config.get("num_phi", 40)
    theta_min_deg = config.get("theta_min_deg", 1e-3)
    theta_max_deg = config.get("theta_max_deg", np.pi - 1e-3)
    n_val = float(config.get("n_val", 1.0))

    print(f"[TCR Engine] Initializing Pure Hyperbolic TCR Engine (n = {n_val}, No r-scale)...")

    
    theta_deg = np.linspace(theta_min_deg, theta_max_deg, num_theta)
    theta_rad = np.radians(theta_deg)
    phi_rad = np.linspace(0, 2 * np.pi, num_phi, endpoint=False)

    TH, PH = np.meshgrid(theta_rad, phi_rad, indexing='ij')

    
    tan_TH = np.tan(TH)
    r_xy = np.sqrt(n_val / tan_TH)  

    X = np.cos(PH) * r_xy
    Y = np.sin(PH) * r_xy
    Z = np.sqrt(n_val * tan_TH)

   
    vertices = []
    node_map = {}
    flat_counter = 0

    for i in range(num_theta):
        for j in range(num_phi):
            vertices.append({
                "index": [i, j],
                "pos": [float(X[i, j]), float(Y[i, j]), float(Z[i, j])]
            })
            node_map[(i, j)] = flat_counter
            flat_counter += 1

   
    elements_quad4 = []
    for i in range(num_theta - 1):
        for j in range(num_phi):
            j_next = (j + 1) % num_phi

            n0 = node_map[(i, j)]
            n1 = node_map[(i + 1, j)]
            n2 = node_map[(i + 1, j_next)]
            n3 = node_map[(i, j_next)]

            elements_quad4.append([n0, n1, n2, n3])

    
    mesh_data = {
        "metadata": {
            "solver": "Pure Hyperbolic Closed-Form TCR Engine (No r-scale)",
            "n_constant": n_val,
            "implicit_equation": f"(X^2 + Y^2) * Z^2 = {n_val}",
            "grid_shape": [num_theta, num_phi],
            "theta_range_deg": [theta_min_deg, theta_max_deg]
        },
        "grid_shape": [num_theta, num_phi],
        "vertices": vertices,
        "elements": elements_quad4
    }

    print(f"[TCR Engine] Pure Hyperbolic Grid generation completed successfully.")
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

    print(f"[TCR Engine] Saved mesh file to '{filename}'")
