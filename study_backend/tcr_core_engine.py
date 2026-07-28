import json
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
        "num_theta_pts": 30,            
        "num_phi": 40,                   
        "output_filename": "tcr_mesh_journal.json"
    }

    return system_equations, mesh_config


# ==============================================================================
#  BLOCK 2: CORE TCR ALGORITHM ENGINE BLOCK (OPTION 2 IMPLEMENTATION)
# ==============================================================================

def execute_tcr_manifold_engine(system_input, config):

    var_names = config["var_names"]
    domain_bounds = config["domain_bounds"]
    num_r = config["num_r"]
    num_theta_pts = config["num_theta_pts"]
    num_phi = config["num_phi"]

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
    phi_arr = np.linspace(0, 2 * np.pi, num_phi)

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

    vertices = []
    grid_shape = (num_r, num_theta_pts, num_phi)

    for i in range(num_r):
        n_t = n_vals[i]
        theta_center = theta_vals[i]
        
        
        t_min, t_max = 1e-3, (np.pi / 2.0) - 1e-3
        scale_factor = theta_center / (np.pi / 4.0)
        
        
        upper_bound = min(t_max, t_max * scale_factor)
        theta_local = np.linspace(t_min, upper_bound, num_theta_pts)

        for j, theta in enumerate(theta_local):
            cot_theta = np.abs(1.0 / np.tan(theta))
            tan_theta = np.abs(np.tan(theta))

            base_xy = np.sqrt(cot_theta * n_t)
            z_sign = np.sign((np.pi / 2.0) - theta)
            z_val = z_sign * np.sqrt(tan_theta * n_t)

            for k, phi in enumerate(phi_arr):
                x_val = np.cos(phi) * base_xy
                y_val = np.sin(phi) * base_xy

                vertices.append({
                    "index": [i, j, k],
                    "pos": [float(x_val), float(y_val), float(z_val)]
                })

    return {"grid_shape": list(grid_shape), "vertices": vertices}


# ==============================================================================
#  BLOCK 3: OUTPUT PIPELINE & MAIN RUNNER BLOCK
# ==============================================================================

def export_mesh_database(mesh_data, filename):

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(mesh_data, f, indent=4)
    return len(mesh_data["vertices"])

if __name__ == "__main__":
    print("=" * 70)
    print("   MULTIVARIATE FIELD-DRIVEN TCR MESH GENERATION PIPELINE")
    print("=" * 70)


    sys_eqs, sys_config = get_coupled_system_specification()
    print(f"[Block 1: Input Loaded] Coupled System Defined.")


    print(f"[Block 2: Engine Processing] Executing TCR Option 2 Algorithm...")
    mesh_results = execute_tcr_manifold_engine(sys_eqs, sys_config)


    out_file = sys_config["output_filename"]
    total_nodes = export_mesh_database(mesh_results, out_file)
    
    print(f"[Block 3: Export Completed]")
    print(f"  ├─ Generated Mesh Nodes : {total_nodes}")
    print(f"  └─ Output File Path     : {out_file}\n")
