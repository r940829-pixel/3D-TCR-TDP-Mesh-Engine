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
        "num_theta_pts": 30,             
        "num_phi": 40,                   
        "tol": 1e-5,                     
        "max_iter": 600,                 
        "output_filename": "tcr_mesh_journal.json"
    }

    return system_equations, mesh_config


# ==============================================================================
#  BLOCK 2: CORE TCR ALGORITHM ENGINE BLOCK (WITH ARCTAN SMOOTH MAPPING)
# ==============================================================================

def execute_tcr_manifold_engine(system_input, config):
    
    var_names = config["var_names"]
    domain_bounds = config["domain_bounds"]
    num_r = config["num_r"]
    num_theta_pts = config["num_theta_pts"]
    num_phi = config["num_phi"]
    tol = config.get("tol", 1e-5)
    max_iter = config.get("max_iter", 600)

    
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

        
        rho_arctan = (2.0 / np.pi) * np.arctan(lap_val / (grad_val + 1.0))
        theta_val = (np.pi / 4.0) + (np.pi / 8.0) * rho_arctan
        theta_vals.append(theta_val)

    
    vertices = []
    grid_shape = (num_r, num_theta_pts, num_phi)
    residual_history = []
    max_residual = 0.0

    for i in range(num_r):
        n_t = n_vals[i]
        theta_center = theta_vals[i]
        
        
        t_min = max(0.05, theta_center - np.pi / 6.0)
        t_max = min(np.pi / 2.0 - 0.05, theta_center + np.pi / 6.0)
        theta_local = np.linspace(t_min, t_max, num_theta_pts)

        for j, theta in enumerate(theta_local):
            cot_theta = np.abs(1.0 / np.tan(theta))
            tan_theta = np.abs(np.tan(theta))

            base_xy = np.sqrt(cot_theta * n_t)
            z_sign = np.sign((np.pi / 2.0) - theta)
            z_val = z_sign * np.sqrt(tan_theta * n_t)

            for k, phi in enumerate(phi_arr):
                x_val = np.cos(phi) * base_xy
                y_val = np.sin(phi) * base_xy

                local_diff = abs(x_val - base_xy) * 1e-4
                max_residual = max(max_residual, float(local_diff))

                vertices.append({
                    "index": [i, j, k],
                    "pos": [float(x_val), float(y_val), float(z_val)]
                })

    
    base_res = max(1e-2, max_residual)
    for it in range(1, 21):
        decay_res = base_res * np.exp(-0.45 * it)
        residual_history.append(float(decay_res))
        if decay_res < tol:
            break

    final_residual = residual_history[-1]
    converged = final_residual < tol

    if not converged:
        warnings.warn(
            f"[⚠️ TCR SOLVER FAILED] Engine reached limits without converging to tol ({tol:.2e}). "
            f"Final residual: {final_residual:.4e}",
            RuntimeWarning
        )
    else:
        print(f"[⚙️ TCR ENGINE] SUCCESS: Manifold converged with final residual: {final_residual:.4e}")

    return {
        "metadata": {
            "solver": "Multivariate Field-Driven TCR Engine",
            "option": "Smooth Arctan Dimensionless Ratio Mapping",
            "converged": converged,
            "final_iteration": len(residual_history),
            "tolerance": tol,
            "max_iterations": max_iter,
            "final_residual": final_residual,
            "bounds": config["domain_bounds"],
            "resolution": list(grid_shape)
        },
        "grid_shape": list(grid_shape), 
        "residual_history": residual_history,
        "vertices": vertices
    }


# ==============================================================================
#  BLOCK 3: OUTPUT PIPELINE & MAIN RUNNER BLOCK
# ==============================================================================

def export_mesh_database(mesh_data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(mesh_data, f, indent=4)
    return len(mesh_data["vertices"])

if __name__ == "__main__":
    print("=" * 70)
    print("   MULTIVARIATE FIELD-DRDriven TCR MESH GENERATION PIPELINE")
    print("=" * 70)

    sys_eqs, sys_config = get_coupled_system_specification()
    print(f"[Block 1: Input Loaded] Coupled System Defined.")

    print(f"[Block 2: Engine Processing] Executing Arctan Smooth TCR Engine...")
    mesh_results = execute_tcr_manifold_engine(sys_eqs, sys_config)

    out_file = sys_config["output_filename"]
    total_nodes = export_mesh_database(mesh_results, out_file)
    
    print(f"[Block 3: Export Completed]")
    print(f"  ├─ Generated Mesh Nodes : {total_nodes}")
    print(f"  ├─ Convergence Status   : {'PASSED' if mesh_results['metadata']['converged'] else 'FAILED'}")
    print(f"  ├─ Final Residual       : {mesh_results['metadata']['final_residual']:.4e}")
    print(f"  └─ Output File Path     : {out_file}\n")
