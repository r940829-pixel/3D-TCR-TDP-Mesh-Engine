import os
import json
import time
import warnings
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib import rcParams


rcParams['font.family'] = 'serif'
rcParams['mathtext.fontset'] = 'cm'
rcParams['axes.linewidth'] = 1.0
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'


from tcr_core_engine import execute_tcr_manifold_engine, get_coupled_system_specification
from pde_core_engine import generate_pure_pde_3d_mesh

R_BOUND = 10.0  
MESH_OUTPUT_DIR = "generated_meshes"  


os.makedirs(MESH_OUTPUT_DIR, exist_ok=True)

# ==============================================================================
#  SECTION 1: FULL 3D MANUFACTURED SOLUTION & SOURCE TERM
# ==============================================================================

def exact_u_3d(x, y, z):
    """ u(x, y, z) = sin(pi*x/R) * sin(pi*y/R) * cos(pi*z/R) + 1.0"""
    k = np.pi / R_BOUND
    return np.sin(k * x) * np.sin(k * y) * np.cos(k * z) + 1.0

def exact_grad_u_3d(x, y, z):
    """ [du/dx, du/dy, du/dz]"""
    k = np.pi / R_BOUND
    du_dx =  k * np.cos(k * x) * np.sin(k * y) * np.cos(k * z)
    du_dy =  k * np.sin(k * x) * np.cos(k * y) * np.cos(k * z)
    du_dz = -k * np.sin(k * x) * np.sin(k * y) * np.sin(k * z)
    return np.array([du_dx, du_dy, du_dz])

def source_term_f_3d(x, y, z):
    """ f(x, y, z) = - (d2u/dx2 + d2u/dy2 + d2u/dz2)"""
    k = np.pi / R_BOUND
    return 3.0 * (k ** 2) * np.sin(k * x) * np.sin(k * y) * np.cos(k * z)

# ==============================================================================
#  SECTION 2: 3D HEXAHEDRAL (HEX8) MESH RECONSTRUCTION & FEM SOLVER
# ==============================================================================

def load_and_build_3d_hex_mesh(json_filepath):

    with open(json_filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    grid_shape = data["grid_shape"]
    vertices = data["vertices"]
    num_r, num_theta, num_phi = grid_shape

    nodes_3d = np.zeros((len(vertices), 3), dtype=np.float64)
    boundary_mask = np.zeros(len(vertices), dtype=bool)
    node_map = {}

    for node in vertices:
        i, j, k = node["index"]
        flat_idx = i * (num_theta * num_phi) + j * num_phi + k
        nodes_3d[flat_idx] = node["pos"]
        node_map[(i, j, k)] = flat_idx

        
        if i == 0 or i == num_r - 1:
            boundary_mask[flat_idx] = True

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

    return nodes_3d, np.array(elements_hex8), boundary_mask

def solve_poisson_3d_fem(case_name, json_filepath):
    
    try:
        nodes, elements, boundary_mask = load_and_build_3d_hex_mesh(json_filepath)
    except FileNotFoundError:
        print(f"[❌ ERROR] Cannot find mesh file: '{json_filepath}'. Skipping.")
        return None

    num_nodes = len(nodes)
    num_elems = len(elements)

    K = sp.dok_matrix((num_nodes, num_nodes), dtype=np.float64)
    F = np.zeros(num_nodes, dtype=np.float64)

    t0_asm = time.perf_counter()

    # (2x2x2 Gauss Quadrature)
    g_p = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
    gauss_pts_3d = [(xi, eta, zeta) for xi in g_p for eta in g_p for zeta in g_p]

    valid_elements_count = 0
    inverted_elements_count = 0    
    degenerate_elements_count = 0  

    def hex8_shape_funcs(xi, eta, zeta):
        N = 0.125 * np.array([
            (1-xi)*(1-eta)*(1-zeta), (1+xi)*(1-eta)*(1-zeta),
            (1+xi)*(1+eta)*(1-zeta), (1-xi)*(1+eta)*(1-zeta),
            (1-xi)*(1-eta)*(1+zeta), (1+xi)*(1-eta)*(1+zeta),
            (1+xi)*(1+eta)*(1+zeta), (1-xi)*(1+eta)*(1+zeta)
        ])
        dN_dxi = 0.125 * np.array([
            [-(1-eta)*(1-zeta),  (1-eta)*(1-zeta),  (1+eta)*(1-zeta), -(1+eta)*(1-zeta),
             -(1-eta)*(1+zeta),  (1-eta)*(1+zeta),  (1+eta)*(1+zeta), -(1+eta)*(1+zeta)],
            [-(1-xi)*(1-zeta),  -(1+xi)*(1-zeta),   (1+xi)*(1-zeta),   (1-xi)*(1-zeta),
             -(1-xi)*(1+zeta),  -(1+xi)*(1+zeta),   (1+xi)*(1+zeta),   (1-xi)*(1+zeta)],
            [-(1-xi)*(1-eta),   -(1+xi)*(1-eta),   -(1+xi)*(1+eta),   -(1-xi)*(1+eta),
              (1-xi)*(1-eta),    (1+xi)*(1-eta),    (1+xi)*(1+eta),    (1-xi)*(1+eta)]
        ])
        return N, dN_dxi

    
    for elem in elements:
        elem_nodes = nodes[elem].copy()

        
        xi0, eta0, zeta0 = gauss_pts_3d[0]
        _, dN_dp0 = hex8_shape_funcs(xi0, eta0, zeta0)
        J0 = np.dot(dN_dp0, elem_nodes)
        if np.linalg.det(J0) < 0.0:
            elem_nodes = elem_nodes[[4, 5, 6, 7, 0, 1, 2, 3]]

        K_elem = np.zeros((8, 8))
        F_elem = np.zeros(8)
        
        is_inverted = False
        is_degenerate = False

        for xi, eta, zeta in gauss_pts_3d:
            N, dN_dparent = hex8_shape_funcs(xi, eta, zeta)
            J_3d = np.dot(dN_dparent, elem_nodes)
            det_J = np.linalg.det(J_3d)

            if det_J <= 0.0:
                is_inverted = True
                break
            elif det_J <= 1e-10:
                is_degenerate = True
                break

            inv_J = np.linalg.inv(J_3d)
            dN_dx = np.dot(inv_J, dN_dparent)

            pos_gp = np.dot(N, elem_nodes)
            wt = det_J

            K_elem += np.dot(dN_dx.T, dN_dx) * wt
            F_elem += N * source_term_f_3d(pos_gp[0], pos_gp[1], pos_gp[2]) * wt

        if is_inverted:
            inverted_elements_count += 1
        elif is_degenerate:
            degenerate_elements_count += 1
        else:
            valid_elements_count += 1
            for a in range(8):
                F[elem[a]] += F_elem[a]
                for b in range(8):
                    K[elem[a], elem[b]] += K_elem[a, b]

    t1_asm = time.perf_counter()
    assembly_ms = (t1_asm - t0_asm) * 1000.0

    
    total_defective = inverted_elements_count + degenerate_elements_count
    if total_defective > 0:
        warn_msg = (
            f"[{case_name}] Found {total_defective} defective element(s) "
            f"(Inv: {inverted_elements_count}, Deg: {degenerate_elements_count})."
        )
        warnings.warn(warn_msg, RuntimeWarning)
        print(f"  ├─ [⚠️ MESH DIAGNOSTICS] Total Defective Elements: {total_defective} "
              f"(Inverted det(J)<=0: {inverted_elements_count}, Degenerate det(J)<=1e-10: {degenerate_elements_count})")
    else:
        print(f"  ├─ [✅ MESH DIAGNOSTICS] Perfect Mesh Topology! Zero Inverted or Degenerate Elements.")

    
    K = K.tolil()
    boundary_indices = np.where(boundary_mask)[0]

    for b_idx in boundary_indices:
        real_x, real_y, real_z = nodes[b_idx]
        g_val = exact_u_3d(real_x, real_y, real_z)
        K[b_idx, :] = 0.0
        K[b_idx, b_idx] = 1.0
        F[b_idx] = g_val

    diag_vals = K.diagonal()
    zero_diags = np.where(np.abs(diag_vals) < 1e-12)[0]
    for z_idx in zero_diags:
        K[z_idx, z_idx] = 1.0
        F[z_idx] = exact_u_3d(nodes[z_idx, 0], nodes[z_idx, 1], nodes[z_idx, 2])

    K = K.tocsr()

    
    t0_sol = time.perf_counter()
    u_h = spla.spsolve(K, F)
    t1_sol = time.perf_counter()
    solve_ms = (t1_sol - t0_sol) * 1000.0

    
    l2_err_sq = 0.0
    h1_err_sq = 0.0
    l2_norm_ex_sq = 0.0

    for elem in elements:
        elem_nodes = nodes[elem].copy()
        
        xi0, eta0, zeta0 = gauss_pts_3d[0]
        _, dN_dp0 = hex8_shape_funcs(xi0, eta0, zeta0)
        if np.linalg.det(np.dot(dN_dp0, elem_nodes)) < 0.0:
            elem_nodes = elem_nodes[[4, 5, 6, 7, 0, 1, 2, 3]]

        u_elem = u_h[elem]

        for xi, eta, zeta in gauss_pts_3d:
            N, dN_dparent = hex8_shape_funcs(xi, eta, zeta)
            J_3d = np.dot(dN_dparent, elem_nodes)
            det_J = np.linalg.det(J_3d)

            if det_J <= 1e-10:
                continue

            inv_J = np.linalg.inv(J_3d)
            dN_dx = np.dot(inv_J, dN_dparent)

            pos_gp = np.dot(N, elem_nodes)
            u_h_gp = np.dot(N, u_elem)
            grad_u_h_gp = np.dot(dN_dx, u_elem)

            u_ex_gp = exact_u_3d(pos_gp[0], pos_gp[1], pos_gp[2])
            grad_u_ex_gp = exact_grad_u_3d(pos_gp[0], pos_gp[1], pos_gp[2])

            wt = det_J
            l2_err_sq += ((u_h_gp - u_ex_gp) ** 2) * wt
            h1_err_sq += np.sum((grad_u_h_gp - grad_u_ex_gp) ** 2) * wt
            l2_norm_ex_sq += (u_ex_gp ** 2) * wt

    rel_l2_error = np.sqrt(l2_err_sq) / max(1e-12, np.sqrt(l2_norm_ex_sq))
    abs_h1_error = np.sqrt(h1_err_sq)

    elem_diags = [np.linalg.norm(nodes[e[6]] - nodes[e[0]]) for e in elements]
    h_3d = float(np.mean(elem_diags)) if elem_diags else 0.0

    return {
        "case": case_name,
        "nodes": num_nodes,
        "elements": num_elems,
        "valid_elems": valid_elements_count,
        "inverted_elems": inverted_elements_count,
        "degenerate_elems": degenerate_elements_count,
        "h": h_3d,
        "assembly_ms": assembly_ms,
        "solve_ms": solve_ms,
        "l2_error": rel_l2_error,
        "h1_error": abs_h1_error
    }

# ==============================================================================
#  SECTION 3: AUTOMATED DUAL-CASE 7-LEVEL MESH CONVERGENCE & EOC SUITE
# ==============================================================================

MESH_LEVELS_7 = [
    {"name": "Level 1", "num_r": 16, "num_theta_pts": 8,  "num_phi": 12},
    {"name": "Level 2", "num_r": 24, "num_theta_pts": 10, "num_phi": 16},
    {"name": "Level 3", "num_r": 32, "num_theta_pts": 14, "num_phi": 20},
    {"name": "Level 4", "num_r": 48, "num_theta_pts": 18, "num_phi": 24},
    {"name": "Level 5", "num_r": 60, "num_theta_pts": 22, "num_phi": 28},
    {"name": "Level 6", "num_r": 72, "num_theta_pts": 26, "num_phi": 32},
    {"name": "Level 7", "num_r": 84, "num_theta_pts": 30, "num_phi": 40}
]

def run_dualcase_study():
    print("=" * 110)
    print("   DUAL-CASE 3D FEM CONVERGENCE BENCHMARK: CASE A (PURE TCR) vs. CASE B (PURE PDE BASELINE)")
    print("=" * 110)

    cases = [
        {"id": "Case A", "name": "Case A: Pure 3D TCR Manifold",       "prefix": "tcr_mesh"},
        {"id": "Case B", "name": "Case B: Pure Laplace PDE Baseline",  "prefix": "pde_mesh"}
    ]

    all_case_results = {}
    sys_eqs, sys_config = get_coupled_system_specification()

    for c in cases:
        print(f"\n" + "#"*70)
        print(f"   RUNNING PIPELINE FOR: {c['name']}")
        print("#"*70)
        
        case_results = []
        for lvl_idx, lvl in enumerate(MESH_LEVELS_7):
            filename = f"{c['prefix']}_lvl{lvl_idx+1}.json"
            filepath = os.path.join(MESH_OUTPUT_DIR, filename)  # 打包至生成的資料夾中
            
            print(f"\n[🔄 GRID & FEM] {c['id']} - {lvl['name']} ({lvl['num_r']}x{lvl['num_theta_pts']}x{lvl['num_phi']})...")

            if c["id"] == "Case A":
                sys_config.update({
                    "num_r": lvl["num_r"], 
                    "num_theta_pts": lvl["num_theta_pts"], 
                    "num_phi": lvl["num_phi"],
                    "output_filename": filepath
                })
                mesh_data = execute_tcr_manifold_engine(sys_eqs, sys_config)
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(mesh_data, f, indent=4)

            else:  # Case B: Standard Elliptic PDE Baseline
                mesh_data = generate_pure_pde_3d_mesh(
                    r_val=5.0, 
                    num_r=lvl["num_r"], 
                    num_theta=lvl["num_theta_pts"], 
                    num_phi=lvl["num_phi"], 
                    max_iter=600, 
                    tol=1e-5,
                    filename=filepath
                )

            res = solve_poisson_3d_fem(c["name"], filepath)
            if res is not None:
                case_results.append(res)

        if not case_results:
            print(f"[❌ ERROR] No valid results produced for {c['id']}. Skipping EOC.")
            continue

        h_vals = [r["h"] for r in case_results]
        l2_errs = [r["l2_error"] for r in case_results]
        h1_errs = [r["h1_error"] for r in case_results]

        eoc_l2 = [0.0]
        eoc_h1 = [0.0]
        for i in range(1, len(case_results)):
            dh = np.log(h_vals[i] / h_vals[i-1])
            eoc_l2.append(np.log(l2_errs[i] / l2_errs[i-1]) / dh)
            eoc_h1.append(np.log(h1_errs[i] / h1_errs[i-1]) / dh)

        all_case_results[c["id"]] = {
            "name": c["name"],
            "results": case_results,
            "h_vals": h_vals,
            "l2_errs": l2_errs,
            "h1_errs": h1_errs,
            "eoc_l2": eoc_l2,
            "eoc_h1": eoc_h1
        }

    
    print("\n" + "="*125)
    print(" INTERNATIONAL PUBLICATION REPORT: DUAL-CASE TCR (CASE A) vs. PDE BASELINE (CASE B) EOC BENCHMARK")
    print("="*125)
    print(f"{'Case ID & Mesh Level':<24} | {'Char. h':<10} | {'Defective Elems (Inv/Deg)':<26} | {'Volume L2 Error':<16} | {'EOC(L2)':<8} | {'Volume H1 Error':<16} | {'EOC(H1)':<8}")
    print("-" * 125)

    for cid in ["Case A", "Case B"]:
        if cid not in all_case_results:
            continue
        cdata = all_case_results[cid]
        print(f"--- {cdata['name']} ---")
        for i in range(len(cdata['results'])):
            r = cdata["results"][i]
            eoc_l2_str = f"{cdata['eoc_l2'][i]:.2f}" if i > 0 else "N/A"
            eoc_h1_str = f"{cdata['eoc_h1'][i]:.2f}" if i > 0 else "N/A"
            defective_str = f"{r['inverted_elems']} / {r['degenerate_elems']}"
            print(f"{MESH_LEVELS_7[i]['name']:<24} | {r['h']:<10.4f} | {defective_str:<26} | {r['l2_error']:<16.6e} | {eoc_l2_str:<8} | {r['h1_error']:<16.6e} | {eoc_h1_str:<8}")
        print("-" * 125)

    if all_case_results:
        plot_dualcase_convergence_curves(all_case_results)

def plot_dualcase_convergence_curves(all_case_results):
    """繪製包含 Case A (TCR) 與 Case B (PDE Baseline) 的收斂圖。"""
    plt.figure(figsize=(8.5, 6), dpi=300)

    styles = {
        "Case A": {"color": "navy", "marker": "o", "ls": "-"},
        "Case B": {"color": "crimson", "marker": "s", "ls": "--"}
    }

    for cid in ["Case A", "Case B"]:
        if cid not in all_case_results:
            continue
        cdata = all_case_results[cid]
        st = styles[cid]
        avg_eoc = np.mean(cdata["eoc_l2"][1:]) if len(cdata["eoc_l2"]) > 1 else 0.0
        plt.loglog(cdata["h_vals"], cdata["l2_errs"], color=st["color"], marker=st["marker"], 
                   linestyle=st["ls"], linewidth=2, markersize=7, 
                   label=f'{cid} ($L_2$ Error, Avg EOC = {avg_eoc:.2f})')

    if "Case B" in all_case_results:
        ref_h = np.array(all_case_results["Case B"]["h_vals"])
        ref_l2_base = all_case_results["Case B"]["l2_errs"][0]
        plt.loglog(ref_h, ref_l2_base * (ref_h / ref_h[0])**2, 'k:', alpha=0.6, label='Theoretical $O(h^2)$ Slope')
        plt.loglog(ref_h, ref_l2_base * (ref_h / ref_h[0])**1, 'k-.', alpha=0.6, label='Theoretical $O(h^1)$ Slope')

    plt.title("3D FEM Dual-Case Mesh Convergence & EOC Benchmark (Case A: TCR vs. Case B: PDE)", fontsize=11, fontweight='bold')
    plt.xlabel("Characteristic Mesh Size $h$", fontsize=11)
    plt.ylabel("Volume $L_2$ Relative Error Norm", fontsize=11)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(fontsize=9)
    plt.tight_layout()

    plot_path = "dualcase_7level_convergence_eoc_plot.png"
    plt.savefig(plot_path, dpi=300)
    print(f"\n[📊 PLOT SAVED] Dual-case convergence chart saved as '{plot_path}'\n")
    plt.show()

# ==============================================================================
#  MAIN ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    run_dualcase_study()