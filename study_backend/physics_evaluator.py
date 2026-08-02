import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['mathtext.fontset'] = 'cm'
rcParams['axes.linewidth'] = 1.0
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

MESH_DIR = "generated_meshes"  
REPORT_DIR = "quality_reports"
VIS_DIR = os.path.join(REPORT_DIR, "visualizations")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)


class MeshQualityEvaluator:

    def __init__(self, json_filepath):
        self.filepath = json_filepath
        self.filename = os.path.basename(json_filepath)
        with open(json_filepath, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.grid_shape = self.data["grid_shape"]
        self.vertices = self.data["vertices"]
        self.is_3d = (len(self.grid_shape) == 3)

        if self.is_3d:
            self.num_r, self.num_theta, self.num_phi = self.grid_shape
        else:
            self.num_theta, self.num_phi = self.grid_shape
            self.num_r = 1

        self._build_mesh_topology()

    def _build_mesh_topology(self):
        self.nodes = np.zeros((len(self.vertices), 3), dtype=np.float64)
        node_map = {}

        if self.is_3d:
            for v in self.vertices:
                i, j, k = v["index"]
                flat_idx = i * (self.num_theta * self.num_phi) + j * self.num_phi + k
                self.nodes[flat_idx] = v["pos"]
                node_map[(i, j, k)] = flat_idx

            self.elements = []
            self.elem_indices = []

            for i in range(self.num_r - 1):
                for j in range(self.num_theta - 1):
                    for k in range(self.num_phi):
                        k_next = (k + 1) % self.num_phi

                        n0 = node_map[(i,   j,   k)]
                        n1 = node_map[(i+1, j,   k)]
                        n2 = node_map[(i+1, j+1, k)]
                        n3 = node_map[(i,   j+1, k)]

                        n4 = node_map[(i,   j,   k_next)]
                        n5 = node_map[(i+1, j,   k_next)]
                        n6 = node_map[(i+1, j+1, k_next)]
                        n7 = node_map[(i,   j+1, k_next)]

                        self.elements.append([n0, n1, n2, n3, n4, n5, n6, n7])
                        self.elem_indices.append((i, j, k))

        else:
            for v in self.vertices:
                i, j = v["index"]
                flat_idx = i * self.num_phi + j
                self.nodes[flat_idx] = v["pos"]
                node_map[(i, j)] = flat_idx

            self.elements = []
            self.elem_indices = []

            for i in range(self.num_theta - 1):
                for j in range(self.num_phi):
                    j_next = (j + 1) % self.num_phi

                    n0 = node_map[(i, j)]
                    n1 = node_map[(i + 1, j)]
                    n2 = node_map[(i + 1, j_next)]
                    n3 = node_map[(i, j_next)]

                    self.elements.append([n0, n1, n2, n3])
                    self.elem_indices.append((0, i, j))

        self.elements = np.array(self.elements)

    @staticmethod
    def _hex8_shape_derivatives(xi, eta, zeta):
        return 0.125 * np.array([
            [-(1-eta)*(1-zeta),  (1-eta)*(1-zeta),  (1+eta)*(1-zeta), -(1+eta)*(1-zeta),
             -(1-eta)*(1+zeta),  (1-eta)*(1+zeta),  (1+eta)*(1+zeta), -(1+eta)*(1+zeta)],
            [-(1-xi)*(1-zeta),  -(1+xi)*(1-zeta),   (1+xi)*(1-zeta),   (1-xi)*(1-zeta),
             -(1-xi)*(1+zeta),  -(1+xi)*(1+zeta),   (1+xi)*(1+zeta),   (1-xi)*(1+zeta)],
            [-(1-xi)*(1-eta),   -(1+xi)*(1-eta),   -(1+xi)*(1+eta),   -(1-xi)*(1+eta),
              (1-xi)*(1-eta),    (1+xi)*(1-eta),    (1+xi)*(1+eta),    (1-xi)*(1+eta)]
        ])

    def evaluate_mesh(self):
        metrics = []
        inverted_list = []

        if self.is_3d:
            g_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
            gauss_3d = [(xi, eta, zeta) for xi in g_pts for eta in g_pts for zeta in g_pts]

            for elem_id, (elem, idx) in enumerate(zip(self.elements, self.elem_indices)):
                pts = self.nodes[elem].copy()
                i_idx, j_idx, k_idx = idx

                dN_center = self._hex8_shape_derivatives(0.0, 0.0, 0.0)
                J_center = np.dot(dN_center, pts)
                det_J_center = np.linalg.det(J_center)

                if det_J_center < 0.0:
                    pts = pts[[4, 5, 6, 7, 0, 1, 2, 3]]
                    J_center = np.dot(dN_center, pts)
                    det_J_center = np.linalg.det(J_center)

                det_J_min = det_J_center
                for xi, eta, zeta in gauss_3d:
                    dN = self._hex8_shape_derivatives(xi, eta, zeta)
                    J_gp = np.dot(dN, pts)
                    det_J_gp = np.linalg.det(J_gp)
                    if det_J_gp < det_J_min:
                        det_J_min = det_J_gp

                is_inverted = det_J_min <= 0.0

                if abs(det_J_center) > 1e-12:
                    inv_J = np.linalg.inv(J_center)
                    norm_J = np.linalg.norm(J_center, 'fro')
                    norm_inv_J = np.linalg.norm(inv_J, 'fro')
                    cond_num = (norm_J * norm_inv_J) / 3.0
                else:
                    cond_num = 1e6

                edges = [
                    pts[1]-pts[0], pts[2]-pts[1], pts[3]-pts[2], pts[0]-pts[3],
                    pts[5]-pts[4], pts[6]-pts[5], pts[7]-pts[6], pts[4]-pts[7],
                    pts[4]-pts[0], pts[5]-pts[1], pts[6]-pts[2], pts[7]-pts[3]
                ]
                edge_lens = [np.linalg.norm(e) for e in edges]
                aspect_ratio = max(edge_lens) / max(1e-4, min(edge_lens))

                v_xi, v_eta, v_zeta = J_center[0, :], J_center[1, :], J_center[2, :]
                norm_xi, norm_eta, norm_zeta = np.linalg.norm(v_xi), np.linalg.norm(v_eta), np.linalg.norm(v_zeta)

                def get_angle_error(v1, v2, n1, n2):
                    if n1 * n2 > 1e-12:
                        cos_val = abs(np.dot(v1, v2) / (n1 * n2))
                        return np.arcsin(np.clip(cos_val, 0.0, 1.0)) * (180.0 / np.pi)
                    return 90.0

                err_xi_eta = get_angle_error(v_xi, v_eta, norm_xi, norm_eta)
                err_eta_zeta = get_angle_error(v_eta, v_zeta, norm_eta, norm_zeta)
                err_zeta_xi = get_angle_error(v_zeta, v_xi, norm_zeta, norm_xi)
                max_ortho_error = max(err_xi_eta, err_eta_zeta, err_zeta_xi)
                skewness = max_ortho_error / 90.0

                is_pole = (j_idx == 0) or (j_idx == self.num_theta - 2)
                is_equator = abs(j_idx - (self.num_theta // 2)) <= 1
                is_seam = (k_idx == 0) or (k_idx == self.num_phi - 1)

                region_tag = "Interior"
                if is_pole:
                    region_tag = "Pole"
                elif is_equator:
                    region_tag = "Equator"
                if is_seam:
                    region_tag += "+Seam" if region_tag != "Interior" else "Seam"

                center_pos = np.mean(pts, axis=0)

                record = {
                    "elem_id": elem_id,
                    "grid_idx_i_j_k": f"{i_idx}_{j_idx}_{k_idx}",
                    "region": region_tag,
                    "center_x": center_pos[0],
                    "center_y": center_pos[1],
                    "center_z": center_pos[2],
                    "det_J_min": det_J_min,
                    "det_J_center": det_J_center,
                    "aspect_ratio": aspect_ratio,
                    "skewness": skewness,
                    "cond_number": cond_num,
                    "ortho_err_xi_eta_deg": err_xi_eta,
                    "ortho_err_eta_zeta_deg": err_eta_zeta,
                    "ortho_err_zeta_xi_deg": err_zeta_xi,
                    "max_ortho_err_deg": max_ortho_error,
                    "is_inverted": is_inverted
                }
                metrics.append(record)
                if is_inverted:
                    inverted_list.append(record)

        else:
            
            for elem_id, (elem, idx) in enumerate(zip(self.elements, self.elem_indices)):
                pts = self.nodes[elem].copy()
                _, j_idx, k_idx = idx

                
                e0 = pts[1] - pts[0]
                e1 = pts[2] - pts[1]
                e2 = pts[3] - pts[2]
                e3 = pts[0] - pts[3]

                edge_lens = [np.linalg.norm(e0), np.linalg.norm(e1), np.linalg.norm(e2), np.linalg.norm(e3)]
                aspect_ratio = max(edge_lens) / max(1e-4, min(edge_lens))

                
                cross_v = np.cross(e0, -e3)
                det_J_center = np.linalg.norm(cross_v)
                det_J_min = det_J_center
                is_inverted = det_J_min <= 0.0

               
                cos_theta = abs(np.dot(e0, -e3) / max(1e-12, edge_lens[0] * edge_lens[3]))
                max_ortho_error = np.arcsin(np.clip(cos_theta, 0.0, 1.0)) * (180.0 / np.pi)
                skewness = max_ortho_error / 90.0
                cond_num = aspect_ratio * (1.0 + skewness)

                is_pole = (j_idx == 0) or (j_idx == self.num_theta - 2)
                is_equator = abs(j_idx - (self.num_theta // 2)) <= 1
                is_seam = (k_idx == 0) or (k_idx == self.num_phi - 1)

                region_tag = "Interior"
                if is_pole:
                    region_tag = "Pole"
                elif is_equator:
                    region_tag = "Equator"
                if is_seam:
                    region_tag += "+Seam" if region_tag != "Interior" else "Seam"

                center_pos = np.mean(pts, axis=0)

                record = {
                    "elem_id": elem_id,
                    "grid_idx_i_j_k": f"0_{j_idx}_{k_idx}",
                    "region": region_tag,
                    "center_x": center_pos[0],
                    "center_y": center_pos[1],
                    "center_z": center_pos[2],
                    "det_J_min": det_J_min,
                    "det_J_center": det_J_center,
                    "aspect_ratio": aspect_ratio,
                    "skewness": skewness,
                    "cond_number": cond_num,
                    "ortho_err_xi_eta_deg": max_ortho_error,
                    "ortho_err_eta_zeta_deg": 0.0,
                    "ortho_err_zeta_xi_deg": 0.0,
                    "max_ortho_err_deg": max_ortho_error,
                    "is_inverted": is_inverted
                }
                metrics.append(record)
                if is_inverted:
                    inverted_list.append(record)

        return metrics, inverted_list


class Mesh3DVisualizer:

    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.nodes = evaluator.nodes
        self.elements = evaluator.elements
        self.is_3d = evaluator.is_3d

    def render_and_save_3d_mesh(self, inverted_list, save_filename):
        fig = plt.figure(figsize=(10, 8), dpi=300)
        ax = fig.add_subplot(111, projection='3d')

        if self.is_3d:
            num_r, num_theta, num_phi = self.evaluator.grid_shape
            step_r = max(1, num_r // 12)
            step_theta = max(1, num_theta // 6)
            step_phi = max(1, num_phi // 16)

            grid_tensor = self.nodes.reshape((num_r, num_theta, num_phi, 3))

            for i in range(0, num_r, step_r):
                for k in range(0, num_phi, step_phi):
                    ax.plot(grid_tensor[i, :, k, 0], grid_tensor[i, :, k, 1], grid_tensor[i, :, k, 2], 
                            color='navy', alpha=0.25, linewidth=0.5)

            for i in range(0, num_r, step_r):
                for j in range(0, num_theta, step_theta):
                    phi_ring_x = np.append(grid_tensor[i, j, :, 0], grid_tensor[i, j, 0, 0])
                    phi_ring_y = np.append(grid_tensor[i, j, :, 1], grid_tensor[i, j, 0, 1])
                    phi_ring_z = np.append(grid_tensor[i, j, :, 2], grid_tensor[i, j, 0, 2])
                    ax.plot(phi_ring_x, phi_ring_y, phi_ring_z, 
                            color='crimson', alpha=0.15, linewidth=0.4)
        else:
            num_theta, num_phi = self.evaluator.grid_shape
            step_theta = max(1, num_theta // 10)
            step_phi = max(1, num_phi // 16)

            grid_tensor = self.nodes.reshape((num_theta, num_phi, 3))

            for k in range(0, num_phi, step_phi):
                ax.plot(grid_tensor[:, k, 0], grid_tensor[:, k, 1], grid_tensor[:, k, 2], 
                        color='navy', alpha=0.4, linewidth=0.6)

            for j in range(0, num_theta, step_theta):
                phi_ring_x = np.append(grid_tensor[j, :, 0], grid_tensor[j, 0, 0])
                phi_ring_y = np.append(grid_tensor[j, :, 1], grid_tensor[j, 0, 1])
                phi_ring_z = np.append(grid_tensor[j, :, 2], grid_tensor[j, 0, 2])
                ax.plot(phi_ring_x, phi_ring_y, phi_ring_z, 
                        color='crimson', alpha=0.3, linewidth=0.5)

        ax.set_title(f"Mesh Manifold Quality Visualization\n[{self.evaluator.filename}]", fontsize=11, fontweight='bold')
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")

        max_range = np.array([
            self.nodes[:, 0].max() - self.nodes[:, 0].min(),
            self.nodes[:, 1].max() - self.nodes[:, 1].min(),
            self.nodes[:, 2].max() - self.nodes[:, 2].min()
        ]).max() / 2.0

        mid_x = (self.nodes[:, 0].max() + self.nodes[:, 0].min()) * 0.5
        mid_y = (self.nodes[:, 1].max() + self.nodes[:, 1].min()) * 0.5
        mid_z = (self.nodes[:, 2].max() + self.nodes[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        out_path = os.path.join(VIS_DIR, save_filename)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

        print(f"  └─ [Mesh Rendered] Saved visualization plot as '{out_path}'")


# ==============================================================================
#  REPORT GENERATION & CSV PIPELINE
# ==============================================================================

def run_mesh_quality_suite():
    print("=" * 110)
    print("   EXPLICIT MESH QUALITY, INVERTED ELEMENT & MESH VISUALIZATION SUITE")
    print("=" * 110)

    mesh_files = [f for f in os.listdir(MESH_DIR) if f.endswith(".json")]
    if not mesh_files:
        print(f"[❌ ERROR] No JSON mesh files found in '{MESH_DIR}'. Please generate meshes first.")
        return

    summary_rows = []

    for mfile in sorted(mesh_files):
        fpath = os.path.join(MESH_DIR, mfile)
        evaluator = MeshQualityEvaluator(fpath)
        metrics, inverted_list = evaluator.evaluate_mesh()

        base_name = os.path.splitext(mfile)[0]

        csv_raw_path = os.path.join(REPORT_DIR, f"{base_name}_raw_metrics.csv")
        with open(csv_raw_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
            writer.writeheader()
            writer.writerows(metrics)

        csv_inv_path = os.path.join(REPORT_DIR, f"{base_name}_INVERTED_ELEMENTS.csv")
        with open(csv_inv_path, "w", newline="", encoding="utf-8") as f:
            if inverted_list:
                writer = csv.DictWriter(f, fieldnames=inverted_list[0].keys())
                writer.writeheader()
                writer.writerows(inverted_list)
            else:
                f.write("No inverted elements detected in this mesh. Topology is perfectly healthy.\n")

        visualizer = Mesh3DVisualizer(evaluator)
        vis_png_name = f"{base_name}_manifold.png"
        visualizer.render_and_save_3d_mesh(inverted_list, vis_png_name)

        poles_metrics = [m for m in metrics if "Pole" in m["region"]]
        equator_metrics = [m for m in metrics if "Equator" in m["region"]]
        seam_metrics = [m for m in metrics if "Seam" in m["region"]]

        def max_metric(arr, key):
            return max([m[key] for m in arr]) if arr else 0.0

        total_elems = len(metrics)
        num_inv = len(inverted_list)
        inv_ratio = (num_inv / total_elems) * 100.0

        summary = {
            "mesh_file": mfile,
            "total_elems": total_elems,
            "inverted_count": num_inv,
            "inverted_ratio_pct": inv_ratio,
            "max_aspect_ratio": max_metric(metrics, "aspect_ratio"),
            "max_skewness": max_metric(metrics, "skewness"),
            "max_cond_number": max_metric(metrics, "cond_number"),
            "max_ortho_err_deg": max_metric(metrics, "max_ortho_err_deg"),
            "pole_max_distortion": max_metric(poles_metrics, "max_ortho_err_deg"),
            "equator_max_distortion": max_metric(equator_metrics, "max_ortho_err_deg"),
            "seam_max_distortion": max_metric(seam_metrics, "max_ortho_err_deg")
        }
        summary_rows.append(summary)

        print(f" Analyzed: {mfile:<28} | Inv Elems: {num_inv:4d} ({inv_ratio:5.2f}%) | "
              f"Max AspectRatio: {summary['max_aspect_ratio']:6.2f} | Max Cond#: {summary['max_cond_number']:7.1f}")

    summary_csv_path = os.path.join(REPORT_DIR, "MESH_QUALITY_COMPARISON_SUMMARY.csv")
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n[📊 REPORTS EXPORTED] Detailed CSV reports and plots generated in directory: '{REPORT_DIR}/'")

    plot_quality_comparison_charts(summary_rows)


def plot_quality_comparison_charts(summary_rows):
    tcr_rows = [r for r in summary_rows if r["mesh_file"].startswith("tcr")]
    pde_rows = [r for r in summary_rows if r["mesh_file"].startswith("pde")]

    levels = [f"L{i+1}" for i in range(len(tcr_rows))]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)

    axes[0, 0].plot(levels, [r["inverted_ratio_pct"] for r in tcr_rows], 'o-', color='navy', label='Case A (TCR)')
    if pde_rows:
        axes[0, 0].plot(levels[:len(pde_rows)], [r["inverted_ratio_pct"] for r in pde_rows], 's--', color='crimson', label='Case B (PDE)')
    axes[0, 0].set_title("Inverted Element Ratio (%)", fontweight='bold')
    axes[0, 0].set_ylabel("Inverted Ratio (%)")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend()

    axes[0, 1].plot(levels, [r["max_cond_number"] for r in tcr_rows], 'o-', color='navy', label='Case A (TCR)')
    if pde_rows:
        axes[0, 1].plot(levels[:len(pde_rows)], [r["max_cond_number"] for r in pde_rows], 's--', color='crimson', label='Case B (PDE)')
    axes[0, 1].set_title("Max Element Condition Number", fontweight='bold')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_ylabel("Condition Number (Log Scale)")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    axes[0, 1].legend()

    axes[1, 0].plot(levels, [r["max_ortho_err_deg"] for r in tcr_rows], 'o-', color='navy', label='Case A (TCR)')
    if pde_rows:
        axes[1, 0].plot(levels[:len(pde_rows)], [r["max_ortho_err_deg"] for r in pde_rows], 's--', color='crimson', label='Case B (PDE)')
    axes[1, 0].set_title("Max Orthogonality Error (Degrees)", fontweight='bold')
    axes[1, 0].set_ylabel("Max Angle Error (°)")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    axes[1, 0].legend()

    if tcr_rows:
        last_tcr = tcr_rows[-1]
        regions = ['Poles', 'Equator', 'Seams']
        distortions = [
            last_tcr["pole_max_distortion"],
            last_tcr["equator_max_distortion"],
            last_tcr["seam_max_distortion"]
        ]
        axes[1, 1].bar(regions, distortions, color=['darkred', 'darkgreen', 'darkblue'], alpha=0.75)
        axes[1, 1].set_title("TCR Regional Max Distortion (Max Mesh Level)", fontweight='bold')
        axes[1, 1].set_ylabel("Max Orthogonality Error (°)")
        axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Mesh Quality, Topology Degradation & Regional Distortion Diagnostics", fontsize=13, fontweight='bold')
    plt.tight_layout()

    plot_path = os.path.join(REPORT_DIR, "mesh_quality_comparison_suite.png")
    plt.savefig(plot_path, dpi=300)
    print(f"[📊 PLOT SAVED] Quality comparison plot saved as '{plot_path}'\n")
    plt.show()


if __name__ == "__main__":
    run_mesh_quality_suite()
