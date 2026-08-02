import os
import sys
import time
import json
import platform
import psutil
import cpuinfo
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


try:
    import tcr_core_engine
except ImportError:
    tcr_core_engine = None

try:
    import pde_core_engine
except ImportError:
    pde_core_engine = None


rcParams['font.family'] = 'serif'
rcParams['mathtext.fontset'] = 'cm'
rcParams['axes.linewidth'] = 1.0
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

BENCHMARK_DIR = "benchmark_results"
os.makedirs(BENCHMARK_DIR, exist_ok=True)


# ==============================================================================
#  1. SYSTEM ENVIRONMENT DIAGNOSTICS
# ==============================================================================

def get_system_environment_info():
    
    try:
        cpu_name = cpuinfo.get_cpu_info().get('brand_raw', platform.processor())
    except Exception:
        cpu_name = platform.processor() or "Unknown CPU"

    ram_bytes = psutil.virtual_memory().total
    ram_gb = ram_bytes / (1024 ** 3)

    env_info = {
        "operating_system": f"{platform.system()} {platform.release()} ({platform.architecture()[0]})",
        "os_version": platform.version(),
        "python_version": sys.version.split()[0],
        "python_compiler": platform.python_compiler(),
        "cpu_architecture": platform.machine(),
        "cpu_model": cpu_name.strip(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "total_ram_gb": round(ram_gb, 2)
    }

    return env_info


def save_environment_report(env_info, filepath):
  
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=4)
    
    print("\n" + "=" * 80)
    print("   SYSTEM & HARDWARE ENVIRONMENT DIAGNOSTICS")
    print("=" * 80)
    for k, v in env_info.items():
        print(f"  • {k:<22}: {v}")
    print("=" * 80 + "\n")


# ==============================================================================
#  2. MESH GENERATION EXECUTORS CALLING TRUE PIPELINE ENGINES
# ==============================================================================

def measure_tcr_pipeline_execution_cost(num_theta, num_phi, output_path):
    
    if tcr_core_engine is None or not hasattr(tcr_core_engine, 'execute_tcr_manifold_engine'):
        raise RuntimeError(
            "[❌ CRITICAL ERROR] Pipeline module 'tcr_core_engine.py' with 'execute_tcr_manifold_engine' "
            "was not found in the Python execution path!"
        )

    t_start = time.perf_counter()

    sys_eqs, sys_config = tcr_core_engine.get_coupled_system_specification()
    config = {
        "num_theta_pts": num_theta,
        "num_phi": num_phi,
        "n_val": 1.0,
        "output_filename": output_path
    }
    
    mesh_data = tcr_core_engine.execute_tcr_manifold_engine(sys_eqs, config)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mesh_data, f)

    t_end = time.perf_counter()
    return t_end - t_start


def measure_pde_pipeline_execution_cost(num_r, num_theta, num_phi, output_path):
    
    if pde_core_engine is None or not hasattr(pde_core_engine, 'generate_pure_pde_3d_mesh'):
        raise RuntimeError(
            "[❌ CRITICAL ERROR] Pipeline module 'pde_core_engine.py' with 'generate_pure_pde_3d_mesh' "
            "was not found in the Python execution path!"
        )

    t_start = time.perf_counter()

    pde_core_engine.generate_pure_pde_3d_mesh(
        r_val=5.0,
        num_r=num_r,
        num_theta=num_theta,
        num_phi=num_phi,
        max_iter=600,
        tol=1e-5,
        filename=output_path
    )

    t_end = time.perf_counter()
    return t_end - t_start


# ==============================================================================
#  3. DYNAMIC COMPLEXITY FITTER
# ==============================================================================

def calculate_empirical_complexity_slope(N_array, time_array):
    
    log_N = np.log10(N_array)
    log_T = np.log10(time_array)

    slope, intercept = np.polyfit(log_N, log_T, 1)
    return slope, intercept


# ==============================================================================
#  4. MAIN BENCHMARK RUNNER & REPORT GENERATOR
# ==============================================================================

def run_scalability_benchmark():
    print("=" * 80)
    print("   SCALABILITY BENCHMARK WITH TRUE PIPELINE & DYNAMIC COMPLEXITY FIT")
    print("=" * 80)

    if tcr_core_engine is None:
        raise ImportError("[❌ ERROR] Cannot find 'tcr_core_engine.py' in current directory!")
    if pde_core_engine is None:
        raise ImportError("[❌ ERROR] Cannot find 'pde_core_engine.py' in current directory!")

    env_info = get_system_environment_info()
    env_json_path = os.path.join(BENCHMARK_DIR, "system_environment_info.json")
    save_environment_report(env_info, env_json_path)

    mesh_levels = [
        {"level": "L1", "shape": (16, 10, 20)},
        {"level": "L2", "shape": (24, 15, 30)},
        {"level": "L3", "shape": (32, 20, 40)},
        {"level": "L4", "shape": (48, 30, 60)},
        {"level": "L5", "shape": (60, 40, 80)},
        {"level": "L6", "shape": (72, 50, 100)},
        {"level": "L7", "shape": (80, 60, 120)}
    ]

    benchmark_records = []
    tcr_times = []
    pde_times = []
    tcr_node_counts = []
    pde_node_counts = []

    print(f"{'Level':<6} | {'Grid Shape (th,ph / r,th,ph)':<30} | {'TCR N':<10} | {'PDE N':<10} | {'TCR Time (s)':<12} | {'PDE Time (s)':<12}")
    print("-" * 95)

    for lvl_info in mesh_levels:
        lvl = lvl_info["level"]
        nr, nth, nph = lvl_info["shape"]
        
        tcr_nodes = nth * nph
        pde_nodes = nr * nth * nph

        tcr_json = os.path.join(BENCHMARK_DIR, f"temp_tcr_{lvl}.json")
        pde_json = os.path.join(BENCHMARK_DIR, f"temp_pde_{lvl}.json")

        num_repeats = 3
        tcr_t_list = [measure_tcr_pipeline_execution_cost(nth, nph, tcr_json) for _ in range(num_repeats)]
        pde_t_list = [measure_pde_pipeline_execution_cost(nr, nth, nph, pde_json) for _ in range(num_repeats)]

        tcr_t_avg = np.mean(tcr_t_list)
        pde_t_avg = np.mean(pde_t_list)

        tcr_times.append(tcr_t_avg)
        pde_times.append(pde_t_avg)
        tcr_node_counts.append(tcr_nodes)
        pde_node_counts.append(pde_nodes)

        if os.path.exists(tcr_json): os.remove(tcr_json)
        if os.path.exists(pde_json): os.remove(pde_json)

        shape_str = f"({nth}x{nph} / {nr}x{nth}x{nph})"
        print(f"{lvl:<6} | {shape_str:<30} | {tcr_nodes:<10,d} | {pde_nodes:<10,d} | {tcr_t_avg:<12.4f} | {pde_t_avg:<12.4f}")

        benchmark_records.append({
            "level": lvl,
            "num_r_pde": nr,
            "num_theta": nth,
            "num_phi": nph,
            "tcr_nodes_N": tcr_nodes,
            "pde_nodes_N": pde_nodes,
            "tcr_runtime_sec": round(tcr_t_avg, 6),
            "pde_runtime_sec": round(pde_t_avg, 6),
            "time_ratio_tcr_vs_pde": round(tcr_t_avg / max(1e-6, pde_t_avg), 4)
        })

    raw_csv_path = os.path.join(BENCHMARK_DIR, "scalability_time_benchmark_raw.csv")
    with open(raw_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=benchmark_records[0].keys())
        writer.writeheader()
        writer.writerows(benchmark_records)
    print(f"\n[📊 CSV EXPORTED] Saved raw timing benchmark data to '{raw_csv_path}'")

    tcr_nodes_arr = np.array(tcr_node_counts, dtype=np.float64)
    pde_nodes_arr = np.array(pde_node_counts, dtype=np.float64)
    tcr_times_arr = np.array(tcr_times, dtype=np.float64)
    pde_times_arr = np.array(pde_times, dtype=np.float64)

    slope_tcr, _ = calculate_empirical_complexity_slope(tcr_nodes_arr, tcr_times_arr)
    slope_pde, _ = calculate_empirical_complexity_slope(pde_nodes_arr, pde_times_arr)

    print("\n" + "=" * 80)
    print("   DYNAMIC EMPIRICAL COMPLEXITY ANALYSIS (SLOPE FIT)")
    print("=" * 80)
    print(f"  • TCR Empirical Complexity Slope p = {slope_tcr:.4f}  => Dynamic O(N^{slope_tcr:.2f})")
    print(f"  • PDE Empirical Complexity Slope p = {slope_pde:.4f}  => Dynamic O(N^{slope_pde:.2f})")
    print("=" * 80 + "\n")

    plot_loglog_scalability_chart(tcr_nodes_arr, pde_nodes_arr, tcr_times_arr, pde_times_arr, 
                                 slope_tcr, slope_pde, env_info)


# ==============================================================================
#  5. LOG-LOG RUNTIME PLOTTING & COMPLEXITY ANNOTATION
# ==============================================================================

def plot_loglog_scalability_chart(tcr_nodes, pde_nodes, tcr_t, pde_t, slope_tcr, slope_pde, env_info):
    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=300)

    ax.loglog(tcr_nodes, tcr_t, 'o-', color='navy', linewidth=2.0, markersize=7, 
              label=f'Case A (Pure TCR) - Dynamic $\mathcal{{O}}(N^{{{slope_tcr:.2f}}})$')
    ax.loglog(pde_nodes, pde_t, 's--', color='crimson', linewidth=1.8, markersize=6, 
              label=f'Case B (PDE Baseline) - Dynamic $\mathcal{{O}}(N^{{{slope_pde:.2f}}})$')

    ref_line = (tcr_nodes / tcr_nodes[0]) * tcr_t[0]
    ax.loglog(tcr_nodes, ref_line, 'k:', alpha=0.5, linewidth=1.2, label=r'Theoretical Reference $\mathcal{O}(N^1)$')

    ax.text(tcr_nodes[-2] * 0.80, tcr_t[-2] * 1.35, f'Slope $p = {slope_tcr:.2f}$', 
            color='navy', fontweight='bold', fontsize=10)
    ax.text(pde_nodes[-2] * 0.80, pde_t[-2] * 0.65, f'Slope $p = {slope_pde:.2f}$', 
            color='crimson', fontweight='bold', fontsize=10)

    ax.set_title("Mesh Scaling & Execution Time Benchmark\n[Log-Log Dynamic Complexity Analysis]", 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel("Total Mesh Nodes / Degrees of Freedom ($N$)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Total End-to-End Execution Time [s] (Log Scale)", fontsize=11, fontweight='bold')
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

    sys_caption = f"CPU: {env_info['cpu_model']} ({env_info['cpu_physical_cores']} Cores) | RAM: {env_info['total_ram_gb']} GB | OS: {env_info['operating_system']} | Python {env_info['python_version']}"
    plt.figtext(0.5, 0.01, sys_caption, ha="center", fontsize=8, color="gray", style="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    
    chart_path = os.path.join(BENCHMARK_DIR, "scalability_loglog_runtime_chart.png")
    plt.savefig(chart_path, dpi=300)
    print(f"[📊 PLOT SAVED] Log-log runtime chart saved to '{chart_path}'\n")
    plt.show()


if __name__ == "__main__":
    run_scalability_benchmark()
