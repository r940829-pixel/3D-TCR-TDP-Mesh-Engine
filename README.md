# 3D TCR & TDP: Time-Coherent Resonance Manifold Mesh Generation Engine

[![License: Academic Use](https://img.shields.io/badge/License-Academic_Research-blue.svg)]()
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)]()
[![DOI](https://zenodo.org/badge/1291166293.svg)](https://doi.org/10.5281/zenodo.21224235)

An advanced, high-fidelity boundary-conformal mesh pre-processing engine developed for advanced computational physics and multi-orbital electron kinematics simulations. This repository implements the **Three-Dimensional Time-Coherent Resonance (3D TCR)** analytical manifold mapping system and the hybrid **TCR-Driven Poisson Elliptic PDE (TDP)** framework.

---

## 🏛️ Theoretical Foundation

Traditional elliptic grid generation techniques (e.g., homogeneous Laplace solvers) suffer from severe geometric distortion and numerical dissipation near singular boundary points. This framework resolves these limitations by implementing a closed-form analytical tensor projection. 

The core 3D TCR transformation maps parametric computational space $(\xi, \eta, \zeta)$ to physical space $(x, y, z)$ via a modulated absolute tensor operator, strictly guaranteeing a real-valued metric mapping with an exact Jacobian determinant:

$$|J| = \frac{n(\theta)}{2} \csc(2\theta)$$

Where $n(\theta) = \frac{1}{2}r^2|\sin(2\theta)|$. At the base harmonic state, the system naturally degenerates into standard Cartesian spherical mapping ($O(1)$ complexity), while automatically enforcing structural logarithmic-like grid contraction tunnels along the polar singularity axes without iterative relaxation costs.

---

## 📊 Benchmarking & Experimental Verification

The architecture evaluates three explicit comparative cases across an ablation study pipeline:
- **Case A: Pure Analytical 3D TCR Manifold** ($O(1)$ analytic closed-form)
- **Case B: TCR-Driven Poisson Elliptic PDE (TDP)** ($O(N^3)$ hybrid field tracking)
- **Case C: Pure Non-Source Laplace Elliptic PDE** ($O(N^3)$ classical homogeneous baseline)

### High-Impact Quantization Metrics
The integrated evaluation suite computes relativistic electron flight paths using a 4th-Order Runge-Kutta (RK4) Lorentz kinematics solver directly over the real generated grid matrices. True numerical dissipation and trajectory drift are extracted by measuring Euclidean variances against continuous field ground truths.

---

## 🛠️ Reproducibility Pipeline (Step-by-Step Instructions)

To guarantee exact numerical replication of the publication-grade metrics and visual data plots, follow the structured execution sequence below.

---

### 1. Environment Setup
Ensure you have Python 3.8+ deployed. Install the standardized scientific computing core stack:
```bash
pip install numpy matplotlib
```

### 2. Execution Sequence
Run the core mesh generation engines in order to populate the serialized cache databases (*.json):
Generate Case A: Pure Algebraic Analytical Grid Matrix (Outputs: tcr_mesh_journal.json)
```bash
python tcr_core_engine.py
```

Generate Case C: Homogeneous Laplace Baseline Grid Matrix (Outputs: pde_mesh_journal.json)
```bash
python pde_core_engine.py
```

Generate Case B: Complete 3D Tensor Source Modulated Grid Matrix (Outputs: tdp_mesh_journal.json)
```bash
python tdp_core_engine.py
```

### 3. Quantitative Evaluation & Trajectory Visualization
Execute the dynamic physics conformal solver. This script reads the real grid node coordinates, executes grid-interpolated RK4 kinematics, outputs verification text summaries, and generates publication plots:
```bash
python physics_evaluator.py
```

Expected Text Artifacts Generated: tcr_final.txt, tdp_final.txt, pde_final.txt

Expected Visual Figure Generated: journal_trajectory_comparison.png (Featuring localized high-density micro-inset zoom subplots at origin (0,0)).

### 4. Empirical Complexity & Scaling Test
To verify the theoretical $O(1)$ efficiency of the analytical manifold mapping against the iterative $O(N^3)$ stencils, run the automated resolution scalability suite:
```bash
python scalability_time_benchmark.py
```

Expected Visual Figure Generated: mesh_scalability_benchmark.png (Log-log resolution curve verifying microsecond flat execution overhead for Case A).

---

```test
📈 Expected Replicated Performance Profile
Upon pipeline completion, your exported log documents will reflect the following quantitative trends:
Geometric Integrity (Metric 1): Case A maintains optimal grid orthogonality scores due to its analytical foundations. Case B (TDP) shows balanced orthogonality optimization, effectively preventing the complete structural distortion seen in Case C (~12.29 score collapse).
Kinematic Conformal Drift (Metric 2): Case C exhibits prominent outward numerical trajectory drift under core microinsets due to localized field grid dilution. Case A tracks sub-pixel relativistic orbits with minimized absolute errors.
Computational Complexity (Metric 3): The log-log benchmark figure explicitly mirrors a constant horizontal microsecond latency boundary ($O(1)$) for TCR, contrasted against the exponential cubic workload curve ($O(N^3)$) governing traditional SOR solvers.
```

---

```test
📝 Citation & Academic Contact
This software suite is part of an active academic research framework on geometric topology and classical emulation fields.
Author: Zhuang Huaijie
Affiliation: Department of Information Engineering, I-Shou University (ISU)
License: Academic Research Use Only. For licensing or collaborative verification queries, please open an issue tracking request via GitHub Codespaces.
```

---
