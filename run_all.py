"""
Master Execution Pipeline for PRISMX Reproduction & Novel Extensions
=====================================================================
Executes all reproduction experiments and novel improvement benchmarks
in sequence, generating all publication-ready figures in outputs/.
"""

import sys
import subprocess
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BENCHMARKS = [
    ("Automated Unit Test Suite (Core PRISM, Gating & SVD Conservation)", "tests/run_tests.py"),
    ("Experiment 1: Faithfulness via Representation Swapping on Canines (Fig. 4)", "experiments/exp1_faithfulness.py"),
    ("Experiment 2: Interpretability Insight on Spiders (Fig. 6)", "experiments/exp2_interpretability.py"),
    ("Experiment 2b: Interpretability Insight on Trains (Fig. 7)", "experiments/exp2_trains_interpretability.py"),
    ("Experiment 3: Variance Distribution on 2-Class & Exact 5-Class Sets (Fig. 16/17)", "experiments/exp3_variance_analysis.py"),
    ("Experiment 4: Faithfulness Swapping on Food & Trains (Fig. 3 & Fig. 5)", "experiments/exp4_multiclass_faithfulness.py"),
    ("Improvement 1: Resolving 3-PCA Bottleneck (Spectral Bands & High-Variance PRISMX)", "improvements/imp1_resolve_3pca_bottleneck.py"),
    ("Improvement 2: Resolving Distractors (Grad-PRISM & Contrastive cPRISM)", "improvements/imp2_class_attentive_grad_prism.py"),
    ("Evaluation 1: Side-by-Side 3-Case Fidelity Breakdown", "experiments/eval_fidelity_comparison.py"),
    ("Evaluation 2: Unveiling Higher PCs Residual Energy Heatmap", "experiments/eval_wolf_difference_revealed.py"),
    ("Evaluation 3: Resolving Distractors on Malamute + Mushroom (Fig. 12)", "experiments/eval_distractor_malamute_mushroom.py"),
    ("Evaluation 4: Multi-Domain Validations (Vehicles, Food, Insects, Multi-Animal, Street Clutter)", "experiments/eval_multidomain_improvements.py"),
]

def main():
    print("=" * 85)
    print("PRISMX REPRODUCTION & RESEARCH EXTENSIONS: MASTER RUNNER")
    print("=" * 85)
    
    python_exe = sys.executable
    print(f"Active Python Environment: {python_exe}\n")
    
    for idx, (title, rel_path) in enumerate(BENCHMARKS, 1):
        print("\n" + "=" * 85)
        print(f"[{idx}/{len(BENCHMARKS)}] RUNNING: {title}")
        print(f"Script: {rel_path}")
        print("=" * 85)
        
        script_full_path = os.path.join(BASE_DIR, rel_path)
        result = subprocess.run([python_exe, script_full_path], cwd=BASE_DIR)
        
        if result.returncode != 0:
            print(f"[ERROR] Script {rel_path} failed with exit code {result.returncode}")
        else:
            print(f"[COMPLETED] {title} successfully executed.")
            
    print("\n" + "=" * 85)
    print("[ALL TASKS FINISHED] All publication-ready figures successfully saved to outputs/")
    print("=" * 85)

if __name__ == "__main__":
    main()
