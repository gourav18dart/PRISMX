"""
Master Execution Pipeline for PRISM Reproduction & Novel Extensions
=====================================================================
Executes all 3 reproduction experiments and 2 novel improvements in sequence,
generating all publication-ready figures in outputs/.
"""

import sys
import subprocess
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    ("Experiment 1: Faithfulness via Representation Swapping", "experiments/exp1_faithfulness.py"),
    ("Experiment 2: Interpretability Insight with Exact Spiders", "experiments/exp2_interpretability.py"),
    ("Experiment 3: Variance Distribution on 2-Class & Exact 5-Class Sets", "experiments/exp3_variance_analysis.py"),
    ("Improvement 1: Resolving 3-PCA Bottleneck (Spectral & High-Variance PRISM)", "improvements/imp1_resolve_3pca_bottleneck.py"),
    ("Improvement 2: Resolving Class-Agnostic Distractors (Grad-PRISM & cPRISM)", "improvements/imp2_class_attentive_grad_prism.py"),
]

def main():
    print("=" * 80)
    print("PRISM REPRODUCTION & RESEARCH EXTENSIONS: MASTER RUNNER")
    print("=" * 80)
    
    python_exe = sys.executable
    print(f"Using Python Environment: {python_exe}\n")
    
    for idx, (title, rel_path) in enumerate(SCRIPTS, 1):
        print("\n" + "=" * 80)
        print(f"[{idx}/{len(SCRIPTS)}] RUNNING: {title}")
        print(f"Script: {rel_path}")
        print("=" * 80)
        
        script_full_path = os.path.join(BASE_DIR, rel_path)
        result = subprocess.run([python_exe, script_full_path], cwd=BASE_DIR)
        
        if result.returncode != 0:
            print(f"[ERROR] Script {rel_path} failed with exit code {result.returncode}")
        else:
            print(f"[COMPLETED] {title} successfully executed.")
            
    print("\n" + "=" * 80)
    print("[ALL TASKS FINISHED] All figures and results have been saved to outputs/")
    print("=" * 80)

if __name__ == "__main__":
    main()
