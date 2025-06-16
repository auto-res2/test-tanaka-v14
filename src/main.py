#!/usr/bin/env python
"""
Main experimental script for ACIP (Adaptive Confidence-Integrated Purification).

This script implements and evaluates a novel diffusion purification method that extends 
Purify++ by dynamically integrating classifier confidence feedback into the reverse 
diffusion process.

Experiments:
1. Robustness evaluation (ACIP vs baseline on adversarial examples)
2. Ablation study (contribution of adaptive guidance and dynamic noise scheduling)  
3. Efficiency analysis (computational cost and convergence curves)

All plots are saved as high-quality PDFs in .research/iteration1/images/
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless execution

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluate import evaluate_all_experiments

def setup_environment():
    """
    Setup experimental environment and verify GPU availability.
    """
    print("ACIP Experimental Setup")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("Running on CPU (GPU not available)")
    
    output_dir = ".research/iteration1/images"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    print("Environment setup complete")
    print("=" * 60)
    
    return device

def run_quick_test():
    """
    Run a quick test to verify the implementation works correctly.
    Uses minimal data and timesteps for fast execution.
    """
    print("Running Quick Test...")
    print("-" * 40)
    
    try:
        from train import train_models, acip_reverse_sample
        from preprocess import preprocess_data_for_experiment
        
        print("✓ All imports successful")
        
        classifier, diffusion_model = train_models()
        print("✓ Models initialized successfully")
        
        data_loader, _ = preprocess_data_for_experiment(max_batches=1)
        print("✓ Data loading successful")
        
        for data, target in data_loader:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            data, target = data.to(device), target.to(device)
            
            purified = acip_reverse_sample(
                diffusion_model, classifier, data, timesteps=3, target_label=target,
                adaptive_weight=0.1
            )
            print("✓ ACIP purification test successful")
            break
        
        print("✓ Quick test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Quick test failed: {str(e)}")
        return False

def main():
    """
    Main experimental pipeline for ACIP evaluation.
    """
    print("ADAPTIVE CONFIDENCE-INTEGRATED PURIFICATION (ACIP)")
    print("Experimental Evaluation Script")
    print("=" * 80)
    
    device = setup_environment()
    
    if not run_quick_test():
        print("Quick test failed. Aborting full experiments.")
        return
    
    print("\nStarting Full Experimental Evaluation...")
    print("=" * 80)
    
    try:
        results = evaluate_all_experiments()
        
        print("\n" + "=" * 80)
        print("EXPERIMENTAL RESULTS SUMMARY")
        print("=" * 80)
        
        exp1 = results['experiment1']
        exp3 = results['experiment3']
        
        print(f"🎯 ROBUSTNESS IMPROVEMENT:")
        print(f"   ACIP Accuracy: {exp1['acip_accuracy']:.3f}")
        print(f"   Baseline Accuracy: {exp1['baseline_accuracy']:.3f}")
        print(f"   Improvement: +{exp1['improvement']:.3f}")
        
        print(f"\n⚡ EFFICIENCY ANALYSIS:")
        print(f"   ACIP: {exp3['acip_steps']:.1f} steps, {exp3['acip_time']:.3f}s")
        print(f"   Baseline: {exp3['baseline_steps']:.1f} steps, {exp3['baseline_time']:.3f}s")
        
        print(f"\n📊 OUTPUTS GENERATED:")
        output_dir = ".research/iteration1/images"
        pdf_files = [f for f in os.listdir(output_dir) if f.endswith('.pdf')]
        for pdf_file in pdf_files:
            print(f"   ✓ {pdf_file}")
        
        print(f"\n🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
        print(f"   Total PDF plots generated: {len(pdf_files)}")
        print(f"   All outputs saved to: {output_dir}")
        
        status_enum = "stopped"
        print(f"\n📋 STATUS: {status_enum}")
        
    except Exception as e:
        print(f"\n❌ EXPERIMENT FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        
        status_enum = "stopped"
        print(f"\n📋 STATUS: {status_enum}")

if __name__ == "__main__":
    main()
