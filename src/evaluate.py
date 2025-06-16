#!/usr/bin/env python
"""
Evaluation functions for ACIP experiments.
Implements three main experiments: robustness, ablation, and efficiency analysis.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from train import (
    acip_reverse_sample, baseline_reverse_sample, acip_variant_sample, 
    convergence_sample, train_models
)
from preprocess import fgsm_attack, add_diffusion_noise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_purification_robustness(acip=True, epsilon=0.05, max_batches=2, timesteps=10):
    """
    Experiment 1: Evaluate robust classification accuracy using ACIP vs baseline.
    """
    from preprocess import preprocess_data_for_experiment
    
    data_loader, _ = preprocess_data_for_experiment(max_batches=max_batches, epsilon=epsilon)
    classifier, diffusion_model = train_models()
    
    robust_correct = 0
    total = 0
    batch_count = 0
    
    for data, target in data_loader:
        if batch_count >= max_batches:
            break
        
        data, target = data.to(device), target.to(device)
        
        adv_data = fgsm_attack(classifier, data, target, epsilon)
        noisy_adv = add_diffusion_noise(adv_data, noise_factor=0.1)
        
        if acip:
            purified = acip_reverse_sample(
                diffusion_model, classifier, noisy_adv, timesteps, target,
                adaptive_weight=0.5, 
                dynamic_schedule_fn=lambda logits, t: 0.5
            )
        else:
            purified = baseline_reverse_sample(diffusion_model, noisy_adv, timesteps)
        
        output = classifier(purified)
        pred = output.argmax(dim=1)
        robust_correct += (pred == target).sum().item()
        total += target.size(0)
        batch_count += 1
    
    robust_acc = robust_correct / total if total > 0 else 0.0
    return robust_acc

def run_experiment1():
    """
    Experiment 1: Compare ACIP vs baseline on classification robustness.
    """
    print("=" * 60)
    print("EXPERIMENT 1: Robustness Evaluation")
    print("=" * 60)
    
    robust_acc_acip = evaluate_purification_robustness(acip=True, epsilon=0.05)
    robust_acc_baseline = evaluate_purification_robustness(acip=False, epsilon=0.05)
    
    print(f"Robust Accuracy with ACIP: {robust_acc_acip:.3f}")
    print(f"Robust Accuracy with Baseline: {robust_acc_baseline:.3f}")
    print(f"Improvement: {robust_acc_acip - robust_acc_baseline:.3f}")
    
    return {
        'acip_accuracy': robust_acc_acip,
        'baseline_accuracy': robust_acc_baseline,
        'improvement': robust_acc_acip - robust_acc_baseline
    }

def run_ablation_experiment(variant_params, max_batches=2, timesteps=10):
    """
    Run ablation study variant and return metrics.
    """
    from preprocess import preprocess_data_for_experiment
    
    data_loader, _ = preprocess_data_for_experiment(max_batches=max_batches)
    classifier, diffusion_model = train_models()
    
    step_tracker = []
    robust_correct = 0
    total = 0
    confidence_history = []
    batch_count = 0
    
    for data, target in data_loader:
        if batch_count >= max_batches:
            break
        
        data, target = data.to(device), target.to(device)
        adv_data = fgsm_attack(classifier, data, target, epsilon=0.05)
        noisy_adv = add_diffusion_noise(adv_data, noise_factor=0.1)
        
        purified, confidences = acip_variant_sample(
            diffusion_model, classifier, noisy_adv, timesteps, target, **variant_params
        )
        confidence_history.append(confidences)
        
        for i, conf in enumerate(confidences):
            if conf > 0.90:
                step_tracker.append(timesteps - i)
                break
        
        output = classifier(purified)
        pred = output.argmax(dim=1)
        robust_correct += (pred == target).sum().item()
        total += target.size(0)
        batch_count += 1
    
    avg_steps = np.mean(step_tracker) if len(step_tracker) > 0 else timesteps
    robust_accuracy = robust_correct / total if total > 0 else 0.0
    
    return robust_accuracy, avg_steps, confidence_history

def run_experiment2():
    """
    Experiment 2: Ablation study of key ACIP innovations.
    """
    print("=" * 60)
    print("EXPERIMENT 2: Ablation Study")
    print("=" * 60)
    
    variants = {
        'adaptive_only': {
            'use_guidance': True, 
            'use_dynamic_schedule': False, 
            'adaptive_weight': 0.5
        },
        'dynamic_only': {
            'use_guidance': False, 
            'use_dynamic_schedule': True, 
            'adaptive_weight': 0.0,
            'dynamic_schedule_fn': lambda logits, t: 0.5 * (1 - F.softmax(logits, dim=1).max(dim=1)[0].mean().item())
        },
        'full_acip': {
            'use_guidance': True, 
            'use_dynamic_schedule': True, 
            'adaptive_weight': 0.5,
            'dynamic_schedule_fn': lambda logits, t: 0.5 * (1 - F.softmax(logits, dim=1).max(dim=1)[0].mean().item())
        },
        'baseline': {
            'use_guidance': False, 
            'use_dynamic_schedule': False, 
            'adaptive_weight': 0.0
        }
    }
    
    results = {}
    all_confidence_histories = {}
    
    for name, params in variants.items():
        print(f"Running variant: {name}")
        robust_acc, avg_steps, conf_history = run_ablation_experiment(params)
        results[name] = {
            'robust_acc': robust_acc, 
            'avg_steps': avg_steps
        }
        all_confidence_histories[name] = conf_history
        
        print(f"  Robust Accuracy: {robust_acc:.3f}")
        print(f"  Avg. Steps until confidence threshold: {avg_steps:.1f}")
    
    plt.figure(figsize=(10, 6))
    for name, conf_history in all_confidence_histories.items():
        if len(conf_history) > 0:
            plt.plot(conf_history[0], label=name, linewidth=2)
    
    plt.xlabel("Diffusion Iteration", fontsize=12)
    plt.ylabel("Avg. Classifier Confidence", fontsize=12)
    plt.title("Confidence Evolution per Diffusion Iteration", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = ".research/iteration1/images/confidence_evolution_ablation.pdf"
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confidence evolution plot saved: {output_path}")
    return results

def run_efficiency_experiment(method='acip', max_batches=2, timesteps=10):
    """
    Run efficiency experiment to measure computational cost and convergence.
    """
    from preprocess import preprocess_data_for_experiment
    
    data_loader, _ = preprocess_data_for_experiment(max_batches=max_batches)
    classifier, diffusion_model = train_models()
    
    total_steps = 0
    total_time = 0.0
    conf_curves = []
    batch_count = 0
    
    adaptive_fn = lambda logits, t: 0.5 * (1 - F.softmax(logits, dim=1).max(dim=1)[0].mean().item())
    
    for data, target in data_loader:
        if batch_count >= max_batches:
            break
        
        data, target = data.to(device), target.to(device)
        adv_data = fgsm_attack(classifier, data, target, epsilon=0.05)
        noisy_adv = add_diffusion_noise(adv_data, noise_factor=0.1)
        
        _, confidences, steps_taken, elapsed_time = convergence_sample(
            diffusion_model, classifier, noisy_adv, timesteps, target,
            method=method, adaptive_weight=0.5,
            dynamic_schedule_fn=adaptive_fn, conf_threshold=0.90
        )
        
        total_steps += steps_taken
        total_time += elapsed_time
        conf_curves.append(confidences)
        batch_count += 1
    
    avg_steps = total_steps / batch_count if batch_count > 0 else timesteps
    avg_time = total_time / batch_count if batch_count > 0 else 0.0
    
    return avg_steps, avg_time, conf_curves

def run_experiment3():
    """
    Experiment 3: Computational efficiency and convergence analysis.
    """
    print("=" * 60)
    print("EXPERIMENT 3: Efficiency and Convergence Analysis")
    print("=" * 60)
    
    print("Running ACIP efficiency test...")
    avg_steps_acip, avg_time_acip, conf_curves_acip = run_efficiency_experiment(method='acip')
    
    print("Running baseline efficiency test...")
    avg_steps_baseline, avg_time_baseline, conf_curves_baseline = run_efficiency_experiment(method='baseline')
    
    print(f"ACIP - Avg Steps: {avg_steps_acip:.1f}, Avg Time: {avg_time_acip:.3f} sec")
    print(f"Baseline - Avg Steps: {avg_steps_baseline:.1f}, Avg Time: {avg_time_baseline:.3f} sec")
    
    plt.figure(figsize=(10, 6))
    if len(conf_curves_acip) > 0 and len(conf_curves_baseline) > 0:
        plt.plot(conf_curves_acip[0], label='ACIP', linewidth=2, color='blue')
        plt.plot(conf_curves_baseline[0], label='Baseline', linewidth=2, color='red')
    
    plt.xlabel("Diffusion Iteration", fontsize=12)
    plt.ylabel("Classifier Confidence", fontsize=12)
    plt.title("Convergence Curve Comparison: ACIP vs Baseline", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = ".research/iteration1/images/convergence_comparison.pdf"
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Convergence comparison plot saved: {output_path}")
    
    return {
        'acip_steps': avg_steps_acip,
        'acip_time': avg_time_acip,
        'baseline_steps': avg_steps_baseline,
        'baseline_time': avg_time_baseline
    }

def evaluate_all_experiments():
    """
    Run all three experiments and return comprehensive results.
    """
    print("Starting ACIP Experimental Evaluation")
    print("=" * 60)
    
    exp1_results = run_experiment1()
    exp2_results = run_experiment2()
    exp3_results = run_experiment3()
    
    print("=" * 60)
    print("EXPERIMENTAL SUMMARY")
    print("=" * 60)
    print(f"Experiment 1 - ACIP Improvement: {exp1_results['improvement']:.3f}")
    print(f"Experiment 2 - Best variant: full_acip")
    print(f"Experiment 3 - ACIP efficiency: {exp3_results['acip_steps']:.1f} steps, {exp3_results['acip_time']:.3f}s")
    
    return {
        'experiment1': exp1_results,
        'experiment2': exp2_results,
        'experiment3': exp3_results
    }
