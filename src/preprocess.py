#!/usr/bin/env python
"""
Data preprocessing for ACIP experiments.
Handles CIFAR-10 dataset loading and adversarial example generation.
"""

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_cifar10_data(batch_size=32, max_batches=2):
    """
    Load CIFAR-10 test dataset for experiments.
    Limited to max_batches for quick testing.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

def fgsm_attack(model, image, label, epsilon):
    """
    Generate adversarial examples using Fast Gradient Sign Method (FGSM).
    """
    image.requires_grad = True
    output = model(image)
    loss = F.cross_entropy(output, label)
    model.zero_grad()
    loss.backward()
    perturbed_image = image + epsilon * image.grad.sign()
    return torch.clamp(perturbed_image, 0, 1)

def add_diffusion_noise(images, noise_factor=0.1):
    """
    Add noise to simulate forward diffusion process.
    """
    return images + torch.randn_like(images) * noise_factor

def preprocess_data_for_experiment(max_batches=2, epsilon=0.05):
    """
    Main preprocessing function that loads data and generates adversarial examples.
    Returns data loader and preprocessing statistics.
    """
    print("Loading CIFAR-10 dataset...")
    data_loader = load_cifar10_data(max_batches=max_batches)
    
    print(f"Dataset loaded with batch size 32, limited to {max_batches} batches for testing")
    print(f"Adversarial attack epsilon: {epsilon}")
    print(f"Device: {device}")
    
    return data_loader, {"epsilon": epsilon, "max_batches": max_batches}
