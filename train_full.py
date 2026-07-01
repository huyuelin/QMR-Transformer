#!/usr/bin/env python3
"""Full training script for SuccinctBound experiments."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import json
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import models
from models.architectures import (
    VanillaTransformer,
    SparseTransformer,
    RelativeTransformer,
    NoPETransformer,
    SSMTransformer,
    RNNTransformer,
)


def create_model(architecture, vocab_size, num_classes, **kwargs):
    """Factory function to create model."""
    d_model = kwargs.get('d_model', 128)
    num_heads = kwargs.get('num_heads', 4)
    num_layers = kwargs.get('num_layers', 2)
    d_ff = kwargs.get('d_ff', 512)
    dropout = kwargs.get('dropout', 0.1)
    
    if architecture == 'vanilla':
        return VanillaTransformer(vocab_size, d_model, num_heads, num_layers, d_ff, dropout, num_classes)
    elif architecture == 'sparse':
        k = kwargs.get('k', 64)
        return SparseTransformer(vocab_size, d_model, num_heads, num_layers, d_ff, dropout, k, num_classes)
    elif architecture == 'relative':
        return RelativeTransformer(vocab_size, d_model, num_heads, num_layers, d_ff, dropout, num_classes)
    elif architecture == 'nope':
        return NoPETransformer(vocab_size, d_model, num_heads, num_layers, d_ff, dropout, num_classes)
    elif architecture == 'ssm':
        d_state = kwargs.get('d_state', 16)
        expand = kwargs.get('expand', 2)
        return SSMTransformer(vocab_size, d_model, num_heads, num_layers, d_ff, dropout, d_state, expand, num_classes)
    elif architecture == 'rnn':
        rnn_type = kwargs.get('rnn_type', 'gru')
        return RNNTransformer(vocab_size, d_model, num_heads, num_layers, d_ff, dropout, rnn_type, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        inputs, mask, targets = batch
        inputs = inputs.to(device)
        mask = mask.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs, mask)
        loss = criterion(outputs, targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_error = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, mask, targets = batch
            inputs = inputs.to(device)
            mask = mask.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs, mask)
            preds = outputs.argmax(dim=-1)
            
            error = (preds != targets).float().sum().item()
            total_error += error
            num_samples += len(targets)
    
    return total_error / max(num_samples, 1)


def run_experiment(architecture, benchmark, device, config):
    """Run a single experiment."""
    
    print(f"\n{'='*60}")
    print(f"Architecture: {architecture}")
    print(f"Benchmark: {benchmark}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Get vocab_size and num_classes based on benchmark
    if benchmark == 'arithmetic':
        vocab_size = 12  # digits 0-9 + SEP + EOS + PAD
        num_classes = 2  # even/odd
    elif benchmark == 'dyck':
        vocab_size = 9  # PAD(0) + VALID(1) + INVALID(2) + bracket tokens (3-8)
        num_classes = 2  # valid/invalid
    elif benchmark == 'counting':
        vocab_size = config.get('vocab_size', 100)
        num_classes = vocab_size  # regression: predict histogram
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    
    # Import correct dataset
    if benchmark == 'arithmetic':
        from benchmarks.arithmetic_dataset import ArithmeticDataset, arithmetic_collate_fn
        train_dataset = ArithmeticDataset(max_digits=config.get('train_length', 10), num_samples=5000, seed=42)
        test_dataset = ArithmeticDataset(max_digits=config.get('test_length', 50), num_samples=1000, seed=123)
        train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 64), shuffle=True, collate_fn=arithmetic_collate_fn, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=config.get('batch_size', 64), shuffle=False, collate_fn=arithmetic_collate_fn, num_workers=0)
        
    elif benchmark == 'dyck':
        from benchmarks.dyck import DyckDataset, dyck_collate_fn
        train_dataset = DyckDataset(max_depth=config.get('train_depth', 6), num_types=config.get('num_types', 3), num_samples=5000, seed=42)
        test_dataset = DyckDataset(max_depth=config.get('test_depth', 24), num_types=config.get('num_types', 3), num_samples=1000, seed=123)
        train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 64), shuffle=True, collate_fn=dyck_collate_fn, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=config.get('batch_size', 64), shuffle=False, collate_fn=dyck_collate_fn, num_workers=0)
        
    elif benchmark == 'counting':
        from benchmarks.counting_dataset import CountingDataset, counting_collate_fn
        train_dataset = CountingDataset(max_length=config.get('train_length', 20), vocab_size=vocab_size, num_samples=5000, seed=42)
        test_dataset = CountingDataset(max_length=config.get('test_length', 100), vocab_size=vocab_size, num_samples=1000, seed=123)
        train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 64), shuffle=True, collate_fn=counting_collate_fn, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=config.get('batch_size', 64), shuffle=False, collate_fn=counting_collate_fn, num_workers=0)
    
    # Create model
    model = create_model(architecture, vocab_size, num_classes)
    model.to(device)
    
    # Set succinctness coefficient
    succinctness_coeffs = {
        'vanilla': 1.0,
        'sparse': 0.7,
        'relative': 0.85,
        'nope': 0.9,
        'ssm': 0.6,
        'rnn': 0.8,
    }
    model.succinctness_coeff = succinctness_coeffs.get(architecture, 1.0)
    
    # Optimizer
    epochs = config.get('epochs', 10)
    learning_rate = config.get('learning_rate', 1e-3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        if (epoch + 1) % 5 == 0:
            test_error = evaluate(model, test_loader, criterion, device)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Test Error: {test_error:.4f}")
    
    # Final evaluation
    test_error = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal Test Error: {test_error:.4f}")
    
    # Save results
    results = {
        'architecture': architecture,
        'benchmark': benchmark,
        'test_error': test_error,
        'succinctness_coeff': model.succinctness_coeff,
    }
    
    results_dir = Path(config.get('results_dir', 'results'))
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / f"{architecture}_{benchmark}_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_dir / f'{architecture}_{benchmark}_results.json'}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run SuccinctBound experiments')
    parser.add_argument('--architecture', type=str, required=True,
                        choices=['vanilla', 'sparse', 'relative', 'nope', 'ssm', 'rnn'])
    parser.add_argument('--benchmark', type=str, required=True,
                        choices=['arithmetic', 'dyck', 'counting'])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--results-dir', type=str, default='results')
    args = parser.parse_args()
    
    # Check device
    if args.device.startswith('cuda'):
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    # Config
    config = {
        'train_length': 10,
        'test_length': 50,
        'train_depth': 6,
        'test_depth': 24,
        'vocab_size': 100,
        'num_samples': 5000,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': 1e-3,
        'seed': 42,
        'results_dir': args.results_dir,
    }
    
    # Run experiment
    run_experiment(args.architecture, args.benchmark, device, config)


if __name__ == '__main__':
    main()
