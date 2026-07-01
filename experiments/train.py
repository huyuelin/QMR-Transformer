"""
Training script for Succinctness-Generalization Plane experiments.

Trains each architecture variant on the three benchmarks and evaluates
length generalization performance.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import yaml
import argparse
import os
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime

from models import create_model
from benchmarks.arithmetic import create_arithmetic_dataloaders
from benchmarks.dyck import create_dyck_dataloaders
from benchmarks.counting import create_counting_dataloaders


class Trainer:
    """Trainer for architecture variants on length generalization tasks."""
    
    def __init__(self, config: dict, architecture: str, benchmark: str, device: str = "cuda"):
        self.config = config
        self.architecture = architecture
        self.benchmark = benchmark
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Create model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=0.01
        )
        
        # Learning rate scheduler (linear warmup + decay)
        self.scheduler = self._create_scheduler()
        
        # Loss function (depends on benchmark)
        self.criterion = self._create_criterion()
        
        # Results directory
        self.results_dir = Path(config.get('results_dir', './results'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.test_results = {}
    
    def _create_model(self):
        """Create model based on architecture variant."""
        model_kwargs = {
            'vanilla': {
                'vocab_size': self._get_vocab_size(),
                'd_model': 256,
                'num_heads': 8,
                'num_layers': 6,
                'd_ff': 1024,
                'dropout': 0.1,
            },
            'sparse': {
                'vocab_size': self._get_vocab_size(),
                'd_model': 256,
                'num_heads': 8,
                'num_layers': 6,
                'd_ff': 1024,
                'dropout': 0.1,
                'k': self.config.get('sparse', {}).get('k', 64),
            },
            'relative': {
                'vocab_size': self._get_vocab_size(),
                'd_model': 256,
                'num_heads': 8,
                'num_layers': 6,
                'd_ff': 1024,
                'dropout': 0.1,
            },
            'nope': {
                'vocab_size': self._get_vocab_size(),
                'd_model': 256,
                'num_heads': 8,
                'num_layers': 6,
                'd_ff': 1024,
                'dropout': 0.1,
            },
            'ssm': {
                'vocab_size': self._get_vocab_size(),
                'd_model': 256,
                'num_heads': 8,
                'num_layers': 3,  # Fewer layers due to SSM augmentation
                'd_ff': 1024,
                'dropout': 0.1,
                'd_state': self.config.get('ssm', {}).get('d_state', 16),
                'expand': self.config.get('ssm', {}).get('expand', 2),
            },
            'rnn': {
                'vocab_size': self._get_vocab_size(),
                'd_model': 256,
                'num_heads': 8,
                'num_layers': 6,
                'd_ff': 1024,
                'dropout': 0.1,
                'rnn_type': self.config.get('rnn', {}).get('rnn_type', 'gru'),
            },
        }
        
        return create_model(self.architecture, **model_kwargs[self.architecture])
    
    def _get_vocab_size(self) -> int:
        """Get vocabulary size for the benchmark."""
        if self.benchmark == 'arithmetic':
            return 12  # digits 0-9 + SEP + EOS + PAD
        elif self.benchmark == 'dyck':
            num_types = self.config.get('benchmarks', {}).get('dyck', {}).get('num_types', 3)
            return 3 + 2 * num_types  # PAD + VALID + INVALID + bracket tokens
        elif self.benchmark == 'counting':
            return self.config.get('benchmarks', {}).get('counting', {}).get('vocab_size', 100)
        else:
            raise ValueError(f"Unknown benchmark: {self.benchmark}")
    
    def _create_criterion(self):
        """Create loss function based on benchmark."""
        if self.benchmark == 'arithmetic':
            return nn.CrossEntropyLoss(ignore_index=0)  # Ignore PAD
        elif self.benchmark == 'dyck':
            return nn.CrossEntropyLoss()
        elif self.benchmark == 'counting':
            return nn.MSELoss()
        else:
            raise ValueError(f"Unknown benchmark: {self.benchmark}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        def lr_lambda(step):
            warmup_steps = self.config.get('warmup_steps', 1000)
            if step < warmup_steps:
                return step / warmup_steps
            else:
                # Linear decay
                total_steps = self.config.get('num_epochs', 50) * 1000  # Approximate
                return max(0.0, (total_steps - step) / (total_steps - warmup_steps))
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self, train_loader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc="Training", leave=False):
            inputs, mask, targets = batch
            inputs = inputs.to(self.device)
            mask = mask.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Determine task type
            task_type = "classification" if self.benchmark in ['dyck'] else "seq2seq"
            outputs = self.model(inputs, mask, task_type=task_type)
            
            # Compute loss based on benchmark type
            if self.benchmark == 'arithmetic':
                # outputs: (batch, seq_len, vocab_size)
                # targets: (batch, seq_len)
                loss = self.criterion(
                    outputs.view(-1, outputs.size(-1)),
                    targets.view(-1)
                )
            elif self.benchmark == 'dyck':
                # outputs: (batch, vocab_size) for classification
                # targets: (batch,) with class indices
                loss = self.criterion(outputs, targets)
            elif self.benchmark == 'counting':
                # outputs: (batch, vocab_size) for regression
                # targets: (batch, vocab_size) histogram
                pooled = outputs.mean(dim=1) if outputs.dim() == 3 else outputs
                loss = self.criterion(pooled, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def evaluate(self, test_loader) -> dict:
        """Evaluate model on test data."""
        self.model.eval()
        total_error = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in test_loader:
                inputs, mask, targets = batch
                inputs = inputs.to(self.device)
                mask = mask.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs, mask)
                
                # Compute error based on benchmark type
                if self.benchmark == 'arithmetic':
                    # Sequence accuracy (whole sequence must match)
                    preds = outputs.argmax(dim=-1)
                    # Simplified: compute token-level accuracy
                    error = (preds != targets).float().mean().item()
                elif self.benchmark == 'dyck':
                    # Classification accuracy
                    logits = outputs[:, 1:3]  # VALID/INVALID logits
                    preds = logits.argmax(dim=-1)
                    error = (preds != targets).float().mean().item()
                elif self.benchmark == 'counting':
                    # MSE on histogram prediction
                    pooled = outputs.mean(dim=1)
                    error = F.mse_loss(pooled, targets).item()
                
                total_error += error * inputs.size(0)
                num_samples += inputs.size(0)
        
        return {
            'error': total_error / max(num_samples, 1),
            'num_samples': num_samples,
        }
    
    def run_experiment(self):
        """Run full training and evaluation experiment."""
        print(f"\n{'='*60}")
        print(f"Architecture: {self.architecture}")
        print(f"Benchmark: {self.benchmark}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        # Create dataloaders
        if self.benchmark == 'arithmetic':
            benchmark_config = self.config.get('benchmarks', {}).get('arithmetic', {})
            train_loader, test_loaders = create_arithmetic_dataloaders(benchmark_config)
        elif self.benchmark == 'dyck':
            benchmark_config = self.config.get('benchmarks', {}).get('dyck', {})
            train_loader, test_loaders = create_dyck_dataloaders(benchmark_config)
        elif self.benchmark == 'counting':
            benchmark_config = self.config.get('benchmarks', {}).get('counting', {})
            train_loader, test_loaders = create_counting_dataloaders(benchmark_config)
        else:
            raise ValueError(f"Unknown benchmark: {self.benchmark}")
        
        # Training loop
        num_epochs = self.config.get('num_epochs', 50)
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}")
        
        # Evaluation on test lengths
        print(f"\nEvaluating {self.architecture} on {self.benchmark}...")
        results = {}
        for test_key, test_loader in test_loaders.items():
            eval_result = self.evaluate(test_loader)
            results[test_key] = eval_result
            print(f"  Test length/depth {test_key}: Error = {eval_result['error']:.4f}")
        
        self.test_results = results
        
        # Save results
        self._save_results()
        
        return results
    
    def _save_results(self):
        """Save experiment results to JSON."""
        results = {
            'architecture': self.architecture,
            'benchmark': self.benchmark,
            'succinctness_coeff': getattr(self.model, 'succinctness_coeff', 1.0),
            'train_losses': self.train_losses,
            'test_results': {str(k): v for k, v in self.test_results.items()},
            'timestamp': datetime.now().isoformat(),
        }
        
        filename = f"{self.architecture}_{self.benchmark}_results.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Train architecture variants on benchmarks')
    parser.add_argument('--config', type=str, default='../config.yaml', help='Path to config file')
    parser.add_argument('--architecture', type=str, required=True, 
                       choices=['vanilla', 'sparse', 'relative', 'nope', 'ssm', 'rnn'],
                       help='Architecture variant to train')
    parser.add_argument('--benchmark', type=str, required=True,
                       choices=['arithmetic', 'dyck', 'counting'],
                       help='Benchmark dataset to use')
    parser.add_argument('--device', type=str, default=None, help='Device to use (overrides config)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    if args.device:
        config['device'] = args.device
    
    # Create trainer and run experiment
    trainer = Trainer(config, args.architecture, args.benchmark, config.get('device', 'cuda'))
    trainer.run_experiment()


if __name__ == '__main__':
    main()
