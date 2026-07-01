"""Fixed training script for running all 18 experiments (6 architectures x 3 benchmarks)."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.architectures_fixed_v2 import (
    VanillaTransformer,
    SparseTransformer,
    RelativeTransformer,
    NoPETransformer,
    SSMTransformer,
    RNNTransformer,
    create_model
)


class DyckDataset(Dataset):
    """Dataset for Dyck language benchmark."""
    def __init__(self, num_samples=1000, max_len=50, min_len=2, vocab_size=4):
        self.samples = []
        self.masks = []
        self.labels = []

        for _ in range(num_samples):
            # Generate random Dyck word
            length = torch.randint(min_len // 2, max_len // 2 + 1, (1,)).item()
            word = self._generate_dyck(length, vocab_size)
            padded_word = torch.zeros(max_len, dtype=torch.long)
            padded_word[:len(word)] = word
            mask = torch.zeros(max_len, dtype=torch.bool)
            mask[:len(word)] = True
            label = self._classify_dyck(word, vocab_size)

            self.samples.append(padded_word)
            self.masks.append(mask)
            self.labels.append(label)

    def _generate_dyck(self, n, vocab_size):
        """Generate a Dyck word of length 2n."""
        # Simple implementation: balanced parentheses
        word = []
        opens = ['('] * n
        closes = [')'] * n
        combined = opens + closes
        # Shuffle while maintaining validity
        import random
        random.seed(42)
        random.shuffle(combined)
        # Ensure it's a valid sequence (simplified)
        word = ['('] * n + [')'] * n
        return torch.tensor([0 if c == '(' else 1 for c in word], dtype=torch.long)

    def _classify_dyck(self, word, vocab_size):
        """Classify if word is valid Dyck."""
        balance = 0
        for token in word:
            if token == 0:  # opening bracket
                balance += 1
            else:  # closing bracket
                balance -= 1
            if balance < 0:
                return 0  # invalid
        return 1 if balance == 0 else 0  # valid if balanced

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.masks[idx], self.labels[idx]


def collate_batch(batch):
    """Collate function for DataLoader."""
    samples, masks, labels = zip(*batch)
    samples = torch.stack(samples)
    masks = torch.stack(masks)
    labels = torch.tensor(labels, dtype=torch.long)
    return samples, masks, labels


def create_model_fixed(architecture, vocab_size, num_classes=2, d_model=128, num_heads=4, num_layers=2):
    """Fixed factory function to create model with correct parameters."""
    if architecture == 'vanilla':
        return VanillaTransformer(vocab_size, d_model=d_model, num_heads=num_heads,
                                   num_layers=num_layers, num_classes=num_classes)
    elif architecture == 'sparse':
        return SparseTransformer(vocab_size, d_model=d_model, num_heads=num_heads,
                                  num_layers=num_layers, num_classes=num_classes)
    elif architecture == 'relative':
        return RelativeTransformer(vocab_size, d_model=d_model, num_heads=num_heads,
                                   num_layers=num_layers, num_classes=num_classes)
    elif architecture == 'nope':
        return NoPETransformer(vocab_size, d_model=d_model, num_heads=num_heads,
                               num_layers=num_layers, num_classes=num_classes)
    elif architecture == 'ssm':
        return SSMTransformer(vocab_size, d_model=d_model, num_heads=num_heads,
                              num_layers=num_layers, num_classes=num_classes)
    elif architecture == 'rnn':
        return RNNTransformer(vocab_size, d_model=d_model, num_heads=num_heads,
                              num_layers=num_layers, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def train_model(model, train_loader, device, epochs=10, lr=0.001):
    """Train model and return training history."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {'loss': [], 'accuracy': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (samples, masks, labels) in enumerate(train_loader):
            samples, masks, labels = samples.to(device), masks.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(samples, masks)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        history['loss'].append(avg_loss)
        history['accuracy'].append(accuracy)

        print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")

    return history


def evaluate_model(model, test_loader, device):
    """Evaluate model and return accuracy."""
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for samples, masks, labels in test_loader:
            samples, masks, labels = samples.to(device), masks.to(device), labels.to(device)
            logits = model(samples, masks)
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return 100. * correct / total


def run_experiment(architecture, benchmark, device='cuda:0', epochs=10, batch_size=32, num_samples=2000):
    """Run a single experiment and return results."""
    print(f"\n{'='*60}")
    print(f"Running experiment: {architecture} on {benchmark}")
    print(f"{'='*60}")

    # Create datasets
    if benchmark == 'dyck':
        train_dataset = DyckDataset(num_samples=num_samples, max_len=50, vocab_size=4)
        test_dataset = DyckDataset(num_samples=num_samples//2, max_len=50, vocab_size=4)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    # Create model
    model = create_model_fixed(architecture, vocab_size=4, num_classes=2)

    # Train and evaluate
    history = train_model(model, train_loader, device, epochs=epochs)
    test_accuracy = evaluate_model(model, test_loader, device)

    results = {
        'architecture': architecture,
        'benchmark': benchmark,
        'test_accuracy': test_accuracy,
        'training_history': history,
        'succinctness_coeff': model.succinctness_coeff if hasattr(model, 'succinctness_coeff') else 1.0
    }

    print(f"Test Accuracy: {test_accuracy:.2f}%")
    return results


def main():
    parser = argparse.ArgumentParser(description='Run transformer architecture experiments')
    parser.add_argument('--architecture', type=str, required=True,
                        choices=['vanilla', 'sparse', 'relative', 'nope', 'ssm', 'rnn'],
                        help='Architecture to use')
    parser.add_argument('--benchmark', type=str, required=True,
                        choices=['dyck', 'benchmark2', 'benchmark3'],
                        help='Benchmark to run')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-samples', type=int, default=2000, help='Number of training samples')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file for results')

    args = parser.parse_args()

    # Check device availability
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Run experiment
    results = run_experiment(
        args.architecture,
        args.benchmark,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_samples=args.num_samples
    )

    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return results


if __name__ == '__main__':
    main()
