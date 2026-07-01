"""Arithmetic Addition Benchmark - Full Implementation.

Task: Given two numbers as digit sequences, predict if their sum is even or odd.
This is a binary classification task suitable for length generalization.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np


class ArithmeticDataset(Dataset):
    """Arithmetic addition dataset for length generalization."""
    
    PAD = 0
    SEP = 10
    EOS = 11
    
    def __init__(self, max_digits=10, num_samples=5000, seed=42, split="train"):
        """
        Args:
            max_digits: Maximum number of digits per number
            num_samples: Number of samples to generate
            seed: Random seed
            split: "train" or "test"
        """
        self.max_digits = max_digits
        self.num_samples = num_samples
        self.split = split
        self.vocab_size = 12  # digits 0-9 + SEP + EOS + PAD
        
        random.seed(seed)
        np.random.seed(seed)
        
        self.data = self._generate_data()
    
    def _generate_data(self):
        """Generate (input_sequence, label) pairs.
        Label: 0 = even sum, 1 = odd sum
        """
        data = []
        
        for _ in range(self.num_samples):
            # Sample random numbers with up to max_digits
            num_digits_a = random.randint(1, self.max_digits)
            num_digits_b = random.randint(1, self.max_digits)
            
            # Generate random digits
            digits_a = [random.randint(0, 9) for _ in range(num_digits_a)]
            digits_b = [random.randint(0, 9) for _ in range(num_digits_b)]
            
            # Avoid leading zeros
            if digits_a[0] == 0 and num_digits_a > 1:
                digits_a[0] = random.randint(1, 9)
            if digits_b[0] == 0 and num_digits_b > 1:
                digits_b[0] = random.randint(1, 9)
            
            # Compute sum
            num_a = int(''.join(map(str, digits_a)))
            num_b = int(''.join(map(str, digits_b)))
            total = num_a + num_b
            
            # Label: 0 = even, 1 = odd
            label = total % 2
            
            # Create input sequence: [digits_a, SEP, digits_b]
            input_seq = digits_a + [self.SEP] + digits_b
            
            data.append({
                'input': torch.tensor(input_seq, dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.long)
            })
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def arithmetic_collate_fn(batch):
    """Collate function for arithmetic dataset."""
    inputs = [item['input'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    
    # Pad inputs
    max_len = max(len(x) for x in inputs)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    
    for i, inp in enumerate(inputs):
        padded[i, :len(inp)] = inp
        mask[i, :len(inp)] = True
    
    return padded, mask, labels


def create_arithmetic_dataloaders(config):
    """Create train and test dataloaders for arithmetic benchmark."""
    train_max_digits = config.get('train_length', 10)
    test_max_digits = config.get('test_length', 50)
    num_samples = config.get('num_samples', 5000)
    batch_size = config.get('batch_size', 32)
    seed = config.get('seed', 42)
    
    train_dataset = ArithmeticDataset(
        max_digits=train_max_digits,
        num_samples=num_samples,
        seed=seed,
        split="train"
    )
    
    test_dataset = ArithmeticDataset(
        max_digits=test_max_digits,
        num_samples=num_samples // 5,
        seed=seed + 1000,
        split="test"
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=arithmetic_collate_fn,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=arithmetic_collate_fn,
        num_workers=0
    )
    
    return train_loader, test_loader


if __name__ == '__main__':
    # Test the dataset
    dataset = ArithmeticDataset(max_digits=10, num_samples=100)
    print(f"Dataset size: {len(dataset)}")
    
    # Check a few samples
    for i in range(3):
        sample = dataset[i]
        print(f"Sample {i}: input={sample['input'].tolist()}, label={sample['label'].item()}")
    
    print("\nDataset creation successful!")
