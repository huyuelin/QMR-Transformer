"""Algorithmic Counting Benchmark - Full Implementation.

Task: Count the frequency of each token type in the sequence.
This is a regression task (predict histogram).
"""

import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from collections import Counter


class CountingDataset(Dataset):
    """Dataset for algorithmic counting with variable sequence lengths."""
    
    PAD = 0
    
    def __init__(self, max_length=20, vocab_size=100, num_samples=5000, seed=42, split="train"):
        """
        Args:
            max_length: Maximum sequence length
            vocab_size: Size of token vocabulary
            num_samples: Number of samples to generate
            seed: Random seed
            split: "train" or "test"
        """
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.num_samples = num_samples
        self.split = split
        
        random.seed(seed)
        np.random.seed(seed)
        
        self.data = self._generate_data()
    
    def _generate_data(self):
        """Generate (input_sequence, count_histogram) pairs."""
        data = []
        
        for _ in range(self.num_samples):
            # Sample sequence length
            length = random.randint(1, self.max_length)
            
            # Generate random tokens (avoid PAD=0)
            tokens = [random.randint(1, self.vocab_size - 1) for _ in range(length)]
            
            # Compute histogram (count of each token type)
            counts = Counter(tokens)
            histogram = torch.zeros(self.vocab_size, dtype=torch.float32)
            for token, count in counts.items():
                histogram[token] = count
            
            # Normalize histogram by length
            histogram = histogram / length
            
            data.append({
                'input': torch.tensor(tokens, dtype=torch.long),
                'histogram': histogram
            })
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def counting_collate_fn(batch):
    """Collate function for counting dataset."""
    inputs = [item['input'] for item in batch]
    histograms = torch.stack([item['histogram'] for item in batch])
    
    # Pad inputs
    max_len = max(len(x) for x in inputs)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    
    for i, inp in enumerate(inputs):
        padded[i, :len(inp)] = inp
        mask[i, :len(inp)] = True
    
    return padded, mask, histograms


def create_counting_dataloaders(config):
    """Create train and test dataloaders for counting benchmark."""
    train_max_length = config.get('train_length', 20)
    test_max_length = config.get('test_length', 100)
    vocab_size = config.get('vocab_size', 100)
    num_samples = config.get('num_samples', 5000)
    batch_size = config.get('batch_size', 32)
    seed = config.get('seed', 42)
    
    train_dataset = CountingDataset(
        max_length=train_max_length,
        vocab_size=vocab_size,
        num_samples=num_samples,
        seed=seed,
        split="train"
    )
    
    test_dataset = CountingDataset(
        max_length=test_max_length,
        vocab_size=vocab_size,
        num_samples=num_samples // 5,
        seed=seed + 1000,
        split="test"
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=counting_collate_fn,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=counting_collate_fn,
        num_workers=0
    )
    
    return train_loader, test_loader


if __name__ == '__main__':
    # Test the dataset
    dataset = CountingDataset(max_length=20, vocab_size=100, num_samples=100)
    print(f"Dataset size: {len(dataset)}")
    
    # Check a few samples
    for i in range(3):
        sample = dataset[i]
        print(f"Sample {i}: input={sample['input'].tolist()}, histogram_sum={sample['histogram'].sum().item():.4f}")
    
    print("\nDataset creation successful!")
