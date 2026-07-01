"""
Algorithmic Counting Benchmark
Length extrapolation: train on 20 tokens, test on up to 100 tokens.

Task: Count the frequency of each token type in the sequence.
Input format: [token1, token2, ..., tokenN]
Output format: [count1, count2, ..., countM] (histogram of token frequencies)

Succinctness insight: Architectures with better length generalization should 
maintain counting accuracy on longer sequences.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from typing import List, Tuple
from collections import Counter


class CountingDataset(Dataset):
    """Dataset for algorithmic counting with variable sequence lengths."""
    
    def __init__(self, max_length: int, vocab_size: int = 100, 
                 num_samples: int = 10000, seed: int = 42, split: str = "train"):
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
        
        # Special tokens
        self.PAD = 0
        # Output is a histogram of size vocab_size
        
        random.seed(seed)
        np.random.seed(seed)
        
        self.data = self._generate_data()
    
    def _generate_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Generate (input_sequence, count_histogram) pairs."""
        data = []
        
        for _ in range(self.num_samples):
            # Sample sequence length
            length = random.randint(1, self.max_length)
            
            # Generate random tokens
            tokens = [random.randint(1, self.vocab_size - 1) for _ in range(length)]
            
            # Compute histogram (count of each token type)
            counts = Counter(tokens)
            histogram = torch.zeros(self.vocab_size, dtype=torch.float32)
            for token, count in counts.items():
                histogram[token] = count
            
            data.append((
                torch.tensor(tokens, dtype=torch.long),
                histogram
            ))
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    @staticmethod
    def collate_fn(batch):
        """Pad sequences to max length in batch."""
        inputs, histograms = zip(*batch)
        
        # Pad inputs
        input_lens = [len(x) for x in inputs]
        max_input_len = max(input_lens)
        padded_inputs = torch.zeros(len(batch), max_input_len, dtype=torch.long)
        input_mask = torch.zeros(len(batch), max_input_len, dtype=torch.bool)
        
        for i, (inp, ln) in enumerate(zip(inputs, input_lens)):
            padded_inputs[i, :ln] = inp
            input_mask[i, :ln] = True
        
        # Stack histograms
        histograms = torch.stack(histograms)
        
        return padded_inputs, input_mask, histograms


def create_counting_dataloaders(config: dict):
    """Create train and test dataloaders for counting benchmark."""
    train_dataset = CountingDataset(
        max_length=config['train_length'],
        vocab_size=config.get('vocab_size', 100),
        num_samples=10000,
        seed=config.get('seed', 42),
        split="train"
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 128),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        collate_fn=CountingDataset.collate_fn
    )
    
    # Create test datasets for different lengths
    test_loaders = {}
    for test_len in config.get('test_lengths', [40, 60, 80, 100]):
        test_dataset = CountingDataset(
            max_length=test_len,
            vocab_size=config.get('vocab_size', 100),
            num_samples=1000,
            seed=config.get('seed', 42) + test_len,
            split="test"
        )
        
        test_loaders[test_len] = DataLoader(
            test_dataset,
            batch_size=config.get('batch_size', 128),
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            collate_fn=CountingDataset.collate_fn
        )
    
    return train_loader, test_loaders
