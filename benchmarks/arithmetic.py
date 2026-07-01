"""
Arithmetic Addition Benchmark
Length extrapolation: train on 10-digit numbers, test on up to 50-digit numbers.

Task: Given two numbers as digit sequences, predict their sum.
Input format: [digit1, digit2, ..., digitN, SEP, digit1, digit2, ..., digitN]
Output format: [digit1, digit2, ..., digitM] (variable length sum)

Succinctness insight: More succinct architectures should generalize better to longer sequences.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from typing import List, Tuple


class ArithmeticDataset(Dataset):
    """Dataset for arithmetic addition with variable length sequences."""
    
    # Special tokens
    PAD = 0
    SEP = 10
    EOS = 11
    
    def __init__(self, max_digits: int, num_samples: int = 10000, 
                 seed: int = 42, split: str = "train"):
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
    
    def _generate_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Generate (input_sequence, target_sequence) pairs."""
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
            sum_digits = [int(d) for d in str(total)]
            
            # Create input sequence: [digits_a, SEP, digits_b]
            input_seq = digits_a + [self.SEP] + digits_b
            
            # Create target sequence: [sum_digits, EOS]
            target_seq = sum_digits + [self.EOS]
            
            data.append((
                torch.tensor(input_seq, dtype=torch.long),
                torch.tensor(target_seq, dtype=torch.long)
            ))
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    @staticmethod
    def collate_fn(batch):
        """Pad sequences to max length in batch."""
        inputs, targets = zip(*batch)
        
        # Pad inputs
        input_lens = [len(x) for x in inputs]
        max_input_len = max(input_lens)
        padded_inputs = torch.zeros(len(batch), max_input_len, dtype=torch.long)
        input_mask = torch.zeros(len(batch), max_input_len, dtype=torch.bool)
        
        for i, (inp, ln) in enumerate(zip(inputs, input_lens)):
            padded_inputs[i, :ln] = inp
            input_mask[i, :ln] = True
        
        # Pad targets
        target_lens = [len(x) for x in targets]
        max_target_len = max(target_lens)
        padded_targets = torch.zeros(len(batch), max_target_len, dtype=torch.long)
        
        for i, (tgt, ln) in enumerate(zip(targets, target_lens)):
            padded_targets[i, :ln] = tgt
        
        return padded_inputs, input_mask, padded_targets


def create_arithmetic_dataloaders(config: dict):
    """Create train and test dataloaders for arithmetic benchmark."""
    train_length = config.get('train_length', 10)
    train_dataset = ArithmeticDataset(
        max_digits=train_length,
        num_samples=10000,
        seed=config.get('seed', 42),
        split="train"
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 128),
        shuffle=True,
        num_workers=config.get('num_workers', 0),
        collate_fn=ArithmeticDataset.collate_fn
    )
    
    # Create test datasets for different lengths
    test_loaders = {}
    for test_len in config.get('test_lengths', [20, 30, 40, 50]):
        test_dataset = ArithmeticDataset(
            max_digits=test_len,
            num_samples=1000,
            seed=config.get('seed', 42) + test_len,  # Different seed for each test set
            split="test"
        )
        
        test_loaders[test_len] = DataLoader(
            test_dataset,
            batch_size=config.get('batch_size', 128),
            shuffle=False,
            num_workers=config.get('num_workers', 0),
            collate_fn=ArithmeticDataset.collate_fn
        )
    
    return train_loader, test_loaders
