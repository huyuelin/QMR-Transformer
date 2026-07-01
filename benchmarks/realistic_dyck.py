"""More realistic Dyck dataset for meaningful experiments."""

import torch
from torch.utils.data import Dataset, DataLoader
import random
from typing import List, Tuple
from collections import deque

class RealisticDyckDataset(Dataset):
    """Generate more realistic Dyck sequences with variable lengths."""
    
    def __init__(self, max_depth=6, num_types=3, num_samples=5000, seed=42, split="train"):
        self.max_depth = max_depth
        self.num_types = num_types
        self.num_samples = num_samples
        
        random.seed(seed)
        
        self.data = []
        self._generate_data()
    
    def _generate_valid_dyck(self, length: int) -> List[int]:
        """Generate a valid Dyck word of given length (must be even)."""
        sequence = []
        stack = []
        
        while len(sequence) < length:
            if len(stack) == 0 or (len(stack) < self.max_depth and random.random() < 0.5):
                # Add opening bracket
                bracket_type = random.randint(0, self.num_types - 1)
                token = 3 + 2 * bracket_type  # Opening bracket
                sequence.append(token)
                stack.append(bracket_type)
            else:
                # Add closing bracket
                bracket_type = stack.pop()
                token = 3 + 2 * bracket_type + 1  # Closing bracket
                sequence.append(token)
        
        return sequence
    
    def _generate_invalid_dyck(self, length: int) -> List[int]:
        """Generate an invalid bracket sequence."""
        valid = self._generate_valid_dyck(length)
        
        # Corruption strategies
        strategy = random.choice(['swap', 'delete', 'insert', 'mismatch'])
        
        if strategy == 'swap' and length >= 2:
            idx = random.randint(0, length - 2)
            valid[idx], valid[idx + 1] = valid[idx + 1], valid[idx]
        elif strategy == 'delete' and length >= 3:
            idx = random.randint(0, length - 1)
            valid.pop(idx)
            valid.append(random.choice([3, 5, 7]))
        elif strategy == 'insert':
            idx = random.randint(0, length)
            valid.insert(idx, random.choice([3, 5, 7]))
        elif strategy == 'mismatch':
            for i in range(len(valid)):
                if valid[i] % 2 == 1:  # Closing bracket
                    valid[i] = random.choice([3, 5, 7]) + 1
                    break
        
        return valid
    
    def _generate_data(self):
        """Generate (input_sequence, label) pairs."""
        for _ in range(self.num_samples):
            # Sample sequence length (even number)
            length = random.randint(2, self.max_depth * 2)
            if length % 2 != 0:
                length += 1
            
            # 50% valid, 50% invalid
            if random.random() < 0.5:
                seq = self._generate_valid_dyck(length)
                label = 1  # VALID
            else:
                seq = self._generate_invalid_dyck(length)
                label = 2  # INVALID
            
            self.data.append({
                'input': torch.tensor(seq, dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.long)
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def realistic_collate_fn(batch):
    """Collate function for realistic Dyck dataset."""
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


if __name__ == '__main__':
    # Test the dataset
    dataset = RealisticDyckDataset(max_depth=6, num_samples=100)
    print(f"Dataset size: {len(dataset)}")
    
    # Check a few samples
    for i in range(3):
        sample = dataset[i]
        print(f"Sample {i}: input={sample['input'].tolist()}, label={sample['label'].item()}")
    
    print("\nDataset creation successful!")
