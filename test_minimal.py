#!/usr/bin/env python3
"""Minimal working example to test the training pipeline."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import sys

# Add parent directory to path
sys.path.insert(0, '.')

def test_dyck_classification():
    """Test Dyck parsing as binary classification."""
    
    # Simple Dataset
    class SimpleDyckDataset(Dataset):
        def __init__(self, num_samples=100, max_depth=6):
            self.data = []
            for _ in range(num_samples):
                # Generate valid sequence
                if random.random() > 0.5:
                    seq = [3, 5, 5, 4]  # Valid: ()[]
                    label = 1  # VALID
                else:
                    seq = [3, 5, 4, 5]  # Invalid: ([)]
                    label = 2  # INVALID
                self.data.append((torch.tensor(seq), torch.tensor(label)))
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    # Simple Model (pooling + classification)
    class SimpleTransformer(nn.Module):
        def __init__(self, vocab_size=10, d_model=64, num_heads=4, num_layers=2):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_embedding = nn.Embedding(512, d_model)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(d_model, num_heads, batch_first=True)
                for _ in range(num_layers)
            ])
            self.pooler = nn.AdaptiveAvgPool1d(1)
            self.classifier = nn.Linear(d_model, 3)  # PAD, VALID, INVALID
            
        def forward(self, x, mask=None):
            # x: (batch, seq_len)
            seq_len = x.size(1)
            pos = torch.arange(seq_len).unsqueeze(0).to(x.device)
            
            emb = self.embedding(x) + self.pos_embedding(pos)
            
            for layer in self.layers:
                if mask is not None:
                    # Convert mask to attention mask
                    attn_mask = mask.unsqueeze(1).unsqueeze(2)
                    attn_mask = attn_mask.expand(-1, -1, seq_len, -1)
                    emb = layer(emb, src_mask=attn_mask)
                else:
                    emb = layer(emb)
            
            # Pool over sequence
            pooled = emb.mean(dim=1)  # (batch, d_model)
            logits = self.classifier(pooled)  # (batch, 3)
            return logits
    
    # Test
    print("Testing minimal Dyck classification...")
    
    dataset = SimpleDyckDataset(num_samples=100)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = SimpleTransformer()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Train for 1 epoch
    model.train()
    for batch_idx, (inputs, targets) in enumerate(loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 2 == 0:
            print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    print("  Training successful!")
    return True

if __name__ == '__main__':
    try:
        test_dyck_classification()
        print("\n✓ Minimal test passed! Now scaling up to full experiments...")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
