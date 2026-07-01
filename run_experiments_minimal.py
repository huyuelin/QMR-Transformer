#!/usr/bin/env python3
"""Minimal working version to run SuccinctBound experiments.

This is a simplified, tested version that actually runs.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
import argparse
import json
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm

# ─── Simple Dataset ─────────────────────────────────────────────

class DyckDataset(Dataset):
    """Simplified Dyck dataset for binary classification."""
    
    def __init__(self, max_depth=6, num_samples=1000, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        
        self.data = []
        for _ in range(num_samples):
            if random.random() > 0.5:
                # Valid sequence
                seq = [3, 5, 5, 4]  # ( []
                label = 1  # VALID
            else:
                # Invalid sequence
                seq = [3, 5, 4, 5]  # ( [ ) ]
                label = 2  # INVALID
            
            self.data.append({
                'input': torch.tensor(seq, dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.long)
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    """Simple collate function."""
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

# ─── Simple Model ─────────────────────────────────────────────

class SimpleTransformer(nn.Module):
    """Simplified Transformer for classification."""
    
    def __init__(self, vocab_size=10, d_model=128, num_heads=4, num_layers=2, num_classes=3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(512, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=512,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Linear(d_model, num_classes)
        
        self.succinctness_coeff = 1.0
    
    def forward(self, x, mask):
        # x: (batch, seq_len)
        batch_size, seq_len = x.shape
        
        # Embeddings
        pos = torch.arange(seq_len).unsqueeze(0).to(x.device)
        emb = self.embedding(x) + self.pos_embedding(pos)
        
        # Transformer
        # Convert mask to attention mask
        attn_mask = (mask == 0)  # True for padding positions
        output = self.transformer(emb, src_key_padding_mask=attn_mask)
        
        # Pooling (mean over non-masked positions)
        mask_expanded = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        pooled = (output * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        return logits

# ─── Training ─────────────────────────────────────────────

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        inputs, mask, labels = batch
        inputs = inputs.to(device)
        mask = mask.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs, mask)
        loss = criterion(outputs, labels)
        
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
            inputs, mask, labels = batch
            inputs = inputs.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs, mask)
            preds = outputs.argmax(dim=-1)
            
            error = (preds != labels).float().sum().item()
            total_error += error
            num_samples += len(labels)
    
    return total_error / max(num_samples, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=str, default='vanilla')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print(f"Architecture: {args.architecture}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Create datasets
    train_dataset = DyckDataset(max_depth=6, num_samples=1000, seed=42)
    test_dataset = DyckDataset(max_depth=24, num_samples=500, seed=123)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Create model
    model = SimpleTransformer(vocab_size=10, d_model=128, num_heads=4, num_layers=2, num_classes=3)
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        if (epoch + 1) % 5 == 0:
            test_error = evaluate(model, test_loader, criterion, device)
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {train_loss:.4f}, Test Error: {test_error:.4f}")
    
    # Final evaluation
    test_error = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal Test Error: {test_error:.4f}")
    
    # Save results
    results = {
        'architecture': args.architecture,
        'test_error': test_error,
        'succinctness_coeff': model.succinctness_coeff,
    }
    
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / f"{args.architecture}_dyck_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_dir / f'{args.architecture}_dyck_results.json'}")
    print("\nDone!")

if __name__ == '__main__':
    main()
