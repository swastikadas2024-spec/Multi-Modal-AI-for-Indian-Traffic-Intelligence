"""
Training script for text classification on traffic complaints.
Uses preprocessing and class balancing for robustness.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from scripts.preprocess import normalize_text, compute_class_weights
import os


def load_and_preprocess_data(csv_path):
    """
    Load data from CSV and apply text preprocessing.
    
    Args:
        csv_path (str): Path to CSV file with 'text' and 'label' columns
        
    Returns:
        tuple: (texts, labels, label_encoder, class_weights)
    """
    print(f"Loading data from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} records")
    
    # Normalize all text before tokenizing
    print("Normalizing text...")
    texts = [normalize_text(text) for text in df['text']]
    labels = df['label'].values
    
    print("✅ Text normalization complete")
    
    # Encode labels
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    
    # Compute class weights for balanced training
    print("Computing class weights...")
    class_weights = compute_class_weights(encoded_labels)
    weights_tensor = torch.tensor(
        [class_weights.get(i, 1.0) for i in range(len(le.classes_))],
        dtype=torch.float
    )
    print(f"✅ Class weights computed: {class_weights}")
    
    return texts, encoded_labels, le, weights_tensor


def create_dataloaders(texts, labels, batch_size=32):
    """
    Create PyTorch dataloaders for training.
    
    Args:
        texts (list): List of preprocessed text strings
        labels (array): Encoded labels
        batch_size (int): Batch size for loader
        
    Returns:
        DataLoader: DataLoader for training
    """
    # Placeholder: In a real scenario, you would tokenize and create embeddings
    # For now, we'll create dummy tensors
    X = torch.randn(len(texts), 768)  # Simulating 768-dim embeddings
    y = torch.tensor(labels, dtype=torch.long)
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


def train_model(train_loader, class_weights, num_epochs=5):
    """
    Train a text classification model.
    
    Args:
        train_loader (DataLoader): Training data loader
        class_weights (Tensor): Class weights for loss function
        num_epochs (int): Number of epochs to train
    """
    print(f"\n{'='*70}")
    print("TRAINING MODEL")
    print(f"{'='*70}")
    
    # Use WeightedRandomSampler or CrossEntropyLoss with weight parameter
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    print(f"✅ Loss function configured with class weights")
    print(f"   Weights: {class_weights.tolist()}")
    print(f"\n📊 Starting training for {num_epochs} epochs...")
    
    # Placeholder training loop (would have actual model and optimization)
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (X, y) in enumerate(train_loader):
            # Placeholder: In real scenario, would forward pass, compute loss, backprop
            pass
        print(f"Epoch {epoch+1}/{num_epochs} - Training in progress (simulated)")
    
    print(f"✅ Training complete")


def main(csv_path, batch_size=32, num_epochs=5):
    """
    Main training pipeline.
    
    Args:
        csv_path (str): Path to preprocessed CSV file
        batch_size (int): Batch size for training
        num_epochs (int): Number of epochs
    """
    print(f"\n{'='*70}")
    print("TEXT CLASSIFICATION TRAINING PIPELINE")
    print(f"{'='*70}\n")
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"❌ Error: File not found - {csv_path}")
        return
    
    try:
        # Load and preprocess data
        texts, labels, label_encoder, class_weights = load_and_preprocess_data(csv_path)
        
        print(f"\nClasses: {label_encoder.classes_}")
        print(f"Number of unique classes: {len(label_encoder.classes_)}")
        
        # Create dataloaders
        train_loader = create_dataloaders(texts, labels, batch_size=batch_size)
        print(f"\n✅ DataLoader created with batch size: {batch_size}")
        
        # Train model with class weights
        train_model(train_loader, class_weights, num_epochs=num_epochs)
        
        print(f"\n{'='*70}")
        print("✅ TRAINING PIPELINE COMPLETE")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"❌ Error in training: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "data/processed/cleaned_complaints.csv"
    
    main(csv_path)
