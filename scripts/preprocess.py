"""
Text preprocessing and class balancing module for traffic complaints.
"""

import textacy.preprocessing
import unidecode
import pandas as pd
import numpy as np
import re
import os
from sklearn.utils.class_weight import compute_class_weight


def normalize_text(text):
    """
    Normalize text by:
    - Converting to lowercase
    - Removing extra whitespace
    - Decoding non-ASCII characters (Hinglish)
    - Removing special characters but keeping alphanumeric and spaces
    
    Args:
        text (str): Raw text to normalize
        
    Returns:
        str: Normalized text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace using textacy
    text = textacy.preprocessing.normalize.whitespace(text)
    
    # Decode non-ASCII characters (Hinglish, accents, etc.)
    text = unidecode.unidecode(text)
    
    # Remove special characters but keep alphanumeric and spaces
    text = re.sub(r"[^a-z0-9\s]", "", text)
    
    # Remove extra spaces that may have been created
    text = " ".join(text.split())
    
    return text


def compute_class_weights(labels):
    """
    Compute class weights for imbalanced datasets using sklearn's balanced strategy.
    
    Args:
        labels (array-like): Array of class labels
        
    Returns:
        dict: Dictionary mapping class labels to their weights
    """
    unique_classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=labels
    )
    class_weights = {cls: weight for cls, weight in zip(unique_classes, weights)}
    return class_weights


def main(csv_path):
    """
    Main preprocessing pipeline:
    - Read CSV with 'text' and 'label' columns
    - Normalize all text
    - Compute class weights
    - Save cleaned CSV to data/processed/
    
    Args:
        csv_path (str): Path to input CSV file
    """
    print(f"\n{'='*70}")
    print(f"Processing file: {csv_path}")
    print(f"{'='*70}\n")
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"❌ Error: File not found - {csv_path}")
        return
    
    try:
        # Read CSV
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} records from {csv_path}")
        
        # Check required columns
        if 'text' not in df.columns or 'label' not in df.columns:
            print(f"❌ Error: CSV must contain 'text' and 'label' columns")
            print(f"   Available columns: {list(df.columns)}")
            return
        
        # Show before/after examples
        print(f"\n{'─'*70}")
        print("BEFORE/AFTER NORMALIZATION EXAMPLES:")
        print(f"{'─'*70}")
        for i, text in enumerate(df['text'].head(3)):
            normalized = normalize_text(text)
            print(f"\nExample {i+1}:")
            print(f"  BEFORE:  {text[:80]}")
            print(f"  AFTER:   {normalized[:80]}")
        
        # Normalize all text
        print(f"\n{'─'*70}")
        print("Normalizing all text...")
        df['text'] = df['text'].apply(normalize_text)
        print("✅ Text normalization complete")
        
        # Compute class weights
        print(f"\n{'─'*70}")
        print("COMPUTING CLASS WEIGHTS:")
        print(f"{'─'*70}")
        class_weights = compute_class_weights(df['label'].values)
        
        print(f"\nClass distribution:")
        label_counts = df['label'].value_counts()
        for label, count in label_counts.items():
            print(f"  {label}: {count} samples (weight: {class_weights[label]:.4f})")
        
        total_samples = len(df)
        print(f"\nTotal samples: {total_samples}")
        
        # Create output directory if it doesn't exist
        output_dir = "data/processed"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save cleaned CSV
        output_path = os.path.join(output_dir, "cleaned_complaints.csv")
        df.to_csv(output_path, index=False)
        print(f"\n{'─'*70}")
        print(f"✅ Cleaned data saved to: {output_path}")
        print(f"{'─'*70}\n")
        
        # Return class weights for use in training
        return class_weights
        
    except Exception as e:
        print(f"❌ Error processing file: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    import glob
    
    if len(sys.argv) > 1:
        # Process all matching CSV files
        pattern = sys.argv[1]
        files = glob.glob(pattern)
        
        if not files:
            print(f"⚠️  No files matching pattern: {pattern}")
        else:
            print(f"Found {len(files)} file(s) to process")
            for csv_file in files:
                compute_class_weights(pd.read_csv(csv_file)['label'].values)
                main(csv_file)
    else:
        print("Usage: python preprocess.py <csv_path_or_pattern>")
        print("Example: python preprocess.py data/raw/complaints_*.csv")
