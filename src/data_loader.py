"""
Data splitting and loading module.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from src.config import TRAIN_CSV, RESULTS_DIR, SEED, CLASS_NAMES

def split_dataset():
    """
    Performs a 70/15/15 stratified split of the original dataset.
    Saves the splits as train.csv, val.csv, and test.csv in the results folder.
    Plots and prints the per-class counts for verification.
    """
    if not os.path.exists(TRAIN_CSV):
        print(f"Error: {TRAIN_CSV} not found. Please ensure the dataset is downloaded to the correct path.")
        return
        
    df = pd.read_csv(TRAIN_CSV)
    
    # First split: 70% Train, 30% Temp (which will be split 50/50 for Val/Test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.30, 
        random_state=SEED, 
        stratify=df['diagnosis']
    )
    
    # Second split: Split Temp into 15% Val and 15% Test
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.50, 
        random_state=SEED, 
        stratify=temp_df['diagnosis']
    )
    
    # --- Data Leakage Check ---
    # Asserting that no image ID appears in more than one split
    train_ids = set(train_df['id_code'])
    val_ids = set(val_df['id_code'])
    test_ids = set(test_df['id_code'])
    
    assert len(train_ids.intersection(val_ids)) == 0, "Leakage detected between Train and Validation!"
    assert len(train_ids.intersection(test_ids)) == 0, "Leakage detected between Train and Test!"
    assert len(val_ids.intersection(test_ids)) == 0, "Leakage detected between Validation and Test!"
    
    print("Assertion Passed: No image ID appears in more than one split (No Data Leakage).")
    
    # Save the splits to the results folder
    train_df.to_csv(os.path.join(RESULTS_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(RESULTS_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(RESULTS_DIR, "test.csv"), index=False)
    print(f"Splits saved to {RESULTS_DIR}")
    
    # Print Stratification Stats to prove proportions are maintained
    splits = {'Train (70%)': train_df, 'Val (15%)': val_df, 'Test (15%)': test_df}
    
    print("\n--- Stratification Statistics ---")
    plot_data = []
    
    for split_name, split_df in splits.items():
        counts = split_df['diagnosis'].value_counts().sort_index()
        total = len(split_df)
        
        print(f"\n{split_name} - Total Images: {total}")
        for idx, count in counts.items():
            pct = (count / total) * 100
            print(f"  Class {idx} ({CLASS_NAMES[idx]}): {count} ({pct:.1f}%)")
            # Collect data for plotting
            plot_data.append({'Split': split_name, 'Diagnosis': CLASS_NAMES[idx], 'Count': count})
            
    # Plot the distributions
    plot_df = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x='Diagnosis', y='Count', hue='Split', palette='Set2')
    plt.title("Stratified Splitting: Class Distribution Across Splits")
    plt.ylabel("Number of Images")
    plt.xlabel("Diagnosis Class")
    plt.xticks(rotation=45)
    plt.legend(title='Dataset Split')
    plt.tight_layout()
    
    # Save the plot alongside the EDA plots
    eda_dir = os.path.join(RESULTS_DIR, "eda")
    os.makedirs(eda_dir, exist_ok=True)
    plot_path = os.path.join(eda_dir, "stratified_splits.png")
    
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"\nStratification plot saved to {plot_path}")

if __name__ == "__main__":
    split_dataset()
