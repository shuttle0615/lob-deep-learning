import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from loaders.fi2010_loader import Dataset_fi2010
from loaders.binance_loader import create_binance_dataloader

def load_and_compare_samples():
    """Load and compare samples from both datasets"""
    print("\n=== Initializing Datasets ===")
    
    # Initialize FI-2010 dataset
    fi2010_dataset = Dataset_fi2010(
        auction=False,
        normalization='Zscore',
        stock_idx=[0],
        days=[1],
        T=100,          # window size
        k=0,           # prediction horizon
        lighten=False  # use all price levels
    )
    
    # Initialize Binance dataset
    binance_loader = create_binance_dataloader(
        data_dir="/Users/igyuho/Desktop/LOB-data/processed_data",
        batch_size=1,
        window_size=100,  # match FI-2010
        prediction_horizon=120,
        n_levels=10,     # match FI-2010's 10 levels
        threshold=0.0005,
        shuffle=False,
        num_workers=2    # Set to 0 for debugging
    )
    
    # Get single samples from each
    fi2010_features, fi2010_label = next(iter(DataLoader(fi2010_dataset, batch_size=1)))
    binance_features, binance_label = next(iter(binance_loader))
    

    print(f"FI-2010 features shape: {fi2010_features.shape}")
    print(f"Binance features shape: {binance_features.shape}")
    print(f"FI-2010 label: {fi2010_label.item()}")
    print(f"Binance label: {binance_label.item()}")

    # Remove batch dimension
    fi2010_features = fi2010_features.squeeze(0)
    binance_features = binance_features.squeeze(0)
    
    print("\n=== Shape Comparison ===")
    print(f"FI-2010 features shape: {fi2010_features.shape}")
    print(f"Binance features shape: {binance_features.shape}")
    print(f"FI-2010 label: {fi2010_label.item()}")
    print(f"Binance label: {binance_label.item()}")
    
    return fi2010_features, binance_features, fi2010_label, binance_label

def visualize_comparison(fi2010_features, binance_features):
    """Create visualizations to compare the two datasets"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot FI-2010 data
    im1 = ax1.imshow(fi2010_features.T, aspect='auto', cmap='RdYlBu')
    ax1.set_title('FI-2010 LOB Data')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Feature Index')
    plt.colorbar(im1, ax=ax1)
    
    # Plot Binance data
    im2 = ax2.imshow(binance_features.T, aspect='auto', cmap='RdYlBu')
    ax2.set_title('Binance LOB Data')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Feature Index')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.show()

def compare_feature_distributions(fi2010_features, binance_features):
    """Compare the statistical distributions of features"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # FI-2010 distribution
    ax1.hist(fi2010_features.numpy().flatten(), bins=50, alpha=0.7)
    ax1.set_title('FI-2010 Feature Distribution')
    ax1.set_xlabel('Feature Value')
    ax1.set_ylabel('Count')
    
    # Binance distribution
    ax2.hist(binance_features.numpy().flatten(), bins=50, alpha=0.7)
    ax2.set_title('Binance Feature Distribution')
    ax2.set_xlabel('Feature Value')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    plt.show()

def print_feature_statistics(fi2010_features, binance_features):
    """Print statistical information about the features"""
    print("\n=== Feature Statistics ===")
    
    fi2010_stats = {
        'mean': np.mean(fi2010_features.numpy()),
        'std': np.std(fi2010_features.numpy()),
        'min': np.min(fi2010_features.numpy()),
        'max': np.max(fi2010_features.numpy())
    }
    
    binance_stats = {
        'mean': np.mean(binance_features.numpy()),
        'std': np.std(binance_features.numpy()),
        'min': np.min(binance_features.numpy()),
        'max': np.max(binance_features.numpy())
    }
    
    print("\nFI-2010 Statistics:")
    for key, value in fi2010_stats.items():
        print(f"{key}: {value:.4f}")
    
    print("\nBinance Statistics:")
    for key, value in binance_stats.items():
        print(f"{key}: {value:.4f}")

def main():
    # Load and compare samples
    fi2010_features, binance_features, fi2010_label, binance_label = load_and_compare_samples()
    
    # Visualize the data
    visualize_comparison(fi2010_features, binance_features)
    
    # Compare distributions
    compare_feature_distributions(fi2010_features, binance_features)
    
    # Print statistics
    print_feature_statistics(fi2010_features, binance_features)

if __name__ == "__main__":
    main() 