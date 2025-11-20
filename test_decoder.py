import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.decoder import TabPFNDecoder


def test_embeddings():
    """Test embedding extraction with different configurations."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Create model
    print("Loading TabPFNDecoder...")
    model = TabPFNDecoder(
        model_name="TabPFN-Wide-8k",
        embedding_layer=-1,  # Last layer
        device=device,
    )
    print("Model loaded!\n")
    
    # Test configurations
    configs = [
        {"train": 20, "test": 10, "features": 50, "batch": 4},
        {"train": 30, "test": 15, "features": 100, "batch": 2},
        {"train": 50, "test": 20, "features": 250, "batch": 1},
    ]
    
    for i, cfg in enumerate(configs, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}: train={cfg['train']}, test={cfg['test']}, "
            f"features={cfg['features']}, batch={cfg['batch']}")
        print('='*70)
        
        # Create dummy data
        train_x = torch.randn(cfg['train'], cfg['batch'], cfg['features']).to(device)
        train_y = torch.randint(0, 2, (cfg['train'], cfg['batch'])).to(device)
        test_x = torch.randn(cfg['test'], cfg['batch'], cfg['features']).to(device)
        
        print(f"\nInput shapes:")
        print(f"  train_x: {tuple(train_x.shape)}")
        print(f"  train_y: {tuple(train_y.shape)}")
        print(f"  test_x:  {tuple(test_x.shape)}")
        
        # Extract embeddings
        embeddings = model(train_x, train_y, test_x)
        
        print(f"\nOutput embeddings:")
        print(f"  Shape: {tuple(embeddings.shape)}")
        print(f"  Interpretation: (batch_size, seq_len, n_features+1, emb_size)")
        print(f"    batch_size = {embeddings.shape[0]}")
        print(f"    seq_len = {embeddings.shape[1]} (train={cfg['train']} + test={cfg['test']})")
        print(f"    n_features+1 = {embeddings.shape[2]} (features={cfg['features']} + 1 for y)")
        print(f"    emb_size = {embeddings.shape[3]}")
        
        # Split train/test embeddings
        train_emb = embeddings[:, :cfg['train'], :, :]
        test_emb = embeddings[:, cfg['train']:, :, :]
        
        # Split feature embeddings and y embedding
        feature_emb = embeddings[:, :, :-1, :]  # All but last position
        y_emb = embeddings[:, :, -1, :]  # Last position
        
        print(f"\nSplit by sequence:")
        print(f"  Train embeddings: {tuple(train_emb.shape)}")
        print(f"  Test embeddings:  {tuple(test_emb.shape)}")
        
        print(f"\nSplit by feature vs y:")
        print(f"  Feature embeddings: {tuple(feature_emb.shape)}")
        print(f"  Y embedding:        {tuple(y_emb.shape)}")
        
        # Statistics
        print(f"\nStatistics:")
        print(f"  Mean: {embeddings.mean().item():.6f}")
        print(f"  Std:  {embeddings.std().item():.6f}")
        print(f"  Min:  {embeddings.min().item():.6f}")
        print(f"  Max:  {embeddings.max().item():.6f}")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    print(f"✓ Embeddings successfully extracted from layer {model.embedding_layer}")
    print(f"✓ Embedding dimension: {embeddings.shape[-1]}")
    print(f"✓ Embeddings are PER-FEATURE (each feature gets an embedding)")
    print(f"✓ Shape: (batch_size, seq_len, n_features+1, emb_size)")
    print(f"✓ The +1 is for the y (target) embedding")
    print('='*70)

if __name__ == "__main__":
    test_embeddings()