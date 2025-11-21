import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.decoder import TabPFNFeatureSelector


def test_embeddings(model):
    """Test embedding extraction with different configurations."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    
    # Test configurations
    configs = [
        {"train": 20, "test": 10, "features": 50, "batch": 4},
        # {"train": 30, "test": 15, "features": 100, "batch": 2},
        # {"train": 50, "test": 20, "features": 250, "batch": 1},
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

        return embeddings

if __name__ == "__main__": 
    model = TabPFNFeatureSelector(
        model_name="TabPFN-Wide-5k",
        embedding_layer=4,  # Last layer
        device="cuda",
    )
    model.load_decoder_checkpoint("/work/dlclarge2/matusd-tab_priors/checkpoints/train_lasso_4000_20251121_134649/best_model.pt")
    emb=test_embeddings(model)
    model2 = TabPFNFeatureSelector(
        model_name="TabPFN-Wide-5k",
        embedding_layer=4,  # Last layer
        device="cuda",
    )
    emb2=test_embeddings(model2)
    breakpoint()