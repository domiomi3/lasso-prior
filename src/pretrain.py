import torch
from torch import nn
from torch.optim import AdamW
from pathlib import Path
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field

from tabpfn.model.loading import load_model_criterion_config
from src.data.load_datasets import create_dataloader

from src.utils.config import load_config, ModelConfig, OptimizerConfig, TrainingConfig
from src.utils.misc import set_seed


def load_model(config: ModelConfig, device: str):
    """Load pretrained TabPFN model."""
    valid_models = ["TabPFN-Wide-1.5k", "TabPFN-Wide-5k", "TabPFN-Wide-8k", "TabPFNv2"]
    assert config.model_name in valid_models, f"Invalid model: {config.model_name}"
    
    model, _, _ = load_model_criterion_config(
        model_path=None,
        check_bar_distribution_criterion=False,
        cache_trainset_representation=False,
        which='classifier',
        version='v2',
        download=True,
    )
    
    if config.model_name != "TabPFNv2":
        model.features_per_group = 1
        checkpoint_path = Path(f"./external/models/{config.model_name}_submission.pt")
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint)
        print(f"Loaded weights from {checkpoint_path.name}")
    
    return model.to(device)

#TODO: amp  

# def register_embedding_hook(model, layer_idx: int):
#     """
#     Register hook to extract embeddings from specified layer.
#     Returns a dict that will store embeddings.
#     """
#     embeddings = {'data': None}
    
#     def hook_fn(module, input, output):
#         embeddings['data'] = output.detach()
    
#     try:
#         layer = model.transformer_encoder.layers[layer_idx]
#         layer.register_forward_hook(hook_fn)
#         print(f"✓ Registered embedding hook at layer {layer_idx}")
#     except (AttributeError, IndexError) as e:
#         print(f"⚠ Could not register hook at layer {layer_idx}: {e}")
    
#     return embeddings


def set_optimizer_and_scheduler(decoder, config: OptimizerConfig, num_steps: int):
    optimizer = AdamW(
        decoder.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    if config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_steps
        )
    else:
        raise ValueError(f"Unknown scheduler: {config.scheduler}")
    
    return optimizer, scheduler


def prepare_batch(batch, device: str, train_test_split: float):
    X = batch['X'].to(device)
    y = batch['y'].to(device)
    lasso_coeffs = batch['lasso_coeffs'].to(device) # (batch_size, n_features)
    
    # Split into train/test
    batch_size, seq_len, n_features = X.shape
    train_size = int(seq_len * train_test_split)
    
    # TabPFN expects: (seq_len, batch_size, n_features)
    X_train = X[:, :train_size, :].transpose(0, 1).to(device)
    X_test = X[:, train_size:, :].transpose(0, 1).to(device)
    y_train = y[:, :train_size].transpose(0, 1).to(device)
    y_test = y[:, train_size:].transpose(0, 1).to(device)
    
    return X_train, y_train, X_test, y_test, lasso_coeffs


def save_checkpoint(path: Path, step: int, model, optimizer, scheduler, config: TrainingConfig):
    """Save training checkpoint."""
    checkpoint = {
        "step": step,
        "config": config,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }
    torch.save(checkpoint, path)
    print(f"✓ Saved checkpoint: {path}")


def load_checkpoint(path: str, model, optimizer, scheduler, device: str):
    """Load training checkpoint. Returns step number."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    step = checkpoint["step"]
    
    print(f"✓ Resumed from step {step}")
    return step


# ============================================================================
# Main Training Function
# ============================================================================

def train(config: TrainingConfig):
    """Main training loop."""
    # Setup
    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config.seed)
    
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"Model: {config.model.model_name}")
    print(f"Data path: {config.data_loader.data_dir}")
    print(f"Steps: {config.num_steps}")
    print(f"Device: {device}")
    print(f"Seed: {config.seed}")
    print(f"{'='*60}\n")
    
    # Load model
    print(f"Loading {config.model.model_name}...")
    model = load_model(config.model, device)
    model.train()
    print(f"✓ Model loaded")
    
    # Setup embedding extraction
    # embeddings = register_embedding_hook(model, config.model.embedding_layer)
    
    # Load data
    print(f"\nLoading data from {config.data_loader.data_dir}...")
    
    config.data_loader.__dict__.pop("train_test_split")
    train_loader = create_dataloader(**config.data_loader.__dict__)

    print(f"✓ DataLoader ready: {len(train_loader)} batches")
    
    # Create optimizer & scheduler
    optimizer, scheduler = set_optimizer_and_scheduler(
        model, config.optimizer, config.num_steps
    )
    
    # Loss function (placeholder until decoder is added)
    criterion = nn.MSELoss()
    
    # Resume if needed
    curr_step = 0
    if config.resume_from:
        curr_step = load_checkpoint(
            config.resume_from, model, optimizer, scheduler, device
        )
    
    # Create infinite data iterator
    def infinite_loader():
        while True:
            for batch in train_loader:
                yield batch
    
    data_iter = infinite_loader()
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training")
    print(f"{'='*60}\n")
    
    pbar = tqdm(
        range(curr_step, config.num_steps),
        desc="Training",
        initial=curr_step,
        total=config.num_steps
    )
    
    for step in pbar:
        # Get batch
        batch = next(data_iter)
        if batch is None:
            continue
        
        # Prepare data
        X_train, y_train, X_test, y_test, lasso_coeffs = prepare_batch(
            batch, device, config.data_loader.train_test_split
        )
        
        # Forward pass
        optimizer.zero_grad()
        
        try:
            # TabPFN forward
            pred_logits = model(
                train_x=X_train,
                train_y=y_train,
                test_x=X_test,
            )
            
            # Embeddings available in embeddings['data']
            # Shape: (N, M, K)
            # if embeddings['data'] is not None:
            #     # Your decoder would use embeddings['data'] here
            #     pass
            # PLACEHOLDER LOSS
            dummy_pred = pred_logits.mean()
            dummy_target = lasso_coeffs.mean()
            loss = criterion(dummy_pred, dummy_target)
            
            # Backward
            loss.backward()
            
            if config.optimizer.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.optimizer.gradient_clip
                )
            
            optimizer.step()
            scheduler.step()
            
            # embeddings['data'] = None  # Clear stored embeddings
            del pred_logits  # Delete intermediate tensors
            torch.cuda.empty_cache()  

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n⚠ OOM at step {step}, skipping...")
                torch.cuda.empty_cache()
                continue
            else:
                raise e
        
        # Detailed logging
        if step % config.log_interval == 0:
            print(f"\nStep {step}:")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
            # if embeddings['data'] is not None:
            #     print(f"  Embeddings shape: {embeddings['data'].shape}")
            print()
        
        # Checkpointing
        if step > 0 and step % config.save_interval == 0:
            save_checkpoint(
                checkpoint_dir / f"step_{step}.pt",
                step, model, optimizer, scheduler, config
            )
    
    # Final checkpoint
    save_checkpoint(
        checkpoint_dir / f"step_{config.num_steps}.pt",
        config.num_steps, model, optimizer, scheduler, config
    )
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")


# ============================================================================
# Usage
# ============================================================================

if __name__ == "__main__":
    config = load_config("configs/test.yaml")
    train(config.training)
    