"""
Generate and save datasets generates from TabPFN-wide priors extended by Lasso coefficients.
"""
import pandas as pd
import h5py
import numpy as np

from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from tabularpriors.dataloader import TabICLPriorDataLoader

from src.data.lasso_prior import generate_lasso_batches
from src.utils.config import load_config
from src.utils.misc import generate_exp_name_from_config


def save_batch(X, y, lasso_coeffs, batch_id, output_dir):
    """
    Save a single batch to a HDF5 file with structure:
    batch_000001.h5
        /datasets/
            /0/X  (seq_len, n_features)
            /0/y  (seq_len,)
            /1/X
            /1/y
            ...
        /lasso_coeffs  (batch_size, n_features)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = output_dir / f"batch_{batch_id:06d}.h5"
    batch_size = X.shape[0]
    
    with h5py.File(save_path, 'w') as f:
        datasets_group = f.create_group('datasets')
        
        for dataset_idx in range(batch_size):
            dataset_group = datasets_group.create_group(str(dataset_idx))
            dataset_group.create_dataset(
                'X', 
                data=X[dataset_idx],
                compression='gzip',
                compression_opts=4
            )
            dataset_group.create_dataset(
                'y',
                data=y[dataset_idx],
                compression='gzip',
                compression_opts=4
            )
        
        f.create_dataset(
            'lasso_coeffs',
            data=lasso_coeffs,
            compression='gzip',
            compression_opts=4
        )


def generate_datasets(config, config_path):
    """
    Create a prior dataloader, generate datasets, and save to disk.
        
    Parameters
    ----------
    config : Config
        Configuration object with all hyperparameters
    config_path : str
        Path to the config file
    """
    if config.data_generation.exp_name:
        exp_name = config.data_generation.exp_name
    else:
        exp_name = generate_exp_name_from_config(config)
    
    if config.data_generation.add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = f"{exp_name}_{timestamp}"
    else:
        exp_dir = exp_name
    
    output_dir = Path(config.data_generation.output_dir) / exp_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_batches = config.prior.num_steps

    print("="*60)
    print(f"Experiment: {exp_dir}")
    print("="*60)
    print(f"Batches: {num_batches} ")
    print(f"Batch size: {config.prior.batch_size}")
    print(f"Samples: {config.prior.num_datapoints_min}-{config.prior.num_datapoints_max}")
    print(f"Base features: {config.prior.min_features}-{config.prior.max_features}")
    print(f"Extra features added: {config.feature_adding.add_features_min}-{config.feature_adding.add_features_max}")
    print(f"Classes: 2-{config.prior.max_num_classes}")   
    print(f"Lasso bootstrap iterations: {config.feature_selection.n_bootstrap}")
    print("="*60)
    print(f"Config file: {config_path}")
    print(f"Output dir: {output_dir}")
    print("="*60)

    prior_loader = TabICLPriorDataLoader(**config.prior.__dict__)
    
    generator = generate_lasso_batches(
        prior_dataloader=prior_loader,
        feature_adding_config=config.feature_adding,
        feature_selection_config=config.feature_selection,
        n_batches=num_batches
    )
    
    metadata_records = []
    metadata_path = output_dir / "metadata.csv"
    
    # save data
    for batch_id, (X, y, lasso_coeffs, metadata) in enumerate(tqdm(generator, total=num_batches, desc="Generating batches")):
        save_batch(X, y, lasso_coeffs, batch_id, output_dir)
        
        for dataset_id in range(X.shape[0]):
            n_samples, n_features = X[dataset_id].shape
            n_classes = len(np.unique(y[dataset_id]))
            coeffs = lasso_coeffs[dataset_id]
            
            record = {
                'batch_id': batch_id,
                'dataset_id': dataset_id,
                'n_samples': n_samples,
                'n_features': n_features,
                'n_classes': n_classes,
                'sparsity': metadata[dataset_id]['sparsity'],
                'C_sampled': metadata[dataset_id]['C'],
                'max_coeff': coeffs.max(),
                'mean_coeff': coeffs.mean(),
                'median_coeff': np.median(coeffs),
            }
            metadata_records.append(record)
        
        sparsity_batch = [i["sparsity"] for i in metadata]
        print(f"Avg batch sparsity: {sum(sparsity_batch)/len(sparsity_batch)*100:.2f} %")  

        if (batch_id + 1) % 10 == 0: # incremental saving
            df = pd.DataFrame(metadata_records)
            df.to_csv(metadata_path, index=False)
            print(f"Saved metadata ({len(metadata_records)} datasets)")

    # save data metadata 
    df = pd.DataFrame(metadata_records)
    df.to_csv(metadata_path, index=False)
    print(f"\n{'='*60}")
    print(f"Saved metadata to {metadata_path}")
    print(f"Saved {num_batches} batches to {output_dir}")
    
    # save experiment metadata 
    try:
        config.to_yaml(str(output_dir / "config.yaml"))
        print(f"Saved config to {output_dir / 'config.yaml'}")
    except Exception as e:
        print(f"Warning: Could not save config.yaml: {e}")
    
    print(f"\n{'='*60}")
    print(f"Generation complete")
    print(f"Location: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    config_path = "configs/test.yaml"
    config = load_config(config_path)
    
    generate_datasets(config, config_path)
