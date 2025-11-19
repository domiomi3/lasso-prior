"""
Generate and save datasets generates from TabPFN-wide priors extended by Lasso coefficients.
"""
import pandas as pd
import h5py
import numpy as np
import argparse

from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from tabularpriors.dataloader import TabICLPriorDataLoader

from src.data.lasso_prior import generate_lasso_batches
from src.utils.config import load_config
from src.utils.misc import generate_exp_name_from_config, setup_logger

logger = setup_logger(__name__)

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

    logger.info("="*60)
    logger.info(f"Experiment: {exp_dir}")
    logger.info("="*60)
    logger.info(f"Batches: {num_batches} ")
    logger.info(f"Batch size: {config.prior.batch_size}")
    logger.info(f"Samples: {config.prior.num_datapoints_min}-{config.prior.num_datapoints_max}")
    logger.info(f"Base features: {config.prior.min_features}-{config.prior.max_features}")
    logger.info(f"Extra features added: {config.feature_adding.add_features_min}-{config.feature_adding.add_features_max}")
    logger.info(f"Classes: 2-{config.prior.max_num_classes}")   
    logger.info(f"Lasso bootstrap iterations: {config.feature_selection.n_bootstrap}")
    logger.info("="*60)
    logger.info(f"Config file: {config_path}")
    logger.info(f"Output dir: {output_dir}")
    logger.info("="*60)

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
        logger.info(f"Avg batch sparsity: {sum(sparsity_batch)/len(sparsity_batch)*100:.2f} %")  

        if (batch_id + 1) % 10 == 0: # incremental saving
            df = pd.DataFrame(metadata_records)
            df.to_csv(metadata_path, index=False)
            logger.info(f"Saved metadata ({len(metadata_records)} datasets)")

    # save data metadata 
    df = pd.DataFrame(metadata_records)
    df.to_csv(metadata_path, index=False)
    logger.info(f"\n{'='*60}")
    logger.info(f"Saved metadata to {metadata_path}")
    logger.info(f"Saved {num_batches} batches to {output_dir}")
    
    # save experiment metadata 
    try:
        config.to_yaml(str(output_dir / "config.yaml"))
        logger.info(f"Saved config to {output_dir / 'config.yaml'}")
    except Exception as e:
        logger.info(f"Warning: Could not save config.yaml: {e}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Generation complete")
    logger.info(f"Location: {output_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('--config_path', type=str, default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config_path)
    
    generate_datasets(config, args.config_path)
