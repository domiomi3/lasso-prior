import h5py
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from src.utils.config import load_config


class LassoPriorDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        max_sparsity_dataset: float = 0.9,
        max_sparsity_batch: float = 0.85,
        normalize_coeffs: bool = True,
        min_valid_datasets: int = 1,
        load_to_memory: bool = False,
        verbose: bool = True
    ):
        """
        Parameters
        ----------
        data_dir : str
            Directory containing *.h5 files
        max_sparsity_dataset : float
            Max sparsity for individual dataset (0.9 = skip if >90% zeros)
        max_sparsity_batch : float
            Max average sparsity for batch (0.85 = skip if >85% zeros)
        normalize_coeffs : bool
            Normalize coefficients to [0, 1] per dataset
        min_valid_datasets : int
            Skip the entire batch if fewer valid datasets
        load_to_memory : bool
            Load all valid batches into RAM (recommended for <10GB data)
        verbose : bool
            Print basic info about the batches
        """
        self.data_dir = Path(data_dir)
        self.max_sparsity_dataset = max_sparsity_dataset
        self.max_sparsity_batch = max_sparsity_batch
        self.normalize_coeffs = normalize_coeffs
        self.min_valid_datasets = min_valid_datasets
        
        self.batch_files = sorted(list(self.data_dir.glob("batch_*.h5")))
        
        if len(self.batch_files) == 0:
            raise ValueError(f"No batch files found in {self.data_dir}")
        
        if verbose:
            print(f"Found {len(self.batch_files)} batch files in {self.data_dir}")
        
        if load_to_memory: #load to RAM
            if verbose:
                print("Loading batches into RAM...")
            self._load_all_to_memory()
            if verbose:
                size_mb = sum(
                    d['X'].nbytes + d['y'].nbytes + d['lasso_coeffs'].nbytes 
                    for d in self.memory_cache
                ) / 1024 / 1024
                print(f"Loaded {len(self.memory_cache)} valid batches ({size_mb:.1f} MB)")
        else:
            self.memory_cache = None
    
    def _load_batch(self, batch_file):
        with h5py.File(batch_file, 'r') as f:
            num_datasets = len(f['datasets'])
            lasso_coeffs_all = f['lasso_coeffs'][:]
            
            X_list = []
            y_list = []
            coeffs_list = []
            
            for dataset_idx in range(num_datasets):
                X_list.append(f['datasets'][str(dataset_idx)]['X'][:])
                y_list.append(f['datasets'][str(dataset_idx)]['y'][:])
                
                coeffs = lasso_coeffs_all[dataset_idx]
                
                # normalize log coefficients for better signal
                epsilon = 1e-10
                log_coeffs = -np.log(coeffs + epsilon)
                log_min = log_coeffs.min()
                log_max = log_coeffs.max()
                
                if log_max > log_min:
                    log_norm = (log_coeffs - log_min) / (log_max - log_min)
                else:
                    log_norm = np.zeros_like(coeffs)
                
                coeffs_list.append(log_norm)
            
            X = np.stack(X_list)
            y = np.stack(y_list)
            lasso_coeffs = np.stack(coeffs_list)
        
        return {
            'X': X,
            'y': y,
            'lasso_coeffs': lasso_coeffs
        }
    
    def _load_all_to_memory(self):
        """Load all batches into RAM."""
        self.memory_cache = []
        
        for batch_file in self.batch_files:
            data = self._load_batch(batch_file)
            if data is not None:
                self.memory_cache.append(data)
    
    def __len__(self):
        """Return number of batches."""
        if self.memory_cache is not None:
            return len(self.memory_cache)
        return len(self.batch_files)
    
    def __getitem__(self, idx):
        """        
        Returns
        -------
        dict
            'X': torch.Tensor (batch_size, seq_len, n_features)
            'y': torch.Tensor (batch_size, seq_len)
            'lasso_coeffs': torch.Tensor (batch_size, n_features)
        """
        if self.memory_cache is not None: # load from ram
            data = self.memory_cache[idx]
        else: # load from disc
            batch_file = self.batch_files[idx]
            data = self._load_and_filter_batch(batch_file)
              
        return {
            'X': torch.from_numpy(data['X']).float(),
            'y': torch.from_numpy(data['y']).long(),
            'lasso_coeffs': torch.from_numpy(data['lasso_coeffs']).float()
        }


def collate_fn_skip_none(batch):
    """Skip None batches (filtered out)."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return batch[0] #unwrapping list


def create_dataloader(
    data_dir: str,
    max_sparsity_dataset: float = 0.95,
    max_sparsity_batch: float = 0.85,
    normalize_coeffs: bool = True,
    min_valid_datasets: int = 1,
    load_to_memory: bool = False,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    verbose: bool = True
):
    """    
    Parameters
    ----------
    data_dir : str
        Directory with batch_*.h5 files
    max_sparsity_dataset : float
        Skip datasets with sparsity > this (0.95 = skip if >95% zeros)
    max_sparsity_batch : float
        Skip batches with avg sparsity > this (0.85 = skip if >85% zeros)
    normalize_coeffs : bool
        Normalize coefficients to [0, 1] per dataset
    min_valid_datasets : int
        Skip batch if fewer valid datasets than this
    load_to_memory : bool
        Load all data to RAM (FAST but uses memory)
        Recommended if data < 10GB
    shuffle : bool
        Shuffle batches
    num_workers : int
        DataLoader workers (use 0 for HDF5)
    pin_memory : bool
        Pin memory for faster GPU transfer
    verbose : bool
        Print info
    
    Returns
    -------
    DataLoader
    """
    dataset = LassoPriorDataset(
        data_dir=data_dir,
        max_sparsity_dataset=max_sparsity_dataset,
        max_sparsity_batch=max_sparsity_batch,
        normalize_coeffs=normalize_coeffs,
        min_valid_datasets=min_valid_datasets,
        load_to_memory=load_to_memory,
        verbose=verbose
    )

    loader = DataLoader(
        dataset,
        batch_size=1,  #already a batch
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn_skip_none
    )
    
    return loader


if __name__ == "__main__":
    # quick check
    config = load_config(config_path="configs/test.yaml")
    config.training.data_loader.__dict__.pop("train_test_split")

    lasso_prior_dataloader = create_dataloader(**config.training.data_loader.__dict__)
    print(f"\nDataLoader ready: {len(lasso_prior_dataloader)} batches\n")
    
    for i, batch in enumerate(lasso_prior_dataloader):      
        X = batch['X']
        y = batch['y']
        lasso_coeffs = batch['lasso_coeffs']
        
        print(f"Batch {i}:")
        print(f"  X: {X.shape}")
        print(f"  y: {y.shape}")
        print(f"  lasso_coeffs: {lasso_coeffs.shape}")
        print(f"  Coeff mean: [{lasso_coeffs.mean():.3f}]")
        
        if i >= 2:
            break