"""
Logic for Lasso-based feature importance. Datasets generated from TabPFN-wide prior
are fit multiple times on Logistic Regression with L1 regularization to rank features
based on the averaged coefficients. 
"""
import numpy as np
import torch
import argparse

from torch import nn
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

from tabicl.train.run import Timer
from tabularpriors.dataloader import TabICLPriorDataLoader

from src.utils.config import load_config
from src.utils.misc import setup_logger

logger = setup_logger(__name__)

def get_lasso_coefficients(X, y, config, verbose=True):
    """
    Returns Lasso coefficients averaged over samples and repetitions.
    These will be used as ground truth labels for pretraining a decoder outputing feature importance.
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    config : Config 
    
    Returns
    -------
    coeff_importance_avg : np.ndarray, shape (n_features,)
        L1/L2 norm of Lasso coefficients averaged over samples and repetitions on the same dataset.
    metadata: dict, 
        Contains dataset's sparsity level and sampled C.
    """
    if config.normalize:
        X = StandardScaler().fit_transform(X)
    
    _, n_features = X.shape
    coeff_importance = np.zeros(n_features)

    C_min = config.C_min 
    C_max = config.C_max  
    C_sampled = np.exp(np.random.uniform(np.log(C_min), np.log(C_max)))
    
    if verbose:
        logger.info(f"Using C={C_sampled:.4f}")
    
    # check if enough samples per class for stratification
    class_vals, class_counts = np.unique(y, return_counts=True)
    n_classes = len(class_vals)
    is_multiclass = n_classes > 2
    min_class_count = class_counts.min()
    
    if min_class_count < 2:
        logger.info(f"Warning: Some classes have fewer than 2 samples. Skipping this dataset.")
        return coeff_importance # zeros
    
    #subsets contain all classes
    splitter = StratifiedShuffleSplit(
        n_splits=config.n_bootstrap,
        test_size=0.5, # split datasets in half (original paper)
        random_state=config.random_state
    )
    
    successful_fits = 0
    for subset1_idx, subset2_idx in splitter.split(X, y):
        for idx in [subset1_idx, subset2_idx]:
            X_sub, y_sub = X[idx], y[idx]
            
            if len(np.unique(y_sub)) < 2:
                continue
            
            model = LogisticRegression(
                penalty="l1", 
                solver="saga", 
                max_iter=config.max_iter,
                warm_start=True, 
                random_state=config.random_state,
                multi_class='ovr',
                tol=config.tolerance,
                C=C_sampled
            )
            model.fit(X_sub, y_sub)
            
            if is_multiclass:
                # L2 norm 
                coeff_importance += np.linalg.norm(model.coef_, axis=0, ord=2)
            else:
                # L1 norm
                coeff_importance += np.abs(model.coef_.ravel())
            
            successful_fits += 1
    
    if successful_fits == 0:
        logger.info("Warning: No successful fits for this dataset.")
        return coeff_importance # zeros
    
    coeff_importance_avg = coeff_importance / successful_fits
    sparsity = np.sum(coeff_importance_avg == 0) / len(coeff_importance_avg)
    if verbose:
        logger.info(f"Dataset sparsity: {sparsity*100:.2f}%")
    
    return coeff_importance_avg, {"sparsity": sparsity, "C": C_sampled}


# ----------------from tabpfn-wide-------------------------
def get_feature_adding_parameters(config, current_step):
    if config.warmup_steps > 0 and current_step < config.warmup_steps:
        max_features_add = config.add_features_min + \
        (config.add_features_max - config.add_features_min) * \
        (current_step / config.warmup_steps)
    else:
        max_features_add = config.add_features_max
    new_features = np.random.randint(config.add_features_min, max_features_add + 1)
    sparsity = np.random.uniform(config.min_sparsity, config.max_sparsity)
    noise = np.random.uniform(config.min_noise, config.max_noise)
    return new_features, sparsity, noise


def get_feature_dependent_noise(x_tensor, std):
    # The noise std is proportional to the standard deviation of each feature
    stds = x_tensor.std(dim=0, keepdim=True)
    stds[stds == 0] = 1  # Avoid division by zero
    noise = torch.randn_like(x_tensor) * (std * stds)
    return noise


def get_linear_added_features(x, features_to_be_added, sparsity, noise_std):
    """
    Adds new linear features to the input tensor with controlled sparsity and feature-dependent noise.
    """
    W_sparse =  nn.Linear(x.shape[-1], features_to_be_added, bias=False)
    W_sparse.weight.data *= (torch.rand_like(W_sparse.weight) < sparsity).float()
    x = W_sparse(x)
    
    dependent_noise = get_feature_dependent_noise(x, noise_std)
    x += dependent_noise
    return x.detach()


def get_new_features(x_tensor, features_to_be_added, sparsity=0.01, noise_std=3, include_original=True, include_original_prob=0.5):
    x_new = get_linear_added_features(x_tensor, features_to_be_added, sparsity=sparsity, noise_std=noise_std)
    if np.random.rand() < include_original_prob and include_original:
        x_new = torch.cat([x_tensor, x_new], dim=-1)
        x_new = x_new[..., torch.randperm(x_new.shape[-1])]
    return x_new.detach()
# --------------------------------------------------------

def generate_lasso_batches(prior_dataloader, feature_adding_config, feature_selection_config, n_batches):
    """
    Generate a batch with Lasso coefficients from TabPFN-wide priors.
    
    Parameters
    ----------
    prior_dataloader : DataLoader
        Dataloader for TabICL prior-generated datasets
    feature_selection_config : Config
        Config for feature selection algorithm
    feature_adding_config: Config
        Config for TabPFN-wide's feature adding process
    n_batches : int
        Number of batches of datasets to generate
    
    Yields
    ------
    X : np.ndarray, shape (batch_size, seq_len, n_features)
    y : np.ndarray, shape (batch_size, seq_len,)
    lasso_coeffs : np.ndarray, shape (batch_size, n_features,)
    metadata: dict (batch_size, 2)
    """
   
    with Timer() as timer:
        prior_loader = iter(prior_dataloader)
    prior_time = timer.elapsed
    logger.info(f"Loading prior loader: {prior_time:.2f}s")


    for i in range(n_batches): #must be equal to num_steps from the prior config
        current_step = i
        logger.info("="*60)
        logger.info(f"Batch {i+1}")
        logger.info("="*60)
        batch = next(prior_loader)
        X, y = batch['x'], batch['y']
        X = torch.Tensor(X) # B, N, M
        y = torch.Tensor(y) # B, N
        batch_size = X.shape[0]

        if feature_adding_config.add_features_max > 0: #that's how we can control expanding feature dim
            new_features, sparsity, noise = get_feature_adding_parameters(feature_adding_config, current_step)
            X = get_new_features(X, new_features, sparsity=sparsity, noise_std=noise)
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}, classes: {len(np.unique(y))}")
  
        X = X.numpy()
        y = y.numpy()
        valid_X = []
        valid_y = []
        valid_coeffs = []
        valid_metadata = []

        for b in range(batch_size):
            with Timer() as timer:
                X_dataset = X[b]  # (seq_len, n_features)
                y_dataset = y[b]  # (seq_len,)
                
                lasso_coeffs, metadata = get_lasso_coefficients(
                    X_dataset, y_dataset, feature_selection_config
                )

                # filter out datasets with very high sparsity (no signal to learn)
                sparsity = metadata['sparsity']
                if sparsity > feature_selection_config.max_sparsity:
                    logger.info(f"Dataset {b} removed: sparsity={sparsity*100:.2f}%)")
                    continue
                                
                valid_X.append(X_dataset)
                valid_y.append(y_dataset)
                valid_coeffs.append(lasso_coeffs)
                valid_metadata.append(metadata)

        min_required = max(1, batch_size // 2)  # At least 50% of original batch_size
        
        if len(valid_X) < min_required:
            logger.info(f"Batch {i} skipped - only {len(valid_X)}/{batch_size} valid datasets")
            continue
        
        X_valid = np.stack(valid_X)
        y_valid = np.stack(valid_y)
        coeffs_valid = np.stack(valid_coeffs) # (batch_size, n_features)
        
        yield X_valid, y_valid, coeffs_valid, valid_metadata


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('--config_path', type=str, default="configs/default.yaml")
    args = parser.parse_args()

    # quick check 
    config = load_config(args.config_path)
    prior_loader = TabICLPriorDataLoader(**config.prior.__dict__)

    for i, (X_batch, y_batch, lasso_coeffs_batch, metadata_batch) in enumerate(
        generate_lasso_batches(
            prior_dataloader=prior_loader,  
            feature_adding_config=config.feature_adding,
            feature_selection_config=config.feature_selection,
            n_batches=config.prior.num_steps
        )
    ):
        sparsity_batch = [i["sparsity"] for i in metadata_batch]
        logger.info(f"Avg batch sparsity: {sum(sparsity_batch)/len(sparsity_batch)*100:.2f} %")  
