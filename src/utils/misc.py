import torch
import numpy as np

def set_seed(seed: int):
    """Set global seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def generate_exp_name_from_config(config):
    name_parts = [
        f"bs{config.prior.batch_size}",
        f"steps{config.prior.num_steps}",
        f"feat{config.prior.min_features}-{config.prior.max_features}",
        f"aug{config.feature_adding.add_features_max}",
        f"boot{config.feature_selection.n_bootstrap}",
    ]
    return "_".join(name_parts)
