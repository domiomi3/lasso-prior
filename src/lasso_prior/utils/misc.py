import torch
import logging
import sys
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


def setup_logger(
    name: str = __name__,
    level: int = logging.INFO,
    format_string: str = '[%(asctime)s][%(levelname)s] %(message)s',
    date_format: str = '%Y-%m-%d %H:%M:%S'
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    logger.handlers.clear()
    
    formatter = logging.Formatter(fmt=format_string, datefmt=date_format)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger(__name__)
import psutil

def log_mem():
    logger.info(f"CPU Memory Used: {psutil.virtual_memory().percent}%")
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory
        logger.info(f"Total GPU memory: {total_mem / 1e9:.2f} GB")
        logger.info(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        logger.info(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

