"""
Configuration management with YAML and dataclasses.
"""
import yaml
from dataclasses import dataclass, asdict
from typing import Literal, Optional


@dataclass
class FeatureSelectionConfig:
    method: Literal["lasso", "elastic_net"] = "lasso"
    n_bootstrap: int = 100
    max_iter: int = 10000
    normalize: bool = True
    random_state: int = 42
    tolerance: float = 0.001
    C_min: float = 0.001
    C_max: float = 0.5
    max_sparsity: float= 0.90 


@dataclass
class TabICLPriorDataLoaderConfig:
    num_steps: int = 1
    batch_size: int = 4
    device: str = "cpu"
    min_features: int = 50
    max_features: int = 250
    max_num_classes: int = 10
    num_datapoints_min: int = 40
    num_datapoints_max: int = 300


@dataclass
class FeatureAddingConfig:
    add_features_min: int = 200
    add_features_max: int = 1500
    warmup_steps: int = 0
    min_sparsity: float = 0.0
    max_sparsity: float = 0.05
    min_noise: float = 0.0
    max_noise: float = 1.0


@dataclass
class DataGenerationConfig:
    output_dir: str = "data/generated"
    exp_name: str = "lasso_prior"
    add_timestamp: bool = True


@dataclass
class DataLoadingConfig:
    data_dir: str = "experiments/data/generated/test_final_20251114_202322"
    max_sparsity_dataset: float = 0.95
    max_sparsity_batch: float = 0.85
    normalize_coeffs: bool = True
    min_valid_datasets: int = 1
    load_to_memory: bool = True
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = True
    verbose: bool = True
    train_test_split: float = 0.8


@dataclass
class ModelConfig:
    model_name: str = "TabPFN-Wide-8k"
    model_path: str = "./models"
    embedding_layer: int = 4


@dataclass
class OptimizerConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    scheduler: str = "cosine"

@dataclass
class WandBConfig:
    use_wandb: True
    project: str = "test"
    entity: str = None
    run_name: str = None

@dataclass
class TrainingConfig:
    data_loader: DataLoadingConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    wandb: WandBConfig

    num_steps: int = 10000
    batch_size: int = 4
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    val_data_dir: str = "experiments/data/val_data"
    save_interval: int = 500
    log_interval: int = 100
    val_interval: int = 100
    resume_from: Optional[str] = None
    device: Optional[str] = None
    grad_accum_steps: int = 4 # effective size is that *batch_size


@dataclass
class Config:
    """Master config containing all sub-configs."""
    prior: TabICLPriorDataLoaderConfig
    feature_adding: FeatureAddingConfig
    feature_selection: FeatureSelectionConfig
    data_generation: DataGenerationConfig
    training: TrainingConfig
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load config from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        training_dict = config_dict.get('training', {})
        data_loader_config = DataLoadingConfig(**training_dict.pop('data_loader', {}))
        model_config = ModelConfig(**training_dict.pop('model', {}))
        optimizer_config = OptimizerConfig(**training_dict.pop('optimizer', {}))
        wandb_config = WandBConfig(**training_dict.pop('wandb', {}))
        return cls(
            prior=TabICLPriorDataLoaderConfig(**config_dict['prior']),
            feature_adding=FeatureAddingConfig(**config_dict['feature_adding']),
            feature_selection=FeatureSelectionConfig(**config_dict['feature_selection']),
            data_generation=DataGenerationConfig(**config_dict['data_generation']),
            training=TrainingConfig(
                data_loader=data_loader_config,
                model=model_config,
                optimizer=optimizer_config,
                wandb=wandb_config,
                **training_dict  
            )
        )
    
    def to_yaml(self, path: str):
        config_dict = {
            'prior': asdict(self.prior),
            'feature_adding': asdict(self.feature_adding),
            'feature_selection': asdict(self.feature_selection),
            'data_generation': asdict(self.data_generation),
            'training': asdict(self.training)
        }
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


def load_config(config_path: str = "configs/default.yaml") -> Config:
    return Config.from_yaml(config_path)


if __name__ == "__main__":
    config = load_config("configs/default.yaml")
    
    print(f"Prior batch size: {config.prior.batch_size}")
    print(f"Feature selection C: {config.feature_selection.C}")
    print(f"Training steps: {config.training.num_steps}")
    print(f"Model: {config.training.model.model_name}")
    print(f"Learning rate: {config.training.optimizer.learning_rate}")
    print(f"Data dir: {config.training.data_loader.data_dir}")
    
    config.to_yaml("configs/saved_config.yaml")