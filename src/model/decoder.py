import torch
import torch.nn as nn

from pathlib import Path
from tabpfn.model.loading import load_model_criterion_config

from src.utils.misc import setup_logger

logger = setup_logger(__name__)

class TabPFNFeatureSelector(nn.Module):
    """
    TabPFN-Wide encoder (frozen) + decoder for feature selection task.
    """
    
    def __init__(
        self,
        model_name: str = "TabPFN-Wide-5k",
        model_checkpoint_dir: str = "external/models",
        embedding_layer: int = 4,
        device: str = "cuda",
    ):
        super().__init__()
        
        self.model_name = model_name
        self.model_checkpoint_dir = model_checkpoint_dir
        self.embedding_layer = embedding_layer
        self.device = device
        
        # load and freeze the model
        self.encoder = self._load_tabpfn()
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        
        # embedding extraction
        self.embeddings = {'data': None}
        self._register_hook()
        self.emb_size = self.encoder.ninp
        
        # decoder
        self.pad_size = 10000
        self.mask = None
        self.hidden_dims = [256, 128, 64, 32]
        self.dropout = 0.1
        self.decoder = self._create_decoder()
    
    def _load_tabpfn(self):
        """Load pretrained TabPFN-Wide model."""
        valid_models = ["TabPFN-Wide-1.5k", "TabPFN-Wide-5k", "TabPFN-Wide-8k", "TabPFNv2"]
        assert self.model_name in valid_models, f"Invalid model: {self.model_name}"
        
        model, _, _ = load_model_criterion_config(
            model_path=None,
            check_bar_distribution_criterion=False,
            cache_trainset_representation=False,
            which='classifier',
            version='v2',
            download=True,
        )
        
        # Load checkpoint for non-v2 models
        if self.model_name != "TabPFNv2":
            model.features_per_group = 1
            checkpoint_path = Path(f"{self.model_checkpoint_dir}/{self.model_name}_submission.pt")
            
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Model not found: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint)
            logger.info(f"[MODEL] Loaded weights from {checkpoint_path}")
        
        return model.to(self.device)
    
    def _create_decoder(self):
        """Simple MLP-based decoder."""
        layers = []
        in_dim = self.emb_size*2
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU(approximate='none'))
            layers.append(nn.Dropout(self.dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers).to(self.device)

    def load_decoder_checkpoint(self, checkpoint_path: str | Path):
        """       
        Args:
            checkpoint_path: Path to checkpoint file (e.g., 'checkpoints/best_model.pt')
        
        Returns:
            self
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"[MODEL] Loading decoder checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.decoder.load_state_dict(checkpoint["model_state"])
        self.decoder.eval()
        
        logger.info(f"[MODEL] Decoder loaded successfully (step: {checkpoint.get('step', 'unknown')})")
        
        return self

    def _register_hook(self):
        """Register forward hook to extract embeddings from layer K."""
        def hook_fn(module, input, output):
            self.embeddings['data'] = output
        try:
            layer = self.encoder.transformer_encoder.layers[self.embedding_layer] # PerFeatureEncoderLayer
            layer.register_forward_hook(hook_fn)
        except (AttributeError, IndexError) as e:
            raise RuntimeError(f"Could not register hook at layer {self.embedding_layer}: {e}")
    
    def forward(self, train_x, train_y, test_x):
        """        
        Args:
            train_x: (train_seq_len, batch_size, n_features)
            train_y: (train_seq_len, batch_size)
            test_x: (test_seq_len, batch_size, n_features)
        
        Returns:
            embeddings: (batch_size, train_seq_len+test_seq_len, n_features+1, embedding_size)
        """
        with torch.no_grad():
            _ = self.encoder(train_x=train_x, train_y=train_y, test_x=test_x) #(batch_size, train_seq_len+test_seq_len, n_features+1, embedding_size)

        # aggregate the feature embeddings across samples and pad to fixed size
        avg_pool_embeddings = self.embeddings["data"].mean(dim=1) #(batch_size, n_features+1, embedding_size)
        batch_size, n_features, emb_size = avg_pool_embeddings.shape
        n_features = n_features-1 #accountig for y in the 1st dim

        if n_features > self.pad_size:
            raise Warning(f"Number of features ({n_features}) exceeds fixed pad size ({self.pad_size})")

        features_emb = avg_pool_embeddings[:, :-1, :]  # (batch_size, n_features, emb_size)
        y_emb = avg_pool_embeddings[:, -1:, :]  # (batch_size, 1, emb_size)
        
        # concatenate y to each feature 
        y_emb = y_emb.expand(-1, n_features, -1)  # (batch_size, n_features, emb_size)
        feature_y_embeddings = torch.cat([features_emb, y_emb], dim=-1)  # (batch_size, n_features, emb_size*2)
        
        padding = torch.zeros(
            batch_size, self.pad_size - n_features, feature_y_embeddings.shape[-1],
            device=feature_y_embeddings.device,
            dtype=feature_y_embeddings.dtype
        )
        padded_embeddings = torch.cat([feature_y_embeddings, padding], dim=1)  # (batch_size, pad_size, emb_size*2)

        self.mask = torch.zeros(batch_size, self.pad_size, dtype=torch.bool, device=feature_y_embeddings.device)
        self.mask[:, :n_features] = True  # (batch_size, pad_size)

        # flat for batchnorm
        flat_embeddings = padded_embeddings.reshape(-1, feature_y_embeddings.shape[-1])  # (batch_size*pad_size, emb_size*2)
                
        # run through decoder
        out = self.decoder(flat_embeddings) # (batch_size * self.pad_size, 1)
        out = out.reshape(batch_size, self.pad_size) # (batch_size, self.pad_size)
        
        return out[self.mask].reshape(batch_size, n_features)