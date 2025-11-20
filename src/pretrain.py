import torch
import time
import wandb
from datetime import datetime

from torch import nn
from torch.optim import AdamW
from pathlib import Path
from contextlib import nullcontext

from src.data.load_datasets import create_dataloader
from src.utils.config import load_config, TrainingConfig
from src.utils.misc import set_seed, setup_logger
from src.model.decoder import TabPFNDecoder


class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.logger = setup_logger()
        set_seed(config.seed)
        self._configure_amp()
        
        
        self.experiment_name = config.experiment_name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = Path(config.checkpoint_dir) / self.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.grad_accum_steps = config.grad_accum_steps

        self._log_config()
        
        self.logger.info(f"[MODEL] Loading {config.model.model_name} with trainable decoder..")
        self.model = self._load_model()
        self.model.train()

        self.logger.info(f"[DATA] Loading data from {config.data_loader.data_dir}..")
        self.train_test_split = config.data_loader.__dict__.pop("train_test_split", 0.3)
        self.train_loader = create_dataloader(**config.data_loader.__dict__)
        self.logger.info(f"[DATA] DataLoader loaded: {len(self.train_loader)} batches")
        
        self.val_interval = getattr(config, 'val_interval', 0)
        if self.val_interval > 0:
            self.val_loader = self._load_validation_data()
        else:
            self.val_loader = None

        self.optimizer, self.scheduler = self._set_optimizer_and_scheduler()
        
        self.criterion = nn.MSELoss()
                
        self.curr_step = 0
        self.oom_errors = 0
        self.best_val_loss = float('inf')
        
        if config.resume_from:
            self.curr_step = self._load_checkpoint()
    
        self._configure_wandb()
    
    def _configure_amp(self): # from tabpfn-wide
        self.amp = "cuda" in self.device
        self.scaler = torch.GradScaler("cuda", enabled=self.amp)
        if self.amp:
            self.amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
        else:
            self.amp_ctx = nullcontext()
    

    def _configure_wandb(self):
        self.use_wandb = getattr(self.config.wandb, 'use_wandb', True)
        
        if self.use_wandb:
            wandb_config = {
                "experiment": self.experiment_name,
                "model": self.config.model.model_name,
                "learning_rate": self.config.optimizer.learning_rate,
                "weight_decay": self.config.optimizer.weight_decay,
                "scheduler": self.config.optimizer.scheduler,
                "gradient_clip": self.config.optimizer.gradient_clip,
                "batch_size": self.config.batch_size,
                "num_steps": self.config.num_steps,
                "grad_accum_steps": self.grad_accum_steps,
                "seed": self.config.seed,
                "device": self.device,
                "train_test_split": self.train_test_split,
            }
            
            wandb_project = getattr(self.config.wandb, 'project', 'tabpfn-training')
            wandb_entity = getattr(self.config.wandb, 'entity', None)
            wandb_run_name = getattr(self.config.wandb, 'run_name', None)
            
            resume_id = getattr(self, 'wandb_id', None) # resuming
            
            self.wandb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name or self.experiment_name,
                config=wandb_config,
                resume="allow" if resume_id else None,
                id=resume_id,
            )
            
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            wandb.run.summary["total_parameters"] = total_params
            wandb.run.summary["trainable_parameters"] = trainable_params
            
            self.logger.info(f"\n[SETUP] WandB initialized: {self.wandb_run.url}")

    def _load_model(self):
        model = TabPFNDecoder(
            model_name=self.config.model.model_name,
            embedding_layer=getattr(self.config.model, 'embedding_layer', -1),
            device=self.device,
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"[MODEL] Total parameters: {total_params:,}")
        self.logger.info(f"[MODEL] Trainable parameters (decoder): {trainable_params:,}")
        
        return model

    def _set_optimizer_and_scheduler(self):
        optimizer = AdamW(
            self.model.decoder.parameters(),
            lr=self.config.optimizer.learning_rate,
            weight_decay=self.config.optimizer.weight_decay
        )
        
        if self.config.optimizer.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.num_steps
            )
        else:
            raise ValueError(f"[SETUP] Unknown scheduler: {self.config.optimizer.scheduler}")
        
        return optimizer, scheduler
    
    def _load_validation_data(self):
        val_data_dir = getattr(self.config, 'val_data_dir', None)
        
        if val_data_dir is None:
            self.logger.info("[DATA] No validation data directory specified, skipping validation")
            return None
        
        self.logger.info(f"[DATA] Loading validation data from {val_data_dir}..")
        
        val_loader_config = self.config.data_loader.__dict__.copy()
        val_loader_config['data_dir'] = val_data_dir
        
        try:
            val_loader = create_dataloader(**val_loader_config)
            self.logger.info(f"[DATA] Validation DataLoader loaded: {len(val_loader)} batches")
            return val_loader
        except Exception as e:
            self.logger.info(f"[DATA] Could not load validation data: {e}")
            return None
        
    def _prepare_batch(self, batch):
        X = batch['X'].to(self.device)
        y = batch['y'].to(self.device)
        lasso_coeffs = batch['lasso_coeffs'].to(self.device)  # (batch_size, n_features)
        
        _, seq_len, _ = X.shape
        train_size = int(seq_len * self.train_test_split)
        
        X_train = X[:, :train_size, :].transpose(0, 1) # (seq_len, batch_size, n_features)
        X_test = X[:, train_size:, :].transpose(0, 1)
        y_train = y[:, :train_size].transpose(0, 1) # (seq_len, batch_size)
        y_test = y[:, train_size:].transpose(0, 1)
        
        return X_train, y_train, X_test, y_test, lasso_coeffs
    
    def _save_checkpoint(self, step: int, is_best: bool = False):
        checkpoint = {
            "step": step,
            "config": self.config,
            "model_state": self.model.decoder.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
        }

        if self.use_wandb:
            checkpoint["wandb_id"] = self.wandb_run.id
        
        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, path)
            self.logger.info(f"[SAVE] Saved best checkpoint: {path}")
        else:
            path = self.checkpoint_dir / f"step_{step}.pt"
            torch.save(checkpoint, path)
            self.logger.info(f"[SAVE] Saved checkpoint: {path}")
    
    def _load_checkpoint(self):
        checkpoint = torch.load(
            self.config.resume_from,
            map_location=self.device,
            weights_only=False
        )
        
        self.model.decoder.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        step = checkpoint["step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        self.wandb_id = checkpoint.get("wandb_id", None)  
        
        self.logger.info(f"[RESUME] Resumed from step {step}")
        return step
    
    def _log_config(self):
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Training Configuration")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Experiment: {self.experiment_name}")
        self.logger.info(f"Model: {self.config.model.model_name}")
        self.logger.info(f"Data path: {self.config.data_loader.data_dir}")
        self.logger.info(f"Steps: {self.config.num_steps}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Seed: {self.config.seed}")
        self.logger.info(f"Gradient Accumulation: {self.grad_accum_steps}")
        self.logger.info(f"Checkpoint dir: {self.checkpoint_dir}")
        self.logger.info(f"{'='*60}\n")
    
    def _infinite_loader(self):
        while True:
            for batch in self.train_loader:
                yield batch

    def _validate(self):
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        val_oom_errors = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                if batch is None:
                    continue
                
                try:
                    X_train, y_train, X_test, y_test, lasso_coeffs = self._prepare_batch(batch)
                    
                    with self.amp_ctx:
                        pred_logits = self.model(
                            train_x=X_train,
                            train_y=y_train,
                            test_x=X_test,
                        )
                        pred_logits = pred_logits.float()
                    
                    pred_logits = pred_logits[self.model.mask]
                    lasso_coeffs = lasso_coeffs.reshape(-1) 
                    loss = self.criterion(pred_logits, lasso_coeffs)
                    
                    total_loss += loss.item()
                    total_batches += 1
                    
                except torch.cuda.OutOfMemoryError:
                    val_oom_errors += 1
                    self.logger.info(f"[VAL] OOM during validation batch, skipping...")
                    torch.cuda.empty_cache()
                    continue
        
        self.model.train()
        
        if total_batches == 0:
            self.logger.info("[VAL] No validation batches processed")
            return None
        
        avg_val_loss = total_loss / total_batches
     
        self.logger.info(f"[VAL] step: {self.curr_step} | loss: {avg_val_loss:.6f} | oom errors: {val_oom_errors}")        

        if self.use_wandb:
            wandb.log({
                "val/loss": avg_val_loss,
                "val/oom_errors": val_oom_errors,
            }, step=self.curr_step)

        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self._save_checkpoint(self.curr_step, is_best=True)
            self.logger.info(f"[VAL] New best validation loss: {avg_val_loss:.6f}")

            if self.use_wandb:
                wandb.run.summary["best_val_loss"] = self.best_val_loss

        return avg_val_loss

    def train(self):
        data_iter = self._infinite_loader()
        
        self.logger.info(f"{'='*60}")
        self.logger.info(f"TRAINING")
        self.logger.info(f"{'='*60}")

        step_start_time = time.time() 

        # training loop
        for step in range(self.curr_step, self.config.num_steps):            
            self.curr_step = step
            
            batch = next(data_iter)
            if batch is None:
                continue
            
            X_train, y_train, X_test, y_test, lasso_coeffs = self._prepare_batch(batch)
            try:
                with self.amp_ctx:
                    pred_logits = self.model(
                        train_x=X_train,
                        train_y=y_train,
                        test_x=X_test,
                    )
                    pred_logits = pred_logits.float() # (batch_size, pad_size)
                
                pred_logits = pred_logits[self.model.mask]
                lasso_coeffs = lasso_coeffs.reshape(-1) 
                loss = self.criterion(pred_logits, lasso_coeffs) / self.grad_accum_steps  # need to scale loss
                
                self.scaler.scale(loss).backward()  # scale up to prevent underflow in float16
                
                if (step + 1) % self.grad_accum_steps == 0:  # gradient accumulation
                    
                    if self.use_wandb:
                        total_norm = 0.0
                        for p in self.model.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** 0.5

                    if self.config.optimizer.gradient_clip > 0:
                        self.scaler.unscale_(self.optimizer)  # unscale gradients
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.optimizer.gradient_clip
                        )
                    
                    # update params
                    scale_before = self.scaler.get_scale()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()  # updating scale factor
                    scale_after = self.scaler.get_scale()
                    
                    # update only if optimizer took step
                    if scale_before <= scale_after:
                        self.scheduler.step()
                    
                    self.optimizer.zero_grad()  # zero out accumulated gradient
                    
                    if self.use_wandb:
                        step_time = time.time() - step_start_time
                        wandb.log({
                            "train/loss": loss.item() * self.grad_accum_steps,
                            "train/learning_rate": self.scheduler.get_last_lr()[0],
                            "train/grad_norm": total_norm,
                            "train/loss_scale": scale_after,
                            "train/step_time": step_time,
                            "train/oom_errors": self.oom_errors,
                        }, step=step)
                        step_start_time = time.time()  # Reset timer

            except torch.cuda.OutOfMemoryError:
                self.oom_errors += 1
                self.logger.info(f"OOM at step {step}")
                self.optimizer.zero_grad(set_to_none=True)  # clear out partial
                torch.cuda.empty_cache()
                if self.oom_errors / (step + 1) > 0.1:
                    raise RuntimeError("Too many OOM errors, stopping training.")
                continue

            # log every 25 steps
            if step % self.config.log_interval == 0:
                self.logger.info(
                    f"[TRAIN] step: {step}/{self.config.num_steps} | "
                    f"loss: {(loss.item() * self.grad_accum_steps):.6f} | "
                    f"oom errors: {self.oom_errors} | "
                    f"lr: {self.scheduler.get_last_lr()[0]:.6f} "
                )

            # validation every 20 steps
            if self.val_interval > 0 and step > 0 and step % self.val_interval == 0:
                self._validate()
            
            # save every 250 steps
            if step > 0 and step % self.config.save_interval == 0:
                self._save_checkpoint(step)
        
        # final checkpoint
        self._save_checkpoint(self.config.num_steps)

        if self.use_wandb:
            wandb.finish()
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("Training complete!")
        if self.best_val_loss < float('inf'):
            self.logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        self.logger.info(f"Checkpoints saved to: {self.checkpoint_dir}")
        self.logger.info(f"{'='*60}")


if __name__ == "__main__":
    config = load_config("configs/default.yaml")
    trainer = Trainer(config.training)
    trainer.train()