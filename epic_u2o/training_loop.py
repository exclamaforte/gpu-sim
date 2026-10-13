#!/usr/bin/env python3
"""Training loop with checkpointing and frequent eval hooks.

Standard training loop wired to the dataset pipeline with checkpointing
and periodic evaluation-suite runs during training.
"""

import os
import time
import json
import shutil
import math
from dataclasses import dataclass, asdict
from typing import Optional, Callable
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Optimization
    lr: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Scheduler
    scheduler_type: str = 'cosine'  # 'cosine', 'cosine_warmup', 'constant'
    warmup_steps: int = 500
    max_steps: int = 10000
    
    # Checkpointing
    checkpoint_dir: str = 'runs/checkpoints'
    save_every_steps: int = 1000
    keep_last_n: int = 5
    save_best: bool = True
    
    # Evaluation
    eval_every_steps: int = 500
    eval_batches: int = 100
    
    # Logging
    log_dir: str = 'runs/logs'
    log_every_steps: int = 100
    
    # Mixed precision
    use_amp: bool = True
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Data
    max_seq_len: int = 2048


class CUDADataset(Dataset):
    """Dataset for CUDA runtime prediction."""
    
    def __init__(self, data: list[dict], tokenizer, max_seq_len: int = 2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize source
        token_ids = self.tokenizer.tokenize_for_model(item['source'], self.max_seq_len)
        
        # Pad or truncate
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]
        else:
            token_ids = token_ids + [0] * (self.max_seq_len - len(token_ids))
        
        # Attention mask
        attention_mask = [1 if t != 0 else 0 for t in token_ids]
        
        # Numeric features
        from epic_u2o.features import extract_normalized_features
        numeric_features = extract_normalized_features(item['source'], None)
        
        # Target
        target = item.get('median_ms', item.get('target_median_ms', 1.0))
        log_target = torch.tensor([torch.log(torch.tensor(target + 1e-6))], dtype=torch.float)
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'numeric_features': torch.tensor(numeric_features, dtype=torch.float),
            'target': log_target,
            'kernel_hash': item.get('kernel_hash', ''),
        }


def create_dataloaders(
    train_data: list[dict],
    val_data: list[dict],
    tokenizer,
    batch_size: int = 32,
    max_seq_len: int = 2048,
    num_workers: int = 0,  # 0 for testing to avoid multiprocessing issues
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    train_dataset = CUDADataset(train_data, tokenizer, max_seq_len)
    val_dataset = CUDADataset(val_data, tokenizer, max_seq_len)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


class TrainingLoop:
    """Main training loop with checkpointing and eval hooks."""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        tokenizer=None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        eval_fn: Optional[Callable] = None,
    ):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.eval_fn = eval_fn
        
        # Setup device
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup AMP
        self.scaler = torch.amp.GradScaler('cuda') if config.use_amp and self.device.type == 'cuda' else None
        
        # Setup checkpointing
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float('inf')
        self.step = 0
        self.epoch = 0
        
        # Setup logging
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f'training_log_{int(time.time())}.jsonl'
        
        # Metrics
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.scheduler_type == 'cosine':
            return CosineAnnealingLR(self.optimizer, T_max=self.config.max_steps)
        elif self.config.scheduler_type == 'cosine_warmup':
            def lr_lambda(step):
                if step < self.config.warmup_steps:
                    return step / self.config.warmup_steps
                progress = (step - self.config.warmup_steps) / (self.config.max_steps - self.config.warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))
            return LambdaLR(self.optimizer, lr_lambda)
        else:
            return LambdaLR(self.optimizer, lambda _: 1.0)
    
    def train_step(self, batch: dict) -> dict:
        """Single training step."""
        self.model.train()
        
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        numeric_features = batch['numeric_features'].to(self.device)
        target = batch['target'].to(self.device)
        
        self.optimizer.zero_grad()
        
        if self.scaler:
            with torch.amp.autocast('cuda'):
                # Forward
                if hasattr(self.model, 'forward') and 'log_var' in str(self.model.forward.__code__.co_varnames):
                    # Dual head model
                    mean, log_var = self.model(input_ids, attention_mask, numeric_features)
                    loss = self._gaussian_nll_loss(mean, log_var, target)
                else:
                    # Single head model
                    pred = self.model(input_ids, attention_mask, numeric_features)
                    loss = nn.functional.mse_loss(pred, target)
            
            # Backward
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Forward
            if hasattr(self.model, 'forward') and 'log_var' in str(self.model.forward.__code__.co_varnames):
                mean, log_var = self.model(input_ids, attention_mask, numeric_features)
                loss = self._gaussian_nll_loss(mean, log_var, target)
            else:
                pred = self.model(input_ids, attention_mask, numeric_features)
                loss = nn.functional.mse_loss(pred, target)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
        
        self.scheduler.step()
        
        return {'loss': loss.item(), 'lr': self.scheduler.get_last_lr()[0]}
    
    def _gaussian_nll_loss(self, mean: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Gaussian NLL loss."""
        var = log_var.exp()
        nll = 0.5 * (log_var + (target - mean).pow(2) / var + math.log(2 * math.pi))
        return nll.mean()
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader, max_batches: Optional[int] = None) -> dict:
        """Evaluate model on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_targets = []
        
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            numeric_features = batch['numeric_features'].to(self.device)
            target = batch['target'].to(self.device)
            
            if hasattr(self.model, 'forward') and 'log_var' in str(self.model.forward.__code__.co_varnames):
                mean, log_var = self.model(input_ids, attention_mask, numeric_features)
                loss = self._gaussian_nll_loss(mean, log_var, target)
                all_preds.extend(mean.cpu().numpy())
            else:
                pred = self.model(input_ids, attention_mask, numeric_features)
                loss = nn.functional.mse_loss(pred, target)
                all_preds.extend(pred.cpu().numpy())
            
            all_targets.extend(target.cpu().numpy())
            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        
        # Compute metrics
        import numpy as np
        preds = np.array(all_preds).flatten()
        targets = np.array(all_targets).flatten()
        
        mape = np.mean(np.abs((targets - preds) / (targets + 1e-6))) * 100
        mse = np.mean((targets - preds) ** 2)
        r2 = 1 - np.sum((targets - preds) ** 2) / np.sum((targets - np.mean(targets)) ** 2)
        
        return {
            'loss': avg_loss,
            'mape': mape,
            'mse': mse,
            'r2': r2,
            'num_samples': total_samples,
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config),
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_step_{self.step}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model at step {self.step}")
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keep only last N."""
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_step_*.pt'))
        while len(checkpoints) > self.config.keep_last_n:
            checkpoints[0].unlink()
            checkpoints = checkpoints[1:]
    
    def load_checkpoint(self, path: str) -> dict:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        return checkpoint
    
    def _log_metrics(self, metrics: dict, prefix: str = 'train'):
        """Log metrics to file."""
        log_entry = {
            'step': self.step,
            'epoch': self.epoch,
            'timestamp': time.time(),
            f'{prefix}_loss': metrics.get('loss'),
            f'{prefix}_mape': metrics.get('mape'),
            f'{prefix}_mse': metrics.get('mse'),
            f'{prefix}_r2': metrics.get('r2'),
        }
        
        log_entry = {k: (float(v.item()) if hasattr(v, "item")
                         else float(v) if isinstance(v, (int, float)) else v)
                     for k, v in log_entry.items() if v is not None}
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def run(
        self,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        max_epochs: Optional[int] = None,
    ):
        """Run the training loop."""
        train_loader = train_loader or self.train_loader
        val_loader = val_loader or self.val_loader
        
        if train_loader is None:
            raise ValueError("No train_loader provided")
        
        print(f"Starting training on {self.device}")
        print(f"Max steps: {self.config.max_steps}")
        print(f"Checkpoint dir: {self.checkpoint_dir}")
        
        while self.step < self.config.max_steps:
            self.epoch += 1
            
            for batch in train_loader:
                if self.step >= self.config.max_steps:
                    break
                
                # Training step
                metrics = self.train_step(batch)
                self.step += 1
                
                # Logging
                if self.step % self.config.log_every_steps == 0:
                    print(f"Step {self.step}: loss={metrics['loss']:.4f}, lr={metrics['lr']:.2e}")
                    self._log_metrics(metrics, 'train')
                    self.train_losses.append(metrics['loss'])
                    self.learning_rates.append(metrics['lr'])
                
                # Evaluation
                if val_loader and self.step % self.config.eval_every_steps == 0:
                    val_metrics = self.evaluate(val_loader, self.config.eval_batches)
                    print(f"Step {self.step} Val: loss={val_metrics['loss']:.4f}, MAPE={val_metrics['mape']:.2f}%, R2={val_metrics['r2']:.3f}")
                    self._log_metrics(val_metrics, 'val')
                    self.val_losses.append(val_metrics['loss'])
                    
                    # Checkpointing
                    is_best = val_metrics['loss'] < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint(is_best=is_best)
                    
                    # Call custom eval hook
                    if self.eval_fn:
                        self.eval_fn(self.model, self.step, val_metrics)
            
            if max_epochs and self.epoch >= max_epochs:
                break
        
        # Final checkpoint
        self.save_checkpoint(is_best=True)
        print("Training completed!")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'total_steps': self.step,
        }


def create_training_loop(
    model: nn.Module,
    tokenizer,
    train_data: list[dict],
    val_data: list[dict],
    config: Optional[TrainingConfig] = None,
) -> TrainingLoop:
    """Factory function to create training loop with data loaders."""
    config = config or TrainingConfig()
    
    train_loader, val_loader = create_dataloaders(
        train_data, val_data, tokenizer,
        batch_size=32,
        max_seq_len=getattr(config, 'max_seq_len', 2048),
    )
    
    return TrainingLoop(
        model=model,
        config=config,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
    )


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/home/gabe/Documents/gpu-sim')

    # Lightweight smoke test: tiny model, single train_step on CPU, no dataset scan.
    from epic_u2o.model import create_model

    tiny = create_model(
        vocab_size=64,
        max_seq_len=32,
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_ff=64,
        numeric_dim=26,
        use_numeric=True,
        dual_head=True,
    )
    training_config = TrainingConfig(
        max_steps=2,
        checkpoint_dir='/tmp/test_checkpoints_smoke',
        log_dir='/tmp/test_logs_smoke',
        eval_every_steps=1,
        log_every_steps=1,
        device='cpu',
        use_amp=False,
    )
    loop = TrainingLoop(model=tiny, config=training_config)

    batch = {
        'input_ids': torch.randint(0, 64, (2, 16)),
        'attention_mask': torch.ones(2, 16, dtype=torch.long),
        'numeric_features': torch.randn(2, 26),
        'target': torch.zeros(2, 1),
    }
    out = loop.train_step(batch)
    assert 'loss' in out and math.isfinite(out['loss']), f"bad train_step output: {out}"
    assert hasattr(loop, 'save_checkpoint') and hasattr(loop, 'evaluate') and hasattr(loop, 'run')
    print(f"Smoke OK: loss={out['loss']:.4f} lr={out['lr']:.2e} params={sum(p.numel() for p in tiny.parameters())}")