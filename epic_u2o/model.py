#!/usr/bin/env python3
"""Transformer encoder + pooling regression head for CUDA runtime prediction.

Small encoder-only transformer (~48M parameters) over tokenized code/PTX,
with mean or [CLS]-style pooling feeding a small MLP regression head.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Configuration for the transformer model."""
    # Vocabulary
    vocab_size: int = 256  # Token vocabulary size
    max_seq_len: int = 2048
    
    # Transformer
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    activation: str = 'gelu'
    
    # Pooling
    pool_type: str = 'mean'  # 'mean', 'cls', 'max'
    
    # Regression head
    head_hidden: int = 256
    head_layers: int = 2
    head_dropout: float = 0.1
    
    # Numeric features (optional multi-modal)
    numeric_dim: int = 26
    use_numeric: bool = True
    fusion_type: str = 'concat'  # 'concat', 'cross_attention'
    
    # Output
    output_dim: int = 1  # Predict log runtime
    
    # Training
    label_smoothing: float = 0.0
    
    @property
    def num_parameters(self) -> int:
        """Estimate number of parameters."""
        # Embedding
        emb_params = self.vocab_size * self.d_model
        
        # Transformer layers
        # Per layer: attention (4*d_model^2) + FFN (2*d_model*d_ff) + norms (4*d_model)
        layer_params = 4 * self.d_model**2 + 2 * self.d_model * self.d_ff + 4 * self.d_model
        transformer_params = layer_params * self.n_layers
        
        # Pooling + head
        head_params = self.d_model * self.head_hidden + self.head_hidden
        for _ in range(self.head_layers - 1):
            head_params += self.head_hidden * self.head_hidden + self.head_hidden
        head_params += self.head_hidden * self.output_dim
        
        # Numeric encoder (if used)
        numeric_params = 0
        if self.use_numeric:
            numeric_params = self.numeric_dim * 128 + 128
            numeric_params += 128 * self.d_model + self.d_model
        
        # Fusion
        fusion_params = 0
        if self.use_numeric and self.fusion_type == 'cross_attention':
            fusion_params = self.d_model * self.d_model * 2
        
        total = emb_params + transformer_params + head_params + numeric_params + fusion_params
        return total


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerEncoderLayer(nn.Module):
    """Pre-LN Transformer encoder layer."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, activation: str):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with pre-LN
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = residual + self.dropout(x)
        
        # FFN with pre-LN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)
        
        return x


class TransformerEncoder(nn.Module):
    """Encoder-only transformer."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_len)
        self.dropout = nn.Dropout(config.dropout)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                config.d_model, config.n_heads, config.d_ff, config.dropout, config.activation
            )
            for _ in range(config.n_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # input_ids: [batch, seq_len]
        x = self.embedding(input_ids)  # [batch, seq_len, d_model]
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Create attention mask for padding
        # attention_mask: 1 for valid, 0 for padding -> key_padding_mask: True for padding
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0  # True where padding
        else:
            key_padding_mask = None
        
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        
        x = self.norm(x)
        return x


class PoolingLayer(nn.Module):
    """Pooling layer for sequence representations."""
    
    def __init__(self, pool_type: str = 'mean'):
        super().__init__()
        self.pool_type = pool_type
        if pool_type == 'cls':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, 512))
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        # attention_mask: [batch, seq_len] (1 for valid, 0 for padding)
        
        if self.pool_type == 'cls':
            # Prepend CLS token
            batch_size = x.size(0)
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat([
                    torch.ones(batch_size, 1, device=x.device),
                    attention_mask
                ], dim=1)
            return x[:, 0]  # [batch, d_model]
        
        elif self.pool_type == 'mean':
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
                x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                x = x.mean(dim=1)
            return x
        
        elif self.pool_type == 'max':
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                x = x.masked_fill(mask == 0, -1e9)
            return x.max(dim=1).values
        
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")


class RegressionHead(nn.Module):
    """MLP regression head."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for i in range(n_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NumericEncoder(nn.Module):
    """Encoder for numeric features."""
    
    def __init__(self, numeric_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(numeric_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiModalFusion(nn.Module):
    """Fusion module for code tokens + numeric features."""
    
    def __init__(self, d_model: int, fusion_type: str = 'concat'):
        super().__init__()
        self.fusion_type = fusion_type
        if fusion_type == 'concat':
            self.proj = nn.Linear(d_model * 2, d_model)
        elif fusion_type == 'cross_attention':
            self.cross_attn = nn.MultiheadAttention(d_model, 8, batch_first=True)
            self.norm = nn.LayerNorm(d_model)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
    
    def forward(self, code_repr: torch.Tensor, numeric_repr: torch.Tensor) -> torch.Tensor:
        # code_repr: [batch, d_model], numeric_repr: [batch, d_model]
        if self.fusion_type == 'concat':
            fused = torch.cat([code_repr, numeric_repr], dim=-1)
            return self.proj(fused)
        elif self.fusion_type == 'cross_attention':
            # Cross-attention: numeric as query, code as key/value
            numeric_q = numeric_repr.unsqueeze(1)  # [batch, 1, d_model]
            code_kv = code_repr.unsqueeze(1)       # [batch, 1, d_model]
            fused, _ = self.cross_attn(numeric_q, code_kv, code_kv)
            fused = self.norm(fused.squeeze(1) + numeric_repr)
            return fused
        else:
            return code_repr


class CUDARuntimePredictor(nn.Module):
    """Main model for CUDA runtime prediction."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Code encoder
        self.encoder = TransformerEncoder(config)
        self.pooling = PoolingLayer(config.pool_type)
        
        # Numeric encoder (optional)
        self.use_numeric = config.use_numeric
        if config.use_numeric:
            self.numeric_encoder = NumericEncoder(config.numeric_dim, config.d_model)
            self.fusion = MultiModalFusion(config.d_model, config.fusion_type)
            head_input_dim = config.d_model
        else:
            head_input_dim = config.d_model
        
        # Regression head
        self.head = RegressionHead(
            head_input_dim,
            config.head_hidden,
            config.output_dim,
            config.head_layers,
            config.head_dropout,
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        numeric_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Encode code tokens
        code_repr = self.encoder(input_ids, attention_mask)  # [batch, seq_len, d_model]
        code_pooled = self.pooling(code_repr, attention_mask)  # [batch, d_model]
        
        # Fuse with numeric features
        if self.use_numeric and numeric_features is not None:
            numeric_repr = self.numeric_encoder(numeric_features)  # [batch, d_model]
            fused = self.fusion(code_pooled, numeric_repr)
        else:
            fused = code_pooled
        
        # Regression head
        output = self.head(fused)  # [batch, output_dim]
        return output


class GaussianNLLLoss(nn.Module):
    """Gaussian Negative Log-Likelihood loss for noise-aware prediction.
    
    Predicts (mean, log_var) of log-runtime.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, mean: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # mean, log_var: [batch, 1] or [batch]
        # target: [batch, 1] or [batch]
        mean = mean.squeeze(-1)
        log_var = log_var.squeeze(-1)
        target = target.squeeze(-1)
        
        # NLL = 0.5 * (log_var + (target - mean)^2 / exp(log_var) + log(2*pi))
        var = log_var.exp()
        nll = 0.5 * (log_var + (target - mean).pow(2) / var + math.log(2 * math.pi))
        
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll


class DualHeadPredictor(nn.Module):
    """Predictor with dual head for mean and log-variance (for Gaussian NLL)."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Shared encoder
        self.encoder = TransformerEncoder(config)
        self.pooling = PoolingLayer(config.pool_type)
        
        # Numeric encoder
        self.use_numeric = config.use_numeric
        if config.use_numeric:
            self.numeric_encoder = NumericEncoder(config.numeric_dim, config.d_model)
            self.fusion = MultiModalFusion(config.d_model, config.fusion_type)
            head_input_dim = config.d_model
        else:
            head_input_dim = config.d_model
        
        # Dual heads
        self.mean_head = RegressionHead(
            head_input_dim,
            config.head_hidden,
            config.output_dim,
            config.head_layers,
            config.head_dropout,
        )
        self.log_var_head = RegressionHead(
            head_input_dim,
            config.head_hidden,
            config.output_dim,
            config.head_layers,
            config.head_dropout,
        )
        
        # Softplus for positive variance
        self.softplus = nn.Softplus()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        numeric_features: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        code_repr = self.encoder(input_ids, attention_mask)
        code_pooled = self.pooling(code_repr, attention_mask)
        
        if self.use_numeric and numeric_features is not None:
            numeric_repr = self.numeric_encoder(numeric_features)
            fused = self.fusion(code_pooled, numeric_repr)
        else:
            fused = code_pooled
        
        mean = self.mean_head(fused)
        log_var = self.log_var_head(fused)
        log_var = self.softplus(log_var) + 1e-6  # Ensure positive
        
        return mean, log_var


def create_model(
    vocab_size: int = 256,
    max_seq_len: int = 2048,
    d_model: int = 512,
    n_heads: int = 8,
    n_layers: int = 6,
    d_ff: int = 2048,
    numeric_dim: int = 26,
    use_numeric: bool = True,
    dual_head: bool = False,
) -> nn.Module:
    """Factory function to create model."""
    config = ModelConfig(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        numeric_dim=numeric_dim,
        use_numeric=use_numeric,
    )
    
    if dual_head:
        return DualHeadPredictor(config)
    else:
        return CUDARuntimePredictor(config)


if __name__ == '__main__':
    # Test model creation and forward pass
    model = create_model(dual_head=False)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Config estimated params: {model.config.num_parameters:,}")
    
    # Test forward
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, 256, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[:, -20:] = 0  # Some padding
    numeric_features = torch.randn(batch_size, 26)
    
    with torch.no_grad():
        output = model(input_ids, attention_mask, numeric_features)
        print(f"Output shape: {output.shape}")
        print(f"Output: {output}")
    
    # Test dual head
    dual_model = create_model(dual_head=True)
    print(f"\nDual-head model parameters: {sum(p.numel() for p in dual_model.parameters()):,}")
    
    with torch.no_grad():
        mean, log_var = dual_model(input_ids, attention_mask, numeric_features)
        print(f"Mean shape: {mean.shape}")
        print(f"Log-var shape: {log_var.shape}")