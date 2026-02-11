from __future__ import annotations

import os
import math
import heapq
from collections import defaultdict
from collections.abc import Iterable
from typing import IO, Any, BinaryIO, Dict, List, Tuple, Iterator, Optional, Union

import numpy as np
import numpy.typing as npt
import regex as re
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Bool, Float, Int
from einops import einsum


# ============================================================================
# Basic Building Blocks
# ============================================================================

class Linear(nn.Module):
    """Custom linear layer with truncated normal initialization."""
    
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None) -> None:
        super().__init__()
        device = torch.device('cpu') if device is None else device
        dtype = torch.float32 if dtype is None else dtype
        
        self.W = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        sigma = math.sqrt(2 / (in_features + out_features))
        init.trunc_normal_(self.W, 0, sigma, -3*sigma, 3*sigma)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.W, '... d_in, d_out d_in -> ... d_out')


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, "d_out d_in"],
    in_features: Float[Tensor, "... d_in"],
) -> Float[Tensor, "... d_out"]:
    """Functional linear transformation."""
    return einsum(weights, in_features, 'd_out d_in, ... d_in -> ... d_out')


class Embedding(nn.Module):
    """Token embedding layer."""
    
    def __init__(self, num_embeddings, embedding_dim, device=torch.device('cpu'), 
                 dtype=torch.float32):
        super().__init__()
        self.embedding = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        self.embedding_dim = embedding_dim
        init.trunc_normal_(self.embedding, mean=0, std=1, a=-3, b=3)
    #forward using run_embedding instead


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, "vocab_size d_model"],
    token_ids: Int[Tensor, "..."],
) -> Float[Tensor, "... d_model"]:
    """Functional embedding lookup."""
    return weights[token_ids]


# ============================================================================
# Normalization
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, d_model: int, eps: float = 1e-5, device=torch.device('cpu'), 
                 dtype=torch.float32):
        super().__init__()
        self.gain_parameter = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.d_model = d_model
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        normalized = x / rms
        return normalized * self.gain_parameter


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, "d_model"],
    in_features: Float[Tensor, "... d_model"],
) -> Float[Tensor, "... d_model"]:
    """Functional RMSNorm."""
    rms = torch.sqrt(torch.mean(in_features ** 2, dim=-1, keepdim=True) + eps)
    normalized = in_features / rms
    return normalized * weights


# ============================================================================
# Feed-Forward Network
# ============================================================================

class SwiGLU(nn.Module):
    """SwiGLU activation function with gating mechanism."""
    
    def __init__(self, d_model: int, device=torch.device('cpu'), dtype=torch.float32):
        super().__init__()
        self.d_model = d_model
        d_ff = (int(d_model * 8 / 3) // 64) * 64
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w1(x))
        value = self.w3(x)
        return self.w2(gate * value)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, "d_ff d_model"],
    w2_weight: Float[Tensor, "d_model d_ff"],
    w3_weight: Float[Tensor, "d_ff d_model"],
    in_features: Float[Tensor, "... d_model"],
) -> Float[Tensor, "... d_model"]:
    """Functional SwiGLU."""
    x1 = torch.matmul(in_features, w1_weight.T)
    x3 = torch.matmul(in_features, w3_weight.T)
    return torch.matmul(F.silu(x1) * x3, w2_weight.T)


def run_silu(in_features: Tensor) -> Tensor:
    """SiLU (Swish) activation function."""
    return in_features * torch.sigmoid(in_features)


# ============================================================================
# Attention Mechanisms
# ============================================================================

def run_scaled_dot_product_attention(
    Q: Float[Tensor, "batch num_heads seq_len_q d_k"],
    K: Float[Tensor, "batch num_heads seq_len_k d_k"],
    V: Float[Tensor, "batch num_heads seq_len_v d_v"],
    mask: Float[Tensor, "1 1 seq_len_q seq_len_k"] | None = None,
) -> Float[Tensor, "batch num_heads seq_len_q d_v"]:
    """Scaled dot-product attention with optional causal mask."""
    d_k = Q.size(-1)
    scores = einsum(Q, K, "b h q d, b h k d -> b h q k") / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -torch.finfo(Q.dtype).max)
    
    attn_weights = torch.softmax(scores, dim=-1)
    output = einsum(attn_weights, V, "b h q k, b h k v -> b h q v")
    return output


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, "d_k d_in"],
    k_proj_weight: Float[Tensor, "d_k d_in"],
    v_proj_weight: Float[Tensor, "d_v d_in"],
    o_proj_weight: Float[Tensor, "d_model d_v"],
    in_features: Float[Tensor, "... sequence_length d_in"],
) -> Float[Tensor, "... sequence_length d_model"]:
    """Multi-head self-attention without RoPE."""
    assert d_model % num_heads == 0, f"d_model({d_model}) must be divisible by num_heads({num_heads})"
    
    d_k_per_head = d_model // num_heads
    batch_shape = list(in_features.shape[:-2])
    seq_len = in_features.size(-2)
    d_in = in_features.size(-1)
    device = in_features.device
    dtype = in_features.dtype
    
    total_batch = torch.prod(torch.tensor(batch_shape)).item() if batch_shape else 1
    x_flat = in_features.reshape(total_batch, seq_len, d_in)
    
    # Project Q, K, V
    Q = einsum(x_flat, q_proj_weight, "b seq d_in, d_model d_in -> b seq d_model")
    K = einsum(x_flat, k_proj_weight, "b seq d_in, d_model d_in -> b seq d_model")
    V = einsum(x_flat, v_proj_weight, "b seq d_in, d_model d_in -> b seq d_model")
    
    # Reshape to multi-head format
    Q = Q.reshape(total_batch, seq_len, num_heads, d_k_per_head).transpose(1, 2)
    K = K.reshape(total_batch, seq_len, num_heads, d_k_per_head).transpose(1, 2)
    V = V.reshape(total_batch, seq_len, num_heads, d_k_per_head).transpose(1, 2)
    
    # Apply causal mask
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype))
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    
    # Attention
    attn_heads = run_scaled_dot_product_attention(Q, K, V, causal_mask)
    
    # Concatenate heads and project
    attn_concat = attn_heads.transpose(1, 2).reshape(total_batch, seq_len, d_model)
    output = einsum(attn_concat, o_proj_weight, "b seq d_v, d_model d_v -> b seq d_model")
    output = output.reshape(*batch_shape, seq_len, d_model)
    
    return output


# ============================================================================
# Rotary Position Embedding (RoPE)
# ============================================================================

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding."""
    
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even, but got {d_k}")
        
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        angles = self._compute_angles(max_seq_len, d_k, theta)
        self.register_buffer('cos_cached', torch.cos(angles), persistent=False)
        self.register_buffer('sin_cached', torch.sin(angles), persistent=False)
    
    def _compute_angles(self, max_seq_len: int, d_k: int, theta: float):
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        k_indices = torch.arange(0, d_k, 2, dtype=torch.float32) / d_k
        positions = positions.unsqueeze(-1)
        k_indices = k_indices.unsqueeze(0)
        angles = positions / (theta ** (2 * k_indices))
        return angles
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        batch_shape = x.shape[:-2]
        seq_len = x.shape[-2]
        d_k = x.shape[-1]
        
        if d_k != self.d_k:
            raise ValueError(f"Input dimension {d_k} doesn't match initialized d_k {self.d_k}")
        
        x_flat = x.reshape(-1, seq_len, d_k)
        pos_flat = token_positions.reshape(-1, seq_len)
        
        cos = self.cos_cached[pos_flat]
        sin = self.sin_cached[pos_flat]
        
        cos_expanded = cos.unsqueeze(-1).expand(-1, -1, -1, 2).reshape(-1, seq_len, d_k)
        sin_expanded = sin.unsqueeze(-1).expand(-1, -1, -1, 2).reshape(-1, seq_len, d_k)
        
        x_even = x_flat[..., 0::2]
        x_odd = x_flat[..., 1::2]
        cos_ = cos_expanded[..., 0::2]
        sin_ = sin_expanded[..., 0::2]
        
        x_even_rotated = cos_ * x_even - sin_ * x_odd
        x_odd_rotated = sin_ * x_even + cos_ * x_odd
        
        x_rotated = torch.zeros_like(x_flat)
        x_rotated[..., 0::2] = x_even_rotated
        x_rotated[..., 1::2] = x_odd_rotated
        x_rotated = x_rotated.reshape(*batch_shape, seq_len, d_k)
        
        return x_rotated


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, "... sequence_length d_k"],
    token_positions: Int[Tensor, "... sequence_length"],
    cos_cache=None,
    sin_cache=None,
) -> Float[Tensor, "... sequence_length d_k"]:
    """Apply Rotary Position Embedding."""
    device, dtype = in_query_or_key.device, in_query_or_key.dtype
    
    if cos_cache is None or sin_cache is None:
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        k_indices = torch.arange(0, d_k, 2, dtype=torch.float32, device=device) / d_k
        angles = positions.unsqueeze(-1) / (theta ** k_indices.unsqueeze(0))
        cos_vals = torch.cos(angles).to(dtype)
        sin_vals = torch.sin(angles).to(dtype)
    else:
        cos_vals = cos_cache
        sin_vals = sin_cache
    
    batch_shape = in_query_or_key.shape[:-2]
    seq_len = in_query_or_key.shape[-2]
    
    x_flat = in_query_or_key.reshape(-1, seq_len, d_k)
    pos_flat = token_positions.reshape(-1, seq_len)
    
    cos = cos_vals[pos_flat]
    sin = sin_vals[pos_flat]
    
    cos_expanded = cos.unsqueeze(-1).expand(-1, -1, -1, 2).reshape(-1, seq_len, d_k)
    sin_expanded = sin.unsqueeze(-1).expand(-1, -1, -1, 2).reshape(-1, seq_len, d_k)
    
    x_even = x_flat[..., 0::2]
    x_odd = x_flat[..., 1::2]
    cos_ = cos_expanded[..., 0::2]
    sin_ = sin_expanded[..., 0::2]
    
    x_even_rotated = cos_ * x_even - sin_ * x_odd
    x_odd_rotated = sin_ * x_even + cos_ * x_odd
    
    x_rotated = torch.zeros_like(x_flat)
    x_rotated[..., 0::2] = x_even_rotated
    x_rotated[..., 1::2] = x_odd_rotated
    x_rotated = x_rotated.reshape(*batch_shape, seq_len, d_k)
    
    return x_rotated


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, "d_k d_in"],
    k_proj_weight: Float[Tensor, "d_k d_in"],
    v_proj_weight: Float[Tensor, "d_v d_in"],
    o_proj_weight: Float[Tensor, "d_model d_v"],
    in_features: Float[Tensor, "... sequence_length d_in"],
    token_positions: Int[Tensor, "... sequence_length"] | None = None,
    rope_cos_cache=None,
    rope_sin_cache=None
) -> Float[Tensor, "... sequence_length d_out"]:
    """Multi-head self-attention with RoPE."""
    assert d_model % num_heads == 0, f"d_model({d_model}) must be divisible by num_heads({num_heads})"
    
    d_k_per_head = d_model // num_heads
    device = in_features.device
    dtype = in_features.dtype
    batch_shape = list(in_features.shape[:-2])
    seq_len = in_features.size(-2)
    d_in = in_features.size(-1)
    
    if token_positions is None:
        pos_shape = (*batch_shape, seq_len)
        token_positions = torch.arange(seq_len, device=device).expand(pos_shape).int()
    else:
        token_positions = token_positions.repeat(*(batch_shape + [1])).int()
    
    total_batch = torch.prod(torch.tensor(batch_shape)).item() if batch_shape else 1
    x_flat = in_features.reshape(total_batch, seq_len, d_in)
    pos_flat = token_positions.reshape(total_batch, seq_len)
    
    # Project Q, K, V
    Q = einsum(x_flat, q_proj_weight, "b seq d_in, d_model d_in -> b seq d_model")
    K = einsum(x_flat, k_proj_weight, "b seq d_in, d_model d_in -> b seq d_model")
    V = einsum(x_flat, v_proj_weight, "b seq d_in, d_model d_in -> b seq d_model")
    
    # Reshape to multi-head format
    Q = Q.reshape(total_batch, seq_len, num_heads, d_k_per_head).transpose(1, 2)
    K = K.reshape(total_batch, seq_len, num_heads, d_k_per_head).transpose(1, 2)
    V = V.reshape(total_batch, seq_len, num_heads, d_k_per_head).transpose(1, 2)
    
    # Apply RoPE
    Q_rope_input = Q.reshape(-1, seq_len, d_k_per_head)
    K_rope_input = K.reshape(-1, seq_len, d_k_per_head)
    pos_rope_input = pos_flat.unsqueeze(1).expand(-1, num_heads, -1).reshape(-1, seq_len)
    
    Q_rotated = run_rope(d_k_per_head, theta, max_seq_len, Q_rope_input, pos_rope_input, 
                        rope_cos_cache, rope_sin_cache)
    K_rotated = run_rope(d_k_per_head, theta, max_seq_len, K_rope_input, pos_rope_input, 
                        rope_cos_cache, rope_sin_cache)
    
    Q = Q_rotated.reshape(total_batch, num_heads, seq_len, d_k_per_head)
    K = K_rotated.reshape(total_batch, num_heads, seq_len, d_k_per_head)
    
    # Apply causal mask
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype))
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    
    # Attention
    attn_heads = run_scaled_dot_product_attention(Q, K, V, causal_mask)
    
    # Concatenate heads and project
    attn_concat = attn_heads.transpose(1, 2).reshape(total_batch, seq_len, d_model)
    output = einsum(attn_concat, o_proj_weight, "b seq d_v, d_model d_v -> b seq d_model")
    output = output.reshape(*batch_shape, seq_len, d_model)
    
    return output


# ============================================================================
# Transformer Block
# ============================================================================

def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, "batch sequence_length d_model"],
    rope_cos_cache=None,
    rope_sin_cache=None
) -> Float[Tensor, "batch sequence_length d_model"]:
    """Single transformer block with pre-norm architecture."""
    eps = 1e-5
    
    # Extract weights
    q_proj = weights['attn.q_proj.weight']
    k_proj = weights['attn.k_proj.weight']
    v_proj = weights['attn.v_proj.weight']
    o_proj = weights['attn.output_proj.weight']
    ln1 = weights['ln1.weight']
    ffn_w1 = weights['ffn.w1.weight']
    ffn_w2 = weights['ffn.w2.weight']
    ffn_w3 = weights['ffn.w3.weight']
    ln2 = weights['ln2.weight']
    
    # Attention block
    normalized = run_rmsnorm(d_model, eps, ln1, in_features)
    attn_out = run_multihead_self_attention_with_rope(
        d_model, num_heads, max_seq_len, theta,
        q_proj, k_proj, v_proj, o_proj, normalized,
        rope_cos_cache=rope_cos_cache, rope_sin_cache=rope_sin_cache
    )
    in_features = attn_out + in_features
    
    # FFN block
    normalized = run_rmsnorm(d_model, eps, ln2, in_features)
    ffn_out = run_swiglu(d_model, d_ff, ffn_w1, ffn_w2, ffn_w3, normalized)
    in_features = ffn_out + in_features
    
    return in_features


# ============================================================================
# Transformer Language Model
# ============================================================================

class TransformerLM(nn.Module):
    """Transformer language model."""
    
    def __init__(self, vocab_size: int, d_model: int, num_layers: int,
                 num_heads: int, d_ff: int, context_length: int,
                 rope_theta: float, device, dtype):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.context_length = context_length
        self.rope_theta = rope_theta
        self.device = device
        self.dtype = dtype
        self.d_k = d_model // num_heads
        
        # Pre-compute RoPE cache
        positions = torch.arange(context_length, device=device, dtype=dtype)
        k_indices = torch.arange(0, self.d_k, 2, device=device, dtype=dtype) / self.d_k
        angles = positions.unsqueeze(-1) / (rope_theta ** k_indices.unsqueeze(0))
        self.register_buffer('rope_cos_cache', torch.cos(angles))
        self.register_buffer('rope_sin_cache', torch.sin(angles))
        
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize all model weights."""
        weights = {}
        
        # Token embeddings
        weights["token_embeddings.weight"] = nn.Parameter(
            Embedding(self.vocab_size, self.d_model, self.device, self.dtype).embedding
        )
        
        # Transformer layers
        for i in range(self.num_layers):
            prefix = f"layers.{i}."
            
            # Attention projections
            for proj in ["q_proj", "k_proj", "v_proj", "output_proj"]:
                weights[prefix + f"attn.{proj}.weight"] = nn.Parameter(
                    Linear(self.d_model, self.d_model, self.device, self.dtype).W
                )
            
            # Layer norms
            weights[prefix + "ln1.weight"] = nn.Parameter(
                RMSNorm(self.d_model, 1e-5, self.device, self.dtype).gain_parameter
            )
            weights[prefix + "ln2.weight"] = nn.Parameter(
                RMSNorm(self.d_model, 1e-5, self.device, self.dtype).gain_parameter
            )
            
            # FFN weights
            weights[prefix + "ffn.w1.weight"] = nn.Parameter(
                Linear(self.d_model, self.d_ff, self.device, self.dtype).W
            )
            weights[prefix + "ffn.w2.weight"] = nn.Parameter(
                Linear(self.d_ff, self.d_model, self.device, self.dtype).W
            )
            weights[prefix + "ffn.w3.weight"] = nn.Parameter(
                Linear(self.d_model, self.d_ff, self.device, self.dtype).W
            )
        
        # Final layer norm and output head
        weights["ln_final.weight"] = nn.Parameter(
            RMSNorm(self.d_model, 1e-5, self.device, self.dtype).gain_parameter
        )
        weights["lm_head.weight"] = nn.Parameter(
            Linear(self.vocab_size, self.d_model, self.device, self.dtype).W
        )
        
        # Register all parameters
        for key, param in weights.items():
            attr_name = key.replace('.', '_')
            self.register_parameter(attr_name, param)
        
        self._weight_keys = list(weights.keys())
    
    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Get all weights as a dictionary."""
        weights = {}
        for key in self._weight_keys:
            attr_name = key.replace('.', '_')
            weights[key] = getattr(self, attr_name)
        return weights
    
    def forward(self, in_indices):
        """Forward pass through the model."""
        return run_transformer_lm(
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            rope_theta=self.rope_theta,
            weights=self.get_weights(),
            in_indices=in_indices,
            rope_cos_cache=self.rope_cos_cache,
            rope_sin_cache=self.rope_sin_cache
        )
    
    def __repr__(self) -> str:
        return (f"TransformerLM(vocab={self.vocab_size}, d_model={self.d_model}, "
                f"layers={self.num_layers}, heads={self.num_heads}, "
                f"context_len={self.context_length})")


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, "batch_size sequence_length"],
    rope_cos_cache=None,
    rope_sin_cache=None
) -> Float[Tensor, "batch_size sequence_length vocab_size"]:
    """Run forward pass through transformer language model."""
    eps = 1e-5
    
    # Embed tokens
    embedding = weights['token_embeddings.weight']
    in_features = run_embedding(vocab_size, d_model, embedding, in_indices)
    
    # Apply transformer blocks
    for i in range(num_layers):
        block_weights = {}
        prefix = f'layers.{i}.'
        weight_names = [
            'attn.q_proj.weight', 'attn.k_proj.weight', 'attn.v_proj.weight',
            'attn.output_proj.weight', 'ln1.weight', 'ln2.weight',
            'ffn.w1.weight', 'ffn.w2.weight', 'ffn.w3.weight'
        ]
        
        for name in weight_names:
            block_weights[name] = weights[prefix + name]
        
        in_features = run_transformer_block(
            d_model, num_heads, d_ff, context_length, rope_theta,
            block_weights, in_features,
            rope_cos_cache=rope_cos_cache,
            rope_sin_cache=rope_sin_cache
        )
    
    # Final layer norm and output projection
    ln_final = weights['ln_final.weight']
    lm_head = weights['lm_head.weight']
    in_features = run_rmsnorm(d_model, eps, ln_final, in_features)
    
    return in_features @ lm_head


# ============================================================================
# Loss Functions
# ============================================================================

def run_softmax(in_features: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    """Numerically stable softmax."""
    x_max = in_features.max(dim=dim, keepdim=True)[0]
    x_exp = torch.exp(in_features - x_max)
    x_sum = x_exp.sum(dim=dim, keepdim=True)
    return x_exp / x_sum


def run_cross_entropy(
    inputs: Float[Tensor, "batch_size vocab_size"],
    targets: Int[Tensor, "batch_size"]
) -> Float[Tensor, ""]:
    """Compute cross-entropy loss."""
    # Numerically stable log-softmax
    max_vals = inputs.max(dim=1, keepdim=True).values
    inputs_stable = inputs - max_vals
    exp_vals = torch.exp(inputs_stable)
    sum_exp_vals = exp_vals.sum(dim=1, keepdim=True)
    log_softmax = inputs_stable - torch.log(sum_exp_vals)
    
    # Gather target log probabilities
    target_log_probs = log_softmax.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
    
    return -target_log_probs.mean()


# ============================================================================
# Training Utilities
# ============================================================================

def run_get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample training batch from dataset."""
    assert len(dataset) >= context_length + 1, \
        f"Dataset length {len(dataset)} must be >= context_length + 1 ({context_length + 1})"
    
    max_start_idx = len(dataset) - context_length
    start_indices = np.random.randint(0, max_start_idx, size=batch_size)
    offsets = np.arange(context_length)
    
    input_indices = start_indices[:, np.newaxis] + offsets
    label_indices = input_indices + 1
    
    input_sequences = dataset[input_indices]
    labels = dataset[label_indices]
    
    input_tensor = torch.from_numpy(input_sequences).long().to(device)
    label_tensor = torch.from_numpy(labels).long().to(device)
    
    return input_tensor, label_tensor


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Clip gradients by global L2 norm."""
    eps = 1e-6
    grads = [p.grad.flatten() for p in parameters if p.grad is not None]
    
    if not grads:
        return
    
    all_grads = torch.cat(grads)
    total_l2_norm = torch.norm(all_grads, p=2)
    
    if total_l2_norm > max_l2_norm:
        scale_factor = max_l2_norm / (total_l2_norm + eps)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(scale_factor)


def get_adamw_cls() -> Any:
    """Returns AdamW optimizer class."""
    
    class AdamW(torch.optim.Optimizer):
        def __init__(
            self,
            params: Iterable[torch.nn.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 1e-2,
        ):
            if not 0.0 <= lr:
                raise ValueError(f"Invalid learning rate: {lr}")
            if not 0.0 <= betas[0] < 1.0:
                raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
            if not 0.0 <= betas[1] < 1.0:
                raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
            if not 0.0 <= eps:
                raise ValueError(f"Invalid epsilon value: {eps}")
            if not 0.0 <= weight_decay:
                raise ValueError(f"Invalid weight decay value: {weight_decay}")
            
            defaults = {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
            }
            super().__init__(params, defaults)
        
        def step(self, closure: Optional[callable] = None) -> Optional[float]:
            """Performs a single optimization step."""
            loss = None
            if closure is not None:
                loss = closure()
            
            for group in self.param_groups:
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    
                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError("AdamW does not support sparse gradients")
                    
                    state = self.state[p]
                    
                    # Initialize state
                    if len(state) == 0:
                        state["step"] = 0
                        state["m"] = torch.zeros_like(p.data)
                        state["v"] = torch.zeros_like(p.data)
                    
                    m = state["m"]
                    v = state["v"]
                    state["step"] += 1
                    t = state["step"]
                    
                    # Update biased first moment estimate
                    m.mul_(beta1).add_(grad, alpha=1 - beta1)
                    
                    # Update biased second moment estimate
                    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    
                    # Compute bias-corrected learning rate
                    beta1_t = beta1 ** t
                    beta2_t = beta2 ** t
                    alpha_t = lr * math.sqrt(1 - beta2_t) / (1 - beta1_t)
                    
                    # Update parameters
                    denom = torch.sqrt(v) + eps
                    p.data.addcdiv_(m, denom, value=-alpha_t)
                    
                    # Apply weight decay
                    if weight_decay != 0:
                        p.data.add_(p.data, alpha=-lr * weight_decay)
            
            return loss
    
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if it < warmup_iters:
        return it * max_learning_rate / warmup_iters
    elif warmup_iters <= it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * \
               (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate


# ============================================================================
# Checkpoint Management
# ============================================================================

def run_save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: Union[str, os.PathLike, BinaryIO, IO[bytes]]
) -> None:
    """Save training checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)


def run_load_checkpoint(
    src: Union[str, os.PathLike, BinaryIO, IO[bytes]],
    model: nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    """Load training checkpoint."""
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']


# ============================================================================
# Text Generation
# ============================================================================

def generate_with_sliding_window(
    model: torch.nn.Module,
    prompt: list[int],
    temperature: float,
    context_length: int,
    max_tokens: int,
    device: str
) -> list[int]:
    """Generate text using sliding window for long sequences."""
    model.eval()
    model.to(device)
    current_sequence = prompt.copy()
    generated_tokens = prompt.copy()
    
    with torch.no_grad():
        for _ in range(max_tokens):
            seq_len = len(current_sequence)
            
            if seq_len <= context_length:
                input_sequence = current_sequence + [0] * (context_length - seq_len)
                pred_position = seq_len - 1
            else:
                input_sequence = current_sequence[-context_length:]
                pred_position = context_length - 1
            
            input_tensor = torch.tensor([input_sequence], dtype=torch.long, device=device)
            logits = model(input_tensor)
            next_token_logits = logits[0, pred_position, :]
            
            probs = F.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            generated_tokens.append(next_token)
            current_sequence.append(next_token)
    
    return generated_tokens


def generate_little_story(
    model: torch.nn.Module,
    prompt: list[int],
    temperature: float,
    context_length: int,
    device: str,
    special_token_id: int
) -> list[int]:
    """Generate text until special token is encountered."""
    model.eval()
    model.to(device)
    current_sequence = prompt.copy()
    generated_tokens = prompt.copy()
    
    with torch.no_grad():
        while True:
            seq_len = len(current_sequence)
            
            if seq_len <= context_length:
                input_sequence = current_sequence + [0] * (context_length - seq_len)
                pred_position = seq_len - 1
            else:
                input_sequence = current_sequence[-context_length:]
                pred_position = context_length - 1
            
            input_tensor = torch.tensor([input_sequence], dtype=torch.long, device=device)
            logits = model(input_tensor)
            next_token_logits = logits[0, pred_position, :]
            
            probs = F.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            if next_token == special_token_id:
                break
            
            generated_tokens.append(next_token)
            current_sequence.append(next_token)
    
    return generated_tokens


# ============================================================================
# Tokenizer
# ============================================================================

def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Create BPE tokenizer."""
    
    class Tokenizer:
        def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]],
                     special_tokens: Optional[List[str]] = None):
            if special_tokens:
                special_tokens.sort(key=lambda x: -len(x))
            
            self.vocab = vocab.copy()
            self.merges = merges.copy()
            self.byte_to_id = {b: i for i, b in self.vocab.items()}
            self._add_special_tokens(special_tokens or [])
            self.merge_set = set(self.merges)
            self.vocab_list = set(self.vocab.values())
            self.special_tokens = special_tokens
            
            # Pre-tokenization pattern
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            self._regex = re.compile(PAT, re.UNICODE)
            
            self.exist_special_tokens = False
            if special_tokens is not None:
                escaped_seps = [re.escape(sep) for sep in special_tokens]
                pattern = '|'.join([f'({sep})' for sep in escaped_seps])
                self.exist_special_tokens = True
                self.pattern = pattern
                self.byte_special_tokens = [x.encode() for x in special_tokens]
        
        def _add_special_tokens(self, special_tokens: List[str]) -> None:
            """Add special tokens to vocabulary."""
            if not special_tokens:
                return
            
            max_existing_id = max(self.vocab.keys()) if self.vocab else 0
            next_id = max_existing_id + 1
            
            for token in special_tokens:
                token_bytes = token.encode("utf-8")
                if token_bytes not in self.byte_to_id:
                    self.vocab[next_id] = token_bytes
                    self.byte_to_id[token_bytes] = next_id
                    self.vocab_list.add(token_bytes)
                    next_id += 1
        
        @classmethod
        def from_files(cls, vocab_filepath: str, merges_filepath: str,
                      special_tokens: Optional[List[str]] = None) -> "Tokenizer":
            """Load tokenizer from files."""
            vocab = {}
            with open(vocab_filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    idx_str, bytes_str = line.split("\t", 1)
                    idx = int(idx_str)
                    byte_seq = eval(bytes_str) if bytes_str.startswith("b'") else bytes_str.encode()
                    vocab[idx] = byte_seq
            
            merges = []
            with open(merges_filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    b1_str, b2_str = line.split("\t", 1)
                    b1 = eval(b1_str) if b1_str.startswith("b'") else b1_str.encode()
                    b2 = eval(b2_str) if b2_str.startswith("b'") else b2_str.encode()
                    merges.append((b1, b2))
            
            return cls(vocab, merges, special_tokens)
        
        def _apply_bpe(self, byte_seq: List[bytes]) -> List[bytes]:
            """Apply BPE merges to byte sequence."""
            if self.exist_special_tokens:
                if len(byte_seq) <= 1 or b''.join(byte_seq) in self.byte_special_tokens:
                    return [b''.join(byte_seq)]
            else:
                if len(byte_seq) <= 1:
                    return byte_seq
            
            merge_rank = {pair: i for i, pair in enumerate(self.merges)}
            heap = []
            
            # Find initial mergeable pairs
            for i in range(len(byte_seq) - 1):
                pair = (byte_seq[i], byte_seq[i + 1])
                if pair in merge_rank:
                    heapq.heappush(heap, (merge_rank[pair], i))
            
            deleted = set()
            
            while heap:
                rank, pos = heapq.heappop(heap)
                
                if pos in deleted or pos + 1 in deleted:
                    continue
                
                pair = (byte_seq[pos], byte_seq[pos + 1])
                if pair not in merge_rank or merge_rank[pair] != rank:
                    continue
                
                byte_seq[pos] = byte_seq[pos] + byte_seq[pos + 1]
                deleted.add(pos + 1)
                
                # Check left neighbor
                if pos > 0 and pos - 1 not in deleted:
                    new_pair = (byte_seq[pos - 1], byte_seq[pos])
                    if new_pair in merge_rank:
                        heapq.heappush(heap, (merge_rank[new_pair], pos - 1))
                
                # Check right neighbor
                if pos + 2 < len(byte_seq) and pos + 2 not in deleted:
                    new_pair = (byte_seq[pos], byte_seq[pos + 2])
                    if new_pair in merge_rank:
                        heapq.heappush(heap, (merge_rank[new_pair], pos))
            
            return [byte_seq[i] for i in range(len(byte_seq)) if i not in deleted]
        
        def _pre_tokenize(self, text: str) -> List[str]:
            """Pre-tokenize text into subwords."""
            new_text = []
            special_tokens_splited_text = (re.split(self.pattern, text) 
                                          if self.exist_special_tokens else [text])
            
            for sub_text in special_tokens_splited_text:
                if sub_text is None:
                    continue
                if self.exist_special_tokens and sub_text in self.special_tokens:
                    new_text.append(sub_text)
                else:
                    new_text += self._regex.findall(sub_text)
            
            return new_text
        
        def encode(self, text: str) -> List[int]:
            """Encode text to token IDs."""
            # Add new special tokens if needed
            new_special_token = []
            for x in text:
                if x.encode() not in self.vocab_list and x not in new_special_token:
                    new_special_token.append(x)
            self._add_special_tokens(new_special_token)
            
            token_ids = []
            pre_tokens = self._pre_tokenize(text)
            
            for pt in pre_tokens:
                byte_list = [bytes([b]) for b in pt.encode("utf-8")]
                if not byte_list:
                    continue
                
                merged_bytes = self._apply_bpe(byte_list)
                for b in merged_bytes:
                    token_ids.append(self.byte_to_id[b])
            
            return token_ids
        
        def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
            """Encode iterable of text chunks."""
            buffer = ""
            for chunk in iterable:
                # Add new special tokens
                new_special_token = []
                for x in chunk:
                    if x.encode() not in self.vocab_list and x not in new_special_token:
                        new_special_token.append(x)
                self._add_special_tokens(new_special_token)
                
                full_text = buffer + chunk
                pre_tokens = self._pre_tokenize(full_text)
                
                if len(pre_tokens) == 0:
                    buffer = ""
                    continue
                
                for pt in pre_tokens[:-1]:
                    byte_list = [bytes([b]) for b in pt.encode("utf-8")]
                    merged_bytes = self._apply_bpe(byte_list)
                    for b in merged_bytes:
                        yield self.byte_to_id[b]
                
                buffer = pre_tokens[-1]
            
            if buffer:
                byte_list = [bytes([b]) for b in buffer.encode("utf-8")]
                merged_bytes = self._apply_bpe(byte_list)
                for b in merged_bytes:
                    yield self.byte_to_id[b]
        
        def decode(self, ids: List[int]) -> str:
            """Decode token IDs to text."""
            byte_list = []
            replacement_byte = '\uFFFD'.encode('utf-8')
            
            for idx in ids:
                if idx in self.vocab:
                    byte_list.append(self.vocab[idx])
                else:
                    byte_list.append(replacement_byte)
            
            all_bytes = b''.join(byte_list)
            return all_bytes.decode("utf-8", errors="replace")
    
    return Tokenizer(vocab, merges, special_tokens)


# ============================================================================
# BPE Training
# ============================================================================

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    max_len=None,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train BPE tokenizer."""
    # Initialize vocab
    vocab = {i: bytes([i]) for i in range(256)}
    to_num = {bytes([i]): i for i in range(256)}
    idx = 256
    
    # Add special tokens
    for token in special_tokens:
        token_bytes = token.encode()
        vocab[idx] = token_bytes
        to_num[token_bytes] = idx
        idx += 1
    
    # Prepare regex patterns
    escaped_special = [re.escape(char) for char in special_tokens]
    split_pattern = '|'.join(escaped_special) if escaped_special else None
    pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    # Read file
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read(max_len) if max_len else f.read()
    
    # Split text
    if split_pattern:
        split_text = re.split(split_pattern, text)
    else:
        split_text = [text]
    
    # Further split and encode
    split_2_text = []
    for sub_text in split_text:
        for match in re.finditer(pat, sub_text):
            split_2_text.append([bytes([b]) for b in match.group().encode()])
    
    epoch = vocab_size - idx
    split_2_text, merges, vocab = _find_biggest_appear_and_transform_list(
        split_2_text, epoch, vocab, idx
    )
    
    return vocab, merges


def _find_biggest_appear_and_transform_list(split_text, epoch, vocab, idx):
    """Core BPE merge algorithm using priority queue."""
    count_dict = defaultdict(int)
    pair_to_texts = defaultdict(set)
    
    # Initial pair counting
    for text_idx, text in enumerate(split_text):
        for i in range(len(text) - 1):
            pair = (text[i], text[i + 1])
            count_dict[pair] += 1
            pair_to_texts[pair].add(text_idx)
    
    heap = [(-count, pair) for pair, count in count_dict.items() if count > 0]
    heapq.heapify(heap)
    
    merges = []
    
    for iteration in range(epoch):
        # Find valid maximum
        max_key = None
        while heap:
            neg_count, candidate_pair = heapq.heappop(heap)
            if -neg_count == count_dict[candidate_pair] and count_dict[candidate_pair] > 0:
                max_key = candidate_pair
                break
        
        if max_key is None:
            break
        
        byte1, byte2 = max_key
        merges.append(max_key)
        new_word = byte1 + byte2
        vocab[idx] = new_word
        
        # Update affected texts
        affected_text_indices = pair_to_texts[max_key].copy()
        new_affected_pairs = defaultdict(int)
        new_pair_to_texts = defaultdict(set)
        
        for text_idx in affected_text_indices:
            text = split_text[text_idx]
            
            if len(text) <= 1:
                continue
            
            new_text = []
            i = 0
            
            while i < len(text):
                if i < len(text) - 1 and (text[i], text[i + 1]) == max_key:
                    # Update left neighbor
                    if i > 0:
                        old_left_pair = (text[i - 1], text[i])
                        count_dict[old_left_pair] -= 1
                        new_affected_pairs[old_left_pair] -= 1
                        pair_to_texts[old_left_pair].discard(text_idx)
                        
                        new_left_pair = (text[i - 1], new_word)
                        new_affected_pairs[new_left_pair] += 1
                        new_pair_to_texts[new_left_pair].add(text_idx)
                    
                    # Update right neighbor
                    if i < len(text) - 2:
                        old_right_pair = (text[i + 1], text[i + 2])
                        count_dict[old_right_pair] -= 1
                        new_affected_pairs[old_right_pair] -= 1
                        pair_to_texts[old_right_pair].discard(text_idx)
                        
                        new_right_pair = (new_word, text[i + 2])
                        new_affected_pairs[new_right_pair] += 1
                        new_pair_to_texts[new_right_pair].add(text_idx)
                    
                    new_text.append(new_word)
                    i += 2
                else:
                    new_text.append(text[i])
                    i += 1
            
            split_text[text_idx] = new_text
        
        # Clear merged pair
        count_dict[max_key] = 0
        pair_to_texts[max_key].clear()
        
        # Batch update counts
        for pair, delta in new_affected_pairs.items():
            count_dict[pair] += delta
            if pair in new_pair_to_texts:
                pair_to_texts[pair].update(new_pair_to_texts[pair])
            
            if count_dict[pair] > 0:
                heapq.heappush(heap, (-count_dict[pair], pair))
        
        idx += 1
    
    return split_text, merges, vocab


def write_vocab_merges(vocab: Dict, merge: List, file_path: str):
    """Write vocabulary and merges to files."""
    with open(f'{file_path}/vocab.txt', 'w', encoding='utf-8') as f:
        for x in vocab.keys():
            f.write(f'{x}\t{vocab[x]}\n')
    
    with open(f'{file_path}/merges.txt', 'w', encoding='utf-8') as f:
        for byte1, byte2 in merge:
            f.write(f'{byte1}\t{byte2}\n')


def read_vocab_merges(file_path: str):
    """Read vocabulary and merges from files."""
    vocab = {}
    merge = []
    
    with open(f'{file_path}/vocab.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    key, value = parts
                    key = int(key)
                    value = eval(value)
                    vocab[key] = value
    
    with open(f'{file_path}/merges.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    byte1, byte2 = parts
                    byte1 = eval(byte1)
                    byte2 = eval(byte2)
                    merge.append((byte1, byte2))
    
    return vocab, merge