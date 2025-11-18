from dataclasses import dataclass, field
from typing import List


@dataclass
class MemoryLayerConfig:
    """
    Configuration for optional memory layers attached to a policy.

    This controls a lightweight, single-GPU Product-Key-like memory that can be
    attached alongside select MLP layers in an expert transformer.
    """

    # Whether to enable memory layers for the policy
    enabled: bool = False

    # Which expert layers to attach the memory to (indices in expert depth).
    # If empty, the last two expert layers are selected by default at runtime.
    layers: List[int] = field(default_factory=list)

    # Optionally, attach memory to VLM text backbone layers (indices in VLM text depth).
    # If empty, no VLM layers are wrapped.
    vlm_layers: List[int] = field(default_factory=list)

    # Memory architecture parameters (kept simple for single-GPU)
    mem_n_keys: int = 128
    mem_heads: int = 4
    mem_knn: int = 16
    mem_share_values: bool = True  # reserved for parity; not used across modules here
    mem_k_dim: int = 256
    mem_v_dim: int = -1  # -1 -> same as model dim
    swilu_projection: bool = True
    value_fixed_lr: float = 1e-3
    mem_gated: bool = True

    # Optimizer override for memory values (param group)
    memory_lr: float = 1e-3
    memory_weight_decay: float = 0.0

    # Metrics: when true, record per-batch selected slot indices to enable
    # diversity/coverage logging during training (logged via policy.forward).
    log_usage: bool = False

    # When true, accumulate CPU-side usage histograms each step (for offline stats).
    # Not required for TF-IDF gating in sequential training.
    aggregate_usage: bool = False

    # Integration mode: when True, output is memory-only (layer_out = mem(x)).
    # Default False keeps residual addition (layer_out = mlp(x) + mem(x)).
    memory_only: bool = False


