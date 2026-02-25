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
    # If empty, no expert layers are wrapped.
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

    # Language-conditioned query projection: when True, concatenate task embedding
    # to query input before projection, biasing each task toward distinct memory slots.
    lang_to_query: bool = False
    # Sentence-transformers model name for computing task embeddings.
    embedding_model: str = "all-MiniLM-L6-v2"
    # Method for fusing language embeddings into query projection:
    # - "concat": concatenate language embedding to hidden state before projection
    # - "film": apply FiLM modulation (Feature-wise Linear Modulation) after projection
    fuse_method: str = "concat"

    # Dropout probability applied to retrieved memory slots during training.
    # When > 0, randomly drops retrieved slots and renormalizes the remaining weights.
    dropout_prob: float = 0.0

    # Query Contrastive Loss: pushes query representations apart across tasks in a
    # batch to encourage task-specific memory slot usage.
    # Weight λ for query contrastive loss; disabled if 0.
    contrastive_loss_weight: float = 0.0
    # Margin for cosine penalty (0 = no margin, hinge-style loss if > 0).
    contrastive_margin: float = 0.0
    # Method for computing the contrastive loss:
    # - "centroid": compute per-task query centroid, penalize pairwise cosine similarity
    #   between centroids (cheap, stable, captures average routing direction)
    # - "sample": supervised contrastive loss (Khosla et al.) on per-sample query vectors,
    #   pulling same-task samples together and pushing cross-task samples apart
    #   (tighter intra-task clusters, tail-overlap reduction, quadratic in batch size)
    contrastive_method: str = "centroid"
    # When True and contrastive_method="sample", the SupCon denominator only
    # sums over cross-task (negative) pairs instead of all non-self pairs.
    # This removes the intra-class uniformity pressure that can cause
    # representation collapse at high contrastive_loss_weight values.
    contrastive_negatives_only: bool = False

    # Value Vector Corruption: adds Gaussian noise to retrieved values during training
    # to build robustness to value drift during sequential adaptation.
    # Per-slot probability of corruption; disabled if 0.
    corruption_prob: float = 0.0
    # Standard deviation of additive Gaussian noise.
    corruption_std: float = 0.1

    # Value type: determines what each memory slot stores and how it's used.
    # - "vector": each slot is a value vector (original behavior, weighted sum of vectors)
    # - "lora": each slot is a tiny LoRA (low-rank transform), output is weighted sum of LoRA outputs
    value_type: str = "vector"
    # Rank for LoRA-style slots (only used when value_type="lora").
    # Lower rank = fewer params but less capacity per slot.
    lora_rank: int = 2


