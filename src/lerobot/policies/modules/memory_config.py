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
    # Depth of the state-path query projection. 1 (default) = the original single
    # linear (byte-identical). >=2 inserts SiLU-separated hidden layers of width
    # query_proj_hidden_dim (0 -> input_dim) before the final projection to
    # heads*k_dim. Motivation: on a FROZEN backbone (train_memory_only staged
    # pretraining) the routing losses cannot separate tasks by co-adapting hidden
    # states; a nonlinear query head restores a learnable routing surface.
    query_proj_layers: int = 1
    query_proj_hidden_dim: int = 0

    # Frozen-base routing (dual-path). When True, memory ROUTING (query projection
    # + gate) reads the backbone features as they would be WITHOUT any memory
    # contribution (a memory-free forward of the frozen expert from the first
    # memory layer onward), while the value/output path still acts on the live
    # residual stream. Motivation (research_log E38): with routing computed from
    # the live stream, sequential value training perturbs the stream and re-points
    # the frozen router above the first memory layer (measured: per-task footprint
    # IoU 0.21-0.33 at L10-14 after 7 blocks vs exactly 1.0 at L8, whose input is
    # immutable) — every earlier task's reads slide off the slots it adapted.
    # Frozen-base routing makes the addressing stationary by construction at every
    # memory layer. Default False = original behavior, byte-identical.
    use_frozen_base_input_features: bool = False

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
    # Optional cross-batch FIFO queue size for sample-wise contrastive.
    # When > 0 and contrastive_method="sample", each layer keeps the latest
    # detached per-sample query vectors and task_ids to increase the pool of
    # negatives/positives without increasing the micro-batch size.
    # 0 preserves the original in-batch-only behavior.
    contrastive_query_queue: int = 0

    # Optional cross-batch FIFO queue (in SAMPLES) of per-token routing queries,
    # used to estimate per-task reference slot-distributions for the inter-task
    # separation loss. Without it, separation only sees the ~B tasks present in
    # the current micro-batch (~1 sample/task when batch_size << num_tasks), so
    # the per-task histograms are noisy and most task pairs are never co-present.
    # When > 0, each step recomputes the queued queries' routing against the
    # CURRENT keys (no grad to the stored queries) to form detached references
    # covering all recently-seen tasks; the current batch is pushed away from
    # them. 0 preserves the original current-batch-only behavior.
    routing_query_queue: int = 0

    # Routing regularizers operate on the joint product-key candidate distribution:
    # 1. Take top-M subkeys in each PQ half (M = routing_loss_topk or mem_knn)
    # 2. Form the M×M Cartesian-product candidate slots (matching retrieval)
    # 3. Softmax over those M² joint candidate scores per sample per head
    # 4. Scatter per-task distributions into the full slot space (n_keys²)
    # 5. Compute locality/separation losses on those full-slot distributions
    #
    # This directly regularizes the slot distribution that retrieval actually
    # uses, rather than the dense PQ-half marginals which can be satisfied by
    # moving tail mass without affecting the retrieved slots.
    #
    # - intra-task locality: keep each task's effective final-slot support in a target range
    # - inter-task separation: penalize overlap between different task slot distributions
    # - global_balance: penalize collapse of the aggregate distribution
    routing_intra_task_locality_weight: float = 0.0
    routing_inter_task_separation_weight: float = 0.0
    # Number of top subkeys per PQ half used to form joint candidates for the
    # routing loss. 0 means use mem_knn (matching retrieval exactly).
    routing_loss_topk: int = 0
    # Effective support bounds for the intra-task locality loss, measured over
    # actual final product-key slot ids (in the n_keys² space) per head.
    # 0 means "use a heuristic default":
    # - min_support -> max(mem_knn, 8)
    # - max_support -> max(8 * mem_knn, candidate_pool // 2)
    routing_intra_task_min_support: int = 0
    routing_intra_task_max_support: int = 0
    # Deprecated aliases preserved for backward compatibility with existing scripts.
    routing_compactness_weight: float = 0.0
    routing_separation_weight: float = 0.0
    routing_global_balance_weight: float = 0.0

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
    # Optional per-layer LoRA ranks (only used when value_type="lora"). When non-empty,
    # its length must equal the number of attached expert `layers` and it is matched to
    # `layers` by order, overriding the scalar `lora_rank` per layer. Empty -> use the
    # scalar `lora_rank` for every layer (backward compatible).
    layer_ranks: List[int] = field(default_factory=list)

    # ---- VLM-side (prefix / paligemma language-model) memory layers (E44 build) ----
    # When vlm_layers is non-empty, an ADDITIONAL set of memory modules is attached to the
    # listed paligemma language_model layers, with their OWN bank geometry below (the expert
    # fields above keep governing the expert-side modules). The VLM modules attach to the
    # TEXT SPAN only: memory output is added to the last `vlm_text_span` prefix positions
    # (the tokenized language field = instruction + state-as-text in pi05), and retrieval is
    # computed only on that slice — the span carries 36-57% of loss leverage at 20x the
    # per-position density of image tokens, with far more open routing geometry
    # (inter-task cos 0.73-0.86 vs 0.96+; E44 span-attribution + placement probes).
    # Empty list (default) = no VLM memory = byte-identical legacy behavior.
    vlm_layers: List[int] = field(default_factory=list)
    vlm_mem_n_keys: int = 256
    vlm_lora_rank: int = 2
    vlm_mem_knn: int = 16
    vlm_text_span: int = 200  # = pi05 tokenizer_max_length (the whole language field)
    # Pooled routing for the state-as-text sub-span (E45). The language field is two
    # different objects: instruction tokens are task-pure routing keys; state tokens are
    # shared digit vocabulary whose per-token routing sprawls task-generically (the E44
    # sweep failure: state-region famIoU 0.38-0.40 vs instruction 0.21-0.22). Modes:
    #   ""         - legacy per-token routing over the whole span (the swept design)
    #   "anchored" - instruction positions route per-token; every position from the
    #                ", State:" boundary on routes with ONE shared per-sample key
    #                a*rms_nrm(mean instr tokens) + b*rms_nrm(mean state tokens)
    #                (task anchor + smooth state offset; value path stays per-position)
    #   "state"    - as "anchored" but the shared key is rms_nrm(mean state tokens) only
    # Component RMS-normalization corrects the coherence-dependent norm shrinkage of
    # pooled means (a mean of N unit-RMS vectors has norm ~sqrt(rho)).
    vlm_router_pool: str = ""
    vlm_router_pool_weights: List[float] = field(default_factory=lambda: [1.0, 1.0])
    # INTERNAL (set by the attach path on the derived per-side config; not a CLI knob):
    # when > 0 on a module's cfg, that module applies memory to the last-N positions only.
    text_span: int = 0

    # Affine LoRA slots ("lora + value", only used when value_type="lora"). When True,
    # each slot additionally stores a per-slot bias vector b_i (v_dim), added to the
    # slot's LoRA output before the retrieval-weighted sum:
    #   value_i(x) = slot_up_i @ SiLU(slot_down_i @ x) + b_i
    # Motivation (research_log E40): the homogeneous LoRA form has no constant term —
    # the DC component of a task's correction must be approximated through x's stable
    # cone direction, spending rank on it. A per-slot bias (i) learns the average
    # correction fast and directly (no U·V product coupling — robust to diluted
    # write schedules), (ii) frees the rank budget for the state-dependent residual,
    # and (iii) restores the additive-constant function class of value_type="vector"
    # without handing state-conditionality back to the router. Biases init to zero:
    # flag-on at init is numerically identical to flag-off, and legacy checkpoints
    # (no slot_bias key) load cleanly with biases at zero. Tagged pk_value_param so
    # the TF-IDF write mask, protection, and value optimizer groups apply unchanged.
    lora_slot_bias: bool = False

    # Inference-time CPU offload for slot tensors (slot_down/slot_up when value_type="lora",
    # or values when value_type="vector"). When enabled, the slot tensors are pinned in CPU
    # memory; the forward gathers only the retrieved slot indices and transfers that subset
    # to GPU per call. Numerically identical to the on-GPU path. Intended for inference on
    # memory-constrained GPUs — not for training.
    offload_slots_to_cpu: bool = False
