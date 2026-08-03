import ast
import contextlib
import json
import logging
import math
import threading
from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from .memory_config import MemoryLayerConfig

logger = logging.getLogger(__name__)


EMBEDDING_DIM_MAP = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "paraphrase-MiniLM-L6-v2": 384,
    "paraphrase-mpnet-base-v2": 768,
}


_CHECKPOINT_RECOMPUTE_STATE = threading.local()


def _is_checkpoint_recompute() -> bool:
    return bool(getattr(_CHECKPOINT_RECOMPUTE_STATE, "active", False))


@contextlib.contextmanager
def _checkpoint_recompute_context():
    prev = _is_checkpoint_recompute()
    _CHECKPOINT_RECOMPUTE_STATE.active = True
    try:
        yield
    finally:
        _CHECKPOINT_RECOMPUTE_STATE.active = prev


def checkpoint_recompute_context_fn():
    """Return forward/recompute contexts for non-reentrant activation checkpointing."""
    return contextlib.nullcontext(), _checkpoint_recompute_context()


class TaskEmbeddingCache:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._encoder = None
        self._cache: dict[str, torch.Tensor] = {}
        self.embedding_dim = EMBEDDING_DIM_MAP.get(model_name, 384)

    def _load_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for language-conditioned memory queries. "
                    "Install it with: pip install sentence-transformers"
                )
            self._encoder = SentenceTransformer(self.model_name, device=self.device)
            self.embedding_dim = self._encoder.get_sentence_embedding_dimension()
        return self._encoder

    def encode(self, task: str) -> torch.Tensor:
        if task in self._cache:
            return self._cache[task]
        encoder = self._load_encoder()
        with torch.no_grad():
            emb = encoder.encode(task, convert_to_tensor=True, show_progress_bar=False)
            emb = emb.to(dtype=torch.float32, device="cpu")
        self._cache[task] = emb
        return emb

    def encode_batch(self, tasks: list[str]) -> torch.Tensor:
        results = []
        for t in tasks:
            results.append(self.encode(t))
        return torch.stack(results, dim=0)

    def precompute_from_metadata(self, dataset_meta) -> None:
        if dataset_meta.tasks is None:
            return
        for task_name in dataset_meta.tasks.index:
            self.encode(str(task_name))

    def get_by_indices(self, task_names: list[str]) -> torch.Tensor:
        return self.encode_batch(task_names)


def _build_query_proj(in_dim: int, out_dim: int, layers: int, hidden_dim: int, bias: bool) -> nn.Module:
    """Single linear when layers<=1 (the original, byte-identical); otherwise a
    SiLU-separated MLP ending in a linear to out_dim."""
    if layers <= 1:
        return nn.Linear(in_dim, out_dim, bias=bias)
    h = hidden_dim if hidden_dim > 0 else in_dim
    mods: list[nn.Module] = []
    d = in_dim
    for _ in range(layers - 1):
        mods += [nn.Linear(d, h, bias=bias), nn.SiLU()]
        d = h
    mods.append(nn.Linear(d, out_dim, bias=bias))
    return nn.Sequential(*mods)


class QueryMLPLite(nn.Module):
    def __init__(self, input_dim: int, heads: int, k_dim: int, bias: bool = True, lang_dim: int = 0, fuse_method: str = "concat", proj_layers: int = 1, proj_hidden_dim: int = 0):
        super().__init__()
        self.input_dim = input_dim
        self.heads = heads
        self.k_dim = k_dim
        self.lang_dim = lang_dim
        self.fuse_method = fuse_method
        self.proj_layers = int(proj_layers)

        if fuse_method not in ("concat", "film"):
            raise ValueError(f"Unknown fuse_method: {fuse_method}. Expected 'concat' or 'film'.")

        if self.proj_layers > 1:
            print(f"Query projection MLP: depth={self.proj_layers}, hidden={proj_hidden_dim or input_dim}")

        if fuse_method == "concat":
            proj_input_dim = input_dim + lang_dim
            self.proj = _build_query_proj(proj_input_dim, heads * k_dim, self.proj_layers, proj_hidden_dim, bias)
        else:
            self.proj = _build_query_proj(input_dim, heads * k_dim, self.proj_layers, proj_hidden_dim, bias)
            if lang_dim > 0:
                hidden_dim = max(lang_dim, heads * k_dim) // 2
                self.film_mlp = nn.Sequential(
                    nn.Linear(lang_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 2 * heads * k_dim),
                )

        try:
            for p in self.proj.parameters():
                p.pk_query_proj_param = True
        except Exception:
            pass

        if fuse_method == "film" and lang_dim > 0:
            for p in self.film_mlp.parameters():
                try:
                    p.pk_query_proj_param = True
                except Exception:
                    pass

    def forward(self, x: torch.Tensor, lang_emb: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T, C)
        B_T = x.shape[0] if x.dim() == 2 else x.shape[0] * x.shape[1]
        x_flat = x.view(-1, self.input_dim)

        if self.fuse_method == "concat":
            if lang_emb is not None and self.lang_dim > 0:
                B = lang_emb.shape[0]
                T = B_T // B
                lang_emb_expanded = lang_emb.unsqueeze(1).expand(B, T, -1).reshape(B_T, -1)
                lang_emb_expanded = lang_emb_expanded.to(device=x_flat.device, dtype=x_flat.dtype)
                x_flat = torch.cat([x_flat, lang_emb_expanded], dim=-1)
            q = self.proj(x_flat)
        else:
            q = self.proj(x_flat)
            if lang_emb is not None and self.lang_dim > 0:
                B = lang_emb.shape[0]
                T = B_T // B
                lang_emb = lang_emb.to(device=q.device, dtype=q.dtype)
                film_params = self.film_mlp(lang_emb)
                gamma = film_params[:, : self.heads * self.k_dim]
                beta = film_params[:, self.heads * self.k_dim :]
                gamma = gamma.unsqueeze(1).expand(B, T, -1).reshape(B_T, -1)
                beta = beta.unsqueeze(1).expand(B, T, -1).reshape(B_T, -1)
                q = q * (1 + gamma) + beta

        return q.view(q.shape[0] * self.heads, self.k_dim)


class HashingMemoryLite(nn.Module):
    """
    Single-GPU, torch-only version of HashingMemory.

    Functionally mirrors the logic of the reference implementation (product keys,
    2-way PQ, kNN over subspaces, embedding_bag value aggregation), without Triton or DTensor.

    Supports two value types:
    - "vector": each slot is a value vector (original behavior, weighted sum of vectors)
    - "lora": each slot is a tiny LoRA (low-rank transform), output is weighted sum of LoRA outputs
    """

    EVAL_MEMORY = True

    def __init__(self, input_dim: int, output_dim: int, cfg: MemoryLayerConfig, lang_dim: int = 0,
                 lora_rank_override: int | None = None, value_noise_sigma_override: float | None = None):
        super().__init__()
        assert cfg.mem_k_dim % 2 == 0

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k_dim = cfg.mem_k_dim
        self.v_dim = cfg.mem_v_dim if cfg.mem_v_dim > 0 else output_dim
        self.heads = cfg.mem_heads
        self.knn = cfg.mem_knn
        self.n_keys = cfg.mem_n_keys
        self.size = self.n_keys ** 2
        self.log_usage = getattr(cfg, "log_usage", False)
        self.aggregate_usage = getattr(cfg, "aggregate_usage", False)
        self.lang_dim = lang_dim
        self.dropout_prob = getattr(cfg, "dropout_prob", 0.0)

        # Value type: "vector" (original) or "lora" (low-rank transform per slot)
        self.value_type = getattr(cfg, "value_type", "vector")
        self.lora_rank = int(lora_rank_override) if lora_rank_override is not None else getattr(cfg, "lora_rank", 2)

        # Value corruption parameters
        self.corruption_prob = getattr(cfg, "corruption_prob", 0.0)
        self.corruption_std = getattr(cfg, "corruption_std", 0.1)

        # Value-INPUT noise (E57): training-only perturbation of the x consumed by the
        # slot transforms. Per-layer sigma resolved at attach time (override); p and the
        # per-row amplitude range are global. sigma<=0 or p<=0 => mechanism fully off.
        self.value_noise_p = float(getattr(cfg, "value_input_noise_p", 0.0) or 0.0)
        self.value_noise_sigma = float(value_noise_sigma_override) if value_noise_sigma_override is not None else 0.0
        _amp = list(getattr(cfg, "value_input_noise_amp", [1.0, 1.0]) or [1.0, 1.0])
        self.value_noise_amp_lo = float(_amp[0])
        self.value_noise_amp_hi = float(_amp[1] if len(_amp) > 1 else _amp[0])

        # Router-only fast path (E49, warm-ups): substitute exact zeros for the
        # pre-projection value output, skipping the per-row slot gather. Bitwise-equal
        # to the real path while every slot_up is at zero init (lora slot output
        # up @ silu(down @ x) == 0 exactly); the projection/gate tail still runs so
        # the value_proj bias survives. Corruption would break the equivalence.
        self.router_only_fast = bool(getattr(cfg, "router_only_fast", False))
        if self.router_only_fast and self.corruption_prob > 0:
            raise ValueError("router_only_fast requires corruption_prob=0 (exactness)")

        # Query contrastive loss parameters
        self.contrastive_loss_weight = getattr(cfg, "contrastive_loss_weight", 0.0)
        self.contrastive_margin = getattr(cfg, "contrastive_margin", 0.0)
        self.contrastive_method = getattr(cfg, "contrastive_method", "centroid")
        self.contrastive_negatives_only = getattr(cfg, "contrastive_negatives_only", False)
        self.contrastive_query_queue = max(0, int(getattr(cfg, "contrastive_query_queue", 0)))
        self.routing_intra_task_locality_weight = float(
            getattr(cfg, "routing_intra_task_locality_weight", 0.0)
        )
        if self.routing_intra_task_locality_weight <= 0:
            self.routing_intra_task_locality_weight = float(getattr(cfg, "routing_compactness_weight", 0.0))
        self.routing_inter_task_separation_weight = float(
            getattr(cfg, "routing_inter_task_separation_weight", 0.0)
        )
        if self.routing_inter_task_separation_weight <= 0:
            self.routing_inter_task_separation_weight = float(getattr(cfg, "routing_separation_weight", 0.0))
        self.routing_global_balance_weight = getattr(cfg, "routing_global_balance_weight", 0.0)
        self.routing_loss_topk = max(0, int(getattr(cfg, "routing_loss_topk", 0)))
        self.routing_intra_task_min_support = max(0, int(getattr(cfg, "routing_intra_task_min_support", 0)))
        self.routing_intra_task_max_support = max(0, int(getattr(cfg, "routing_intra_task_max_support", 0)))

        # Optional cross-batch FIFO queue for sample-wise contrastive.
        # Kept as plain attrs (not buffers) to avoid dtype casting side effects.
        self._contrastive_queue_z = None
        self._contrastive_queue_labels = None
        self._contrastive_queue_ptr = 0
        self._contrastive_queue_count = 0
        # Stage current-step entries outside the live queue so checkpointed
        # recomputation does not observe a different contrastive key set.
        self._pending_contrastive_batches: list[tuple[torch.Tensor, torch.Tensor]] = []

        # Optional cross-batch FIFO queue for the routing separation loss.
        # Stores per-token queries (highest granularity) + per-token task labels.
        # Cap is in SAMPLES; ring buffer is sized lazily to cap * tokens_per_sample.
        # References are recomputed against current keys each step (no grad to the
        # stored queries), so this only feeds the separation term.
        self.routing_query_queue = max(0, int(getattr(cfg, "routing_query_queue", 0)))
        self._routing_queue_q = None        # (cap_rows, heads, k_dim) detached
        self._routing_queue_labels = None   # (cap_rows,) long
        self._routing_queue_ptr = 0
        self._routing_queue_count = 0
        self._routing_queue_cap_rows = 0
        self._pending_routing_batches: list[tuple[torch.Tensor, torch.Tensor, int]] = []

        # Store last contrastive loss and diagnostic metrics for aggregation (set during forward)
        self.last_contrastive_loss = None
        self.last_query_intra_sim = None  # mean cosine sim within same-task pairs
        self.last_query_inter_sim = None  # mean cosine sim across different-task pairs
        self.last_routing_intra_task_locality_loss = None
        self.last_routing_inter_task_similarity_loss = None
        self.last_routing_global_balance_loss = None
        self.last_routing_intra_task_entropy = None
        self.last_routing_intra_task_support = None
        self.last_routing_global_entropy = None
        self.last_routing_compactness_loss = None
        self.last_routing_separation_loss = None
        self.last_routing_task_entropy = None
        self.last_gate_mean = None  # mean sigmoid gate value
        self.last_per_task_unique_slots = None  # dict: task_id -> unique slot count
        self.last_per_task_entropy = None  # dict: task_id -> slot access entropy

        # Keys: (2 * heads * n_keys, k_dim // 2)
        # Keep dtype lightweight (bf16 if default is bf16), otherwise defaults to fp32.
        self.keys = nn.Parameter(torch.empty(2 * self.heads * self.n_keys, self.k_dim // 2))
        try:
            self.keys.pk_keys_param = True
        except Exception:
            pass

        # Value parameters depend on value_type
        if self.value_type == "vector":
            # Original: values (embedding table) kept in float32 for correct CUDA backward
            self.values = nn.Parameter(torch.empty(self.size, self.v_dim, dtype=torch.float32))
            self.values.pk_value_param = True
            self.values.fixed_lr = cfg.value_fixed_lr
        elif self.value_type == "lora":
            # LoRA-style: each slot has a low-rank transform (down @ SiLU @ up)
            # slot_down: (n_slots, input_dim, rank) - projects input to low-rank space
            # slot_up: (n_slots, rank, v_dim) - projects back to output space
            self.slot_down = nn.Parameter(
                torch.empty(self.size, self.input_dim, self.lora_rank, dtype=torch.float32)
            )
            self.slot_up = nn.Parameter(
                torch.empty(self.size, self.lora_rank, self.v_dim, dtype=torch.float32)
            )
            self.slot_down.pk_value_param = True
            self.slot_down.fixed_lr = cfg.value_fixed_lr
            self.slot_up.pk_value_param = True
            self.slot_up.fixed_lr = cfg.value_fixed_lr
            # Affine slots ("lora + value"): per-slot bias added to the LoRA output
            # before the retrieval-weighted sum. Zero-init => numerically identical
            # to plain lora until trained; absent from legacy checkpoints (loads
            # tolerate the missing key and keep the zero init).
            self.lora_slot_bias = bool(getattr(cfg, "lora_slot_bias", False))
            if self.lora_slot_bias:
                self.slot_bias = nn.Parameter(
                    torch.empty(self.size, self.v_dim, dtype=torch.float32)
                )
                self.slot_bias.pk_value_param = True
                self.slot_bias.fixed_lr = cfg.value_fixed_lr
        else:
            raise ValueError(f"Unknown value_type: {self.value_type}. Expected 'vector' or 'lora'.")

        # Optional projection/gating
        self.swilu_proj = cfg.swilu_projection
        self.v_proj = (cfg.mem_v_dim > 0) or self.swilu_proj
        if self.v_proj:
            proj_in = cfg.mem_v_dim if cfg.mem_v_dim > 0 else output_dim
            self.value_proj = nn.Linear(proj_in, output_dim)
        if self.swilu_proj:
            self.swilu_projection = nn.Linear(self.input_dim, proj_in)

        self.gating = nn.Linear(input_dim, 1) if cfg.mem_gated else None
        fuse_method = getattr(cfg, "fuse_method", "concat")
        self.query_proj = QueryMLPLite(
            self.input_dim, self.heads, self.k_dim, lang_dim=lang_dim, fuse_method=fuse_method,
            proj_layers=getattr(cfg, "query_proj_layers", 1),
            proj_hidden_dim=getattr(cfg, "query_proj_hidden_dim", 0),
        )

        self.reset_parameters()

        # Accumulator for per-slot selection counts across training steps (CPU tensor).
        # Not registered as a buffer to avoid dtype/device casting in forward.
        self.usage_counts = None
        # Accumulator for per-batch (binary) slot usage across training steps (CPU tensor).
        self.usage_batch_counts = None

        # Inference-time CPU offload of slot tensors. When True, slot_down/slot_up
        # (LoRA) or values (vector) are kept in pinned CPU memory and only the
        # retrieved slot rows are transferred to the compute device per forward.
        # Read from cfg here so the flag is set BEFORE any `.to(device)` chain runs
        # — `_apply` masks slot params from device moves when this is True, so they
        # never touch GPU memory in the first place (important for memory-constrained
        # inference where the full slot tensor wouldn't fit). The forward inspects
        # the slot tensor's `.device` at runtime so the on-GPU path is untouched
        # when offload is disabled.
        self._slots_offloaded = bool(getattr(cfg, "offload_slots_to_cpu", False))
        if self._slots_offloaded:
            # Best-effort pin; falls back to plain CPU memory if pinning fails
            # (e.g. host has no CUDA driver or already-pinned tensor).
            for p in self._slot_params():
                try:
                    p.data = p.data.pin_memory()
                except (RuntimeError, NotImplementedError):
                    pass

    def _slot_param_names(self) -> tuple[str, ...]:
        if self.value_type == "lora":
            if getattr(self, "lora_slot_bias", False):
                return ("slot_down", "slot_up", "slot_bias")
            return ("slot_down", "slot_up")
        if self.value_type == "vector":
            return ("values",)
        return ()

    def _slot_params(self):
        for name in self._slot_param_names():
            p = self._parameters.get(name)
            if p is not None:
                yield p

    def _apply(self, fn, recurse=True):
        """Override to keep slot params on CPU when offload is enabled.

        ``nn.Module._apply`` is the funnel for every ``.to(device)`` / ``.cuda()`` /
        ``.float()`` call. Temporarily evict slot parameters from ``self._parameters``
        so ``fn`` (which would otherwise move them to ``device``) skips them.
        """
        if not self._slots_offloaded:
            return super()._apply(fn, recurse=recurse)
        saved = {}
        for name in self._slot_param_names():
            if name in self._parameters:
                saved[name] = self._parameters.pop(name)
        try:
            result = super()._apply(fn, recurse=recurse)
        finally:
            for name, param in saved.items():
                self._parameters[name] = param
        return result

    def _ensure_contrastive_queue(self, device: torch.device, dtype: torch.dtype) -> None:
        cap = self.contrastive_query_queue
        if cap <= 0:
            return
        queue_shape = (cap, self.k_dim)
        need_init = (
            self._contrastive_queue_z is None
            or self._contrastive_queue_labels is None
            or self._contrastive_queue_z.shape != queue_shape
            or self._contrastive_queue_z.device != device
            or self._contrastive_queue_z.dtype != dtype
            or self._contrastive_queue_labels.shape[0] != cap
            or self._contrastive_queue_labels.device != device
        )
        if need_init:
            self._contrastive_queue_z = torch.empty(queue_shape, device=device, dtype=dtype)
            self._contrastive_queue_labels = torch.empty((cap,), device=device, dtype=torch.long)
            self._contrastive_queue_ptr = 0
            self._contrastive_queue_count = 0

    def _get_contrastive_queue(
        self, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        cap = self.contrastive_query_queue
        if cap <= 0:
            return None, None
        self._ensure_contrastive_queue(device, dtype)
        count = int(self._contrastive_queue_count)
        if count <= 0:
            return None, None
        return self._contrastive_queue_z[:count], self._contrastive_queue_labels[:count]

    @torch.no_grad()
    def _enqueue_contrastive_queries(self, z: torch.Tensor, labels: torch.Tensor) -> None:
        cap = self.contrastive_query_queue
        if cap <= 0 or z.numel() == 0:
            return
        self._ensure_contrastive_queue(z.device, z.dtype)
        z_detached = z.detach()
        labels_detached = labels.detach().to(device=z.device, dtype=torch.long)
        n = int(z_detached.shape[0])
        if n <= 0:
            return

        if n >= cap:
            self._contrastive_queue_z.copy_(z_detached[-cap:])
            self._contrastive_queue_labels.copy_(labels_detached[-cap:])
            self._contrastive_queue_ptr = 0
            self._contrastive_queue_count = cap
            return

        ptr = int(self._contrastive_queue_ptr)
        end = ptr + n
        if end <= cap:
            self._contrastive_queue_z[ptr:end] = z_detached
            self._contrastive_queue_labels[ptr:end] = labels_detached
        else:
            first = cap - ptr
            self._contrastive_queue_z[ptr:] = z_detached[:first]
            self._contrastive_queue_labels[ptr:] = labels_detached[:first]
            rem = end - cap
            self._contrastive_queue_z[:rem] = z_detached[first:]
            self._contrastive_queue_labels[:rem] = labels_detached[first:]

        self._contrastive_queue_ptr = end % cap
        self._contrastive_queue_count = min(cap, int(self._contrastive_queue_count) + n)

    @torch.no_grad()
    def _stage_contrastive_queries(self, z: torch.Tensor, labels: torch.Tensor) -> None:
        cap = self.contrastive_query_queue
        if cap <= 0 or z.numel() == 0 or _is_checkpoint_recompute():
            return
        z_detached = z.detach()
        labels_detached = labels.detach().to(device=z.device, dtype=torch.long)
        self._pending_contrastive_batches.append((z_detached, labels_detached))

    @torch.no_grad()
    def flush_staged_contrastive_queries(self) -> None:
        # Contrastive and routing queues are flushed independently (either may be
        # active without the other).
        if self._pending_contrastive_batches:
            for z, labels in self._pending_contrastive_batches:
                self._enqueue_contrastive_queries(z=z, labels=labels)
            self._pending_contrastive_batches.clear()
        # Routing-separation queue shares the same flush hook.
        if self._pending_routing_batches:
            for q_rows, labels, rows_per_sample in self._pending_routing_batches:
                self._enqueue_routing_queries(q_rows, labels, rows_per_sample)
            self._pending_routing_batches.clear()

    # ── Routing-separation cross-batch queue ───────────────────────────
    def _ensure_routing_queue(self, device: torch.device, dtype: torch.dtype, rows_per_sample: int) -> None:
        cap = self.routing_query_queue
        if cap <= 0:
            return
        cap_rows = cap * max(1, int(rows_per_sample))
        need_init = (
            self._routing_queue_q is None
            or self._routing_queue_q.shape != (cap_rows, self.heads, self.k_dim)
            or self._routing_queue_q.device != device
            or self._routing_queue_q.dtype != dtype
            or self._routing_queue_labels.device != device
        )
        if need_init:
            self._routing_queue_q = torch.empty((cap_rows, self.heads, self.k_dim), device=device, dtype=dtype)
            self._routing_queue_labels = torch.empty((cap_rows,), device=device, dtype=torch.long)
            self._routing_queue_ptr = 0
            self._routing_queue_count = 0
            self._routing_queue_cap_rows = cap_rows

    def _get_routing_queue(self, device: torch.device, dtype: torch.dtype):
        if self.routing_query_queue <= 0:
            return None, None
        if self._routing_queue_q is None or int(self._routing_queue_count) <= 0:
            return None, None
        n = int(self._routing_queue_count)
        return (
            self._routing_queue_q[:n].to(device=device, dtype=dtype),
            self._routing_queue_labels[:n].to(device=device),
        )

    @torch.no_grad()
    def _stage_routing_queries(self, query: torch.Tensor, B: int, T: int, task_ids: torch.Tensor,
                               size_T: int | None = None) -> None:
        if self.routing_query_queue <= 0 or _is_checkpoint_recompute():
            return
        # query: (B*T*heads, k_dim) in (bs, heads) row-major order with bs = B*T.
        # size_T pins the ring-buffer sizing when T varies batch-to-batch (token-mask slice).
        q_rows = query.detach().view(B * T, self.heads, self.k_dim)
        labels = task_ids.view(B, 1).expand(B, T).reshape(B * T).detach().to(torch.long)
        self._pending_routing_batches.append((q_rows, labels, int(size_T or T)))

    def _router_trainable(self) -> bool:
        """True iff the router (query proj or keys) can receive gradient.

        The routing/contrastive losses and the cross-batch routing queue only
        affect the router, so when it is frozen (sequential adaptation, where only
        the memory values train) they are inert and skipped. Cheap: a few bool
        checks per forward. NOTE: if router training is ever re-enabled during
        sequential (train_query_proj / train_memory_keys), the sequential loop would
        also need the staged-query flush hook that currently lives only in pretrain.
        """
        if self.keys.requires_grad:
            return True
        return any(p.requires_grad for p in self.query_proj.parameters())

    @torch.no_grad()
    def _enqueue_routing_queries(self, q_rows: torch.Tensor, labels: torch.Tensor, rows_per_sample: int) -> None:
        cap = self.routing_query_queue
        if cap <= 0 or q_rows.numel() == 0:
            return
        self._ensure_routing_queue(q_rows.device, q_rows.dtype, rows_per_sample)
        cap_rows = self._routing_queue_cap_rows
        n = int(q_rows.shape[0])
        if n >= cap_rows:
            self._routing_queue_q.copy_(q_rows[-cap_rows:])
            self._routing_queue_labels.copy_(labels[-cap_rows:])
            self._routing_queue_ptr = 0
            self._routing_queue_count = cap_rows
            return
        ptr = int(self._routing_queue_ptr)
        end = ptr + n
        if end <= cap_rows:
            self._routing_queue_q[ptr:end] = q_rows
            self._routing_queue_labels[ptr:end] = labels
        else:
            first = cap_rows - ptr
            self._routing_queue_q[ptr:] = q_rows[:first]
            self._routing_queue_labels[ptr:] = labels[:first]
            rem = end - cap_rows
            self._routing_queue_q[:rem] = q_rows[first:]
            self._routing_queue_labels[:rem] = labels[first:]
        self._routing_queue_ptr = end % cap_rows
        self._routing_queue_count = min(cap_rows, int(self._routing_queue_count) + n)

    @torch.no_grad()
    def offload_slots_to_cpu(self) -> int:
        """Move slot tensors (slot_down/slot_up or values) to pinned CPU memory.

        Intended for inference on memory-constrained GPUs. The forward path
        gathers only the retrieved slot rows and transfers that subset to
        the compute device per call, so the numerical output is identical
        to the on-GPU path.

        Returns the number of bytes freed from GPU memory by the move.
        Safe to call multiple times — re-pinning a CPU tensor is a no-op.
        """
        freed_bytes = 0
        slot_tensors = []
        if self.value_type == "lora":
            slot_tensors = [self.slot_down, self.slot_up]
        elif self.value_type == "vector":
            slot_tensors = [self.values]

        for p in slot_tensors:
            if p.device.type != "cpu":
                freed_bytes += p.numel() * p.element_size()
                cpu_data = p.data.detach().to("cpu")
                try:
                    cpu_data = cpu_data.pin_memory()
                except (RuntimeError, NotImplementedError):
                    # Pinning is best-effort: fails on some hosts (e.g. no CUDA driver,
                    # or already-pinned tensor). Plain CPU memory still works, just slower.
                    pass
                p.data = cpu_data

        self._slots_offloaded = True
        return freed_bytes

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.k_dim)
        nn.init.uniform_(self.keys, a=-bound, b=bound)

        if self.value_type == "vector":
            nn.init.normal_(self.values, mean=0, std=self.v_dim ** -0.5)
        elif self.value_type == "lora":
            # Initialize LoRA params similar to standard LoRA practice
            # down projection: small random init
            nn.init.normal_(self.slot_down, mean=0, std=0.02)
            # up projection: zero init so LoRA starts as identity-ish
            nn.init.zeros_(self.slot_up)
            # affine bias: zero init so flag-on == flag-off until trained
            if getattr(self, "lora_slot_bias", False):
                nn.init.zeros_(self.slot_bias)

        proj_linears = (
            [self.query_proj.proj] if isinstance(self.query_proj.proj, nn.Linear)
            else [m for m in self.query_proj.proj if isinstance(m, nn.Linear)]
        )
        for lin in proj_linears:
            nn.init.xavier_uniform_(lin.weight)
        if self.v_proj:
            nn.init.normal_(self.value_proj.weight, mean=0, std=self.output_dim ** -0.5)
        if self.swilu_proj:
            nn.init.normal_(self.swilu_projection.weight, mean=0, std=self.output_dim ** -0.5)
        if self.gating is not None:
            nn.init.normal_(self.gating.weight, mean=0, std=self.input_dim ** -0.5)

    def forward(self, x: torch.Tensor, lang_emb: torch.Tensor | None = None, task_ids: torch.Tensor | None = None, router_x: torch.Tensor | None = None, token_mask: torch.Tensor | None = None, stat_repeat: torch.Tensor | None = None, return_retrieval: bool = False) -> torch.Tensor:
        # x: (B, T, C)
        # lang_emb: (B, lang_dim) or None
        # task_ids: (B,) optional task indices for contrastive loss
        # router_x: (B, T, C) optional frozen-base routing features (memory-free
        #   backbone forward). When given, the QUERY and the GATE read router_x so
        #   the addressing is stationary under value training; the value/output
        #   path (LoRA transform, swilu) stays on the live stream x.
        # Ensure module parameters/buffers match the input dtype/device without recreating Parameters
        dtype, device = x.dtype, x.device
        if getattr(self, "_param_dtype", None) is not dtype or getattr(self, "_param_device", None) is not device:
            for p in self.parameters(recurse=True):
                is_value = getattr(p, "pk_value_param", False)
                # Skip slot/value params when offloaded — those live in CPU pinned memory
                # and are gathered per-forward via the _slots_offloaded branches below.
                if is_value and self._slots_offloaded:
                    continue
                # Keep value params (vectors or LoRA weights) in float32 for stable gradients
                if is_value:
                    p.data = p.data.to(device=device, dtype=torch.float32)
                else:
                    p.data = p.data.to(device=device, dtype=dtype)
            for b in self.buffers(recurse=True):
                b.data = b.data.to(device=device, dtype=dtype)
            self._param_dtype = dtype
            self._param_device = device

        B, T, C = x.shape
        x_flat = x.view(-1, C)
        bs = x_flat.shape[0]

        if router_x is not None:
            if router_x.shape != x.shape:
                raise ValueError(f"router_x shape {tuple(router_x.shape)} != x shape {tuple(x.shape)}")
            xr = router_x.to(dtype=x.dtype)
            xr_flat = xr.view(-1, C)
        else:
            xr = x
            xr_flat = x_flat

        # stat_repeat (E45 route-once): per-position multiplicity for usage statistics.
        # A compact routed row that SERVES n live positions (the shared state palette)
        # counts n times in TF/usage/audit stats — preserving the write-demand and audit
        # semantics of the redundant per-position routing it replaces — while losses and
        # queues see each unique routing row once (the dedup).
        rep = None
        if stat_repeat is not None:
            rep = stat_repeat.reshape(-1).to(device=device, dtype=torch.long)

        # Query (with optional language conditioning)
        query = self.query_proj(xr, lang_emb=lang_emb)  # (bs*heads, k_dim)

        # Compute query contrastive loss if enabled and task_ids provided
        self.last_contrastive_loss = None
        self.last_query_intra_sim = None
        self.last_query_inter_sim = None
        self.last_routing_intra_task_locality_loss = None
        self.last_routing_inter_task_similarity_loss = None
        self.last_routing_global_balance_loss = None
        self.last_routing_intra_task_entropy = None
        self.last_routing_intra_task_support = None
        self.last_routing_global_entropy = None
        self.last_routing_compactness_loss = None
        self.last_routing_separation_loss = None
        self.last_routing_task_entropy = None
        # The routing/contrastive losses and the cross-batch routing queue only do
        # useful work when the router (query proj + keys) can receive gradient.
        # During sequential adaptation the router is FROZEN (only memory values
        # train), so they are inert and are skipped. Skipping the queue staging is
        # also what prevents an unbounded GPU leak: the staged-query flush hook lives
        # only in the pretrain loop (lerobot_train.py), so in the sequential loop the
        # pending list would otherwise grow every step and OOM (~1k steps).
        router_trainable = self._router_trainable()
        # Token-mask (E44 pad fix): valid tokens are a CONTIGUOUS PREFIX of the span (the
        # tokenized language field pads at the tail), so the loss/queue machinery runs on a
        # rectangular [:, :Tv] slice (Tv = min valid count in the batch) — pads never reach
        # the contrastive means, routing histograms, or the cross-batch queues.
        q_loss, T_loss = query, T
        if token_mask is not None and T > 1:
            tv = max(int(token_mask.sum(dim=1).min().item()), 1)
            if tv < T:
                q_loss = query.view(B, T, self.heads, self.k_dim)[:, :tv].reshape(B * tv * self.heads, self.k_dim)
                T_loss = tv
        if self.training and self.contrastive_loss_weight > 0 and task_ids is not None and router_trainable:
            if self.contrastive_method == "sample":
                self.last_contrastive_loss = self._compute_sample_contrastive_loss(q_loss, B, T_loss, task_ids)
            else:
                self.last_contrastive_loss = self._compute_contrastive_loss(q_loss, B, T_loss, task_ids)

        s1_full, s2_full = self._compute_subkey_scores(query)
        s1_loss, s2_loss = s1_full, s2_full
        if T_loss != T:
            def _slice_scores(sf):
                # subkey scores are (bs=B*T, heads, n_keys); keep that layout after slicing
                return sf.view(B, T, self.heads, -1)[:, :T_loss].reshape(B * T_loss, self.heads, -1)
            s1_loss, s2_loss = _slice_scores(s1_full), _slice_scores(s2_full)
        if self.training and task_ids is not None and router_trainable and (
            self.routing_intra_task_locality_weight > 0
            or self.routing_inter_task_separation_weight > 0
            or self.routing_global_balance_weight > 0
        ):
            (
                self.last_routing_intra_task_locality_loss,
                self.last_routing_inter_task_similarity_loss,
                self.last_routing_global_balance_loss,
                self.last_routing_intra_task_entropy,
                self.last_routing_intra_task_support,
                self.last_routing_global_entropy,
            ) = self._compute_routing_losses(s1_loss, s2_loss, B, T_loss, task_ids)
            self.last_routing_compactness_loss = self.last_routing_intra_task_locality_loss
            self.last_routing_separation_loss = self.last_routing_inter_task_similarity_loss
            self.last_routing_task_entropy = self.last_routing_intra_task_entropy

        # Stage per-token routing queries for the cross-batch separation queue
        # (flushed after the optimizer step in the pretrain loop; skipped when the
        # router is frozen so the pending list cannot leak — see note above).
        if self.training and self.routing_query_queue > 0 and task_ids is not None and router_trainable:
            self._stage_routing_queries(q_loss, B, T_loss, task_ids, size_T=T)

        # Indices and scores
        scores, indices = self._get_indices_from_subkey_scores(s1_full, s2_full)  # (bs*heads, knn)

        # Record selected indices/scores for analysis during eval
        if not self.training and self.EVAL_MEMORY:
            sel_i = indices.view(bs, self.heads, self.knn)
            sel_s = scores.view(bs, self.heads, self.knn)
            if token_mask is not None:
                keep_e = token_mask.reshape(-1).bool()
                sel_i, sel_s = sel_i[keep_e], sel_s[keep_e]
                if rep is not None:
                    r = rep[keep_e]
                    sel_i = sel_i.repeat_interleave(r, dim=0)
                    sel_s = sel_s.repeat_interleave(r, dim=0)
            elif rep is not None:
                sel_i = sel_i.repeat_interleave(rep, dim=0)
                sel_s = sel_s.repeat_interleave(rep, dim=0)
            self.last_indices = sel_i.detach().cpu()
            self.last_scores = sel_s.detach().cpu().float()

        # Softmax in float32 for numerical stability; we will cast as needed later
        weights = F.softmax(scores.float(), dim=-1)
        # Merge heads
        indices = indices.view(bs, self.heads * self.knn)
        weights = weights.view(bs, self.heads * self.knn)

        # Record selected indices/weights during training when log_usage is enabled
        if self.training and self.log_usage:
            sel_i = indices.view(bs, self.heads, self.knn)
            sel_w = weights.view(bs, self.heads, self.knn)
            if token_mask is not None:
                keep = token_mask.reshape(-1).bool()
                sel_i, sel_w = sel_i[keep], sel_w[keep]
                if rep is not None:
                    r = rep[keep]
                    sel_i = sel_i.repeat_interleave(r, dim=0)
                    sel_w = sel_w.repeat_interleave(r, dim=0)
            elif rep is not None:
                sel_i = sel_i.repeat_interleave(rep, dim=0)
                sel_w = sel_w.repeat_interleave(rep, dim=0)
            self.last_indices = sel_i.detach()
            self.last_weights = sel_w.detach()

            # Per-task slot diagnostics (unique count and entropy)
            if task_ids is not None:
                with torch.no_grad():
                    self.last_per_task_unique_slots = {}
                    self.last_per_task_entropy = {}
                    idx_by_sample = indices.view(B, T, self.heads * self.knn)
                    for t in torch.unique(task_ids).tolist():
                        mask = (task_ids == int(t))
                        if not mask.any():
                            continue
                        if token_mask is not None:
                            rows = idx_by_sample[mask][token_mask[mask].bool()]
                            if rep is not None:
                                rr = rep.view(B, T)[mask][token_mask[mask].bool()]
                                rows = rows.repeat_interleave(rr, dim=0)
                            t_idx = rows.reshape(-1).to(torch.long)
                        elif rep is not None:
                            rows = idx_by_sample[mask].reshape(-1, self.heads * self.knn)
                            rr = rep.view(B, T)[mask].reshape(-1)
                            t_idx = rows.repeat_interleave(rr, dim=0).reshape(-1).to(torch.long)
                        else:
                            t_idx = idx_by_sample[mask].reshape(-1).to(torch.long)
                        self.last_per_task_unique_slots[int(t)] = int(torch.unique(t_idx).numel())
                        counts = torch.bincount(t_idx, minlength=self.size).float()
                        total = counts.sum()
                        if total > 0:
                            p = counts[counts > 0] / total
                            H = -(p * p.log()).sum()
                            self.last_per_task_entropy[int(t)] = float(H.item())
            else:
                self.last_per_task_unique_slots = None
                self.last_per_task_entropy = None

        # Apply dropout to retrieved slots during training (per-head normalization)
        if self.training and self.dropout_prob > 0:
            weights_per_head = weights.view(bs, self.heads, self.knn)
            keep_mask = torch.bernoulli(
                torch.full_like(weights_per_head, 1.0 - self.dropout_prob)
            )
            weights_per_head = weights_per_head * keep_mask
            weight_sums = weights_per_head.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            weights_per_head = weights_per_head / weight_sums
            weights = weights_per_head.view(bs, self.heads * self.knn)

        # Accumulate per-slot usage counts across training
        if self.training and self.aggregate_usage:
            with torch.no_grad():
                if rep is not None:
                    flat_idx = indices.repeat_interleave(rep, dim=0).reshape(-1).to(torch.long).detach().cpu()
                else:
                    flat_idx = indices.reshape(-1).to(torch.long).detach().cpu()
                # Lazily initialize accumulator on first use
                if self.usage_counts is None or self.usage_counts.numel() != self.size:
                    self.usage_counts = torch.zeros(self.size, dtype=torch.long)
                batch_counts = torch.bincount(flat_idx, minlength=self.size)
                # In-place accumulate
                self.usage_counts[: batch_counts.shape[0]] += batch_counts
                # Accumulate per-batch (binary) usage: increment by 1 if slot appears in this batch
                if self.usage_batch_counts is None or self.usage_batch_counts.numel() != self.size:
                    self.usage_batch_counts = torch.zeros(self.size, dtype=torch.long)
                batch_present = (batch_counts > 0).to(torch.long)
                self.usage_batch_counts[: batch_present.shape[0]] += batch_present

        # Weighted aggregation with optional value corruption
        # embedding_bag backward with per_sample_weights is not implemented for bf16 on CUDA.
        # Perform the op in float32 and cast back to the model dtype afterwards.
        if self.router_only_fast:
            # E49 warm-up fast path: values pinned at zero init => the real value path
            # returns exact zeros; skip the gather and substitute them directly. The
            # projection/gate tail below is shared, so output == real path bitwise.
            out_fp32 = x_flat.new_zeros(bs, self.v_dim, dtype=torch.float32)
        elif self.value_type == "vector":
            out_fp32 = self._forward_vector_values(x_flat, indices, weights, device)
        elif self.value_type == "lora":
            out_fp32 = self._forward_lora_values(x_flat, indices, weights, device)
        else:
            raise ValueError(f"Unknown value_type: {self.value_type}")
        out = out_fp32.to(dtype)

        if self.v_proj and not self.swilu_proj:
            out = self.value_proj(out)
        if self.swilu_proj:
            out = self.value_proj(out * F.silu(self.swilu_projection(x_flat)))

        out = out.view(B, T, -1)
        if self.gating is not None:
            gate = torch.sigmoid(self.gating(xr_flat)).view(B, T, 1)
            if self.training and self.log_usage:
                self.last_gate_mean = float(gate.mean().item())
            out = gate * out
        if return_retrieval:
            return (
                out,
                indices.view(B, T, self.heads * self.knn),
                weights.view(B, T, self.heads * self.knn),
            )
        return out

    def apply_shared_palette(
        self,
        x_pos: torch.Tensor,
        pos_mask: torch.Tensor,
        idx_row: torch.Tensor,
        w_row: torch.Tensor,
        router_key: torch.Tensor,
    ) -> torch.Tensor:
        """Apply ONE retrieved slot mixture per sample to MANY live positions (E45
        route-once). The slot parameters are gathered once per row — never expanded
        per position — which removes the redundant per-position gather that made the
        broadcast-key path expensive.

        x_pos: (B, N, C) live value inputs (padded); pos_mask: (B, N) bool;
        idx_row/w_row: (B, heads*knn) the row's retrieval (post-softmax/dropout
        weights from the compact routed call); router_key: (B, C) the shared key
        (feeds the gate, mirroring the broadcast path where the gate read the key).
        Returns (B, N, v_dim) with masked positions zeroed.
        """
        B, N, C = x_pos.shape
        x32 = x_pos.float()
        if self.training and self.value_noise_p > 0 and self.value_noise_sigma > 0:
            # E57: noise the down-projection input on VALID positions only (pad rows
            # would deflate the self-calibrating per-dim std); the swilu tail below
            # keeps the clean x_flat.
            flat = x32.reshape(B * N, C).clone()
            valid = pos_mask.reshape(B * N).bool()
            flat[valid] = self._value_input_noise(flat[valid])
            x32 = flat.view(B, N, C)
        if self.value_type == "vector":
            vals = self.values.float()[idx_row]  # (B, K, V)
            row_out = (vals * w_row.float().unsqueeze(-1)).sum(dim=1)  # (B, V)
            out32 = row_out.unsqueeze(1).expand(B, N, -1)
        elif self.value_type == "lora":
            down = self.slot_down[idx_row]  # (B, K, C, r) fp32 value params
            up = self.slot_up[idx_row]  # (B, K, r, V)
            hidden = F.silu(torch.einsum("bnc,bkcr->bnkr", x32, down))
            slot_outputs = torch.einsum("bnkr,bkrv->bnkv", hidden, up)
            if getattr(self, "lora_slot_bias", False):
                slot_outputs = slot_outputs + self.slot_bias[idx_row].unsqueeze(1)
            if self.training and self.corruption_prob > 0:
                corruption_mask = torch.bernoulli(
                    torch.full((B, idx_row.shape[1]), self.corruption_prob, device=x_pos.device)
                ).bool()
                noise = torch.randn_like(slot_outputs) * self.corruption_std
                slot_outputs = slot_outputs + corruption_mask[:, None, :, None].float() * noise
            out32 = (slot_outputs * w_row.float()[:, None, :, None]).sum(dim=2)  # (B, N, V)
        else:
            raise ValueError(f"Unknown value_type: {self.value_type}")
        out = out32.to(x_pos.dtype)
        x_flat = x_pos.reshape(B * N, C)
        out = out.reshape(B * N, -1)
        if self.v_proj and not self.swilu_proj:
            out = self.value_proj(out)
        if self.swilu_proj:
            out = self.value_proj(out * F.silu(self.swilu_projection(x_flat)))
        out = out.view(B, N, -1)
        if self.gating is not None:
            gate = torch.sigmoid(self.gating(router_key.to(x_pos.dtype)))  # (B, 1)
            out = gate.view(B, 1, 1) * out
        return out * pos_mask.to(dtype=out.dtype).unsqueeze(-1)

    def _forward_vector_values(
        self, x_flat: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        Original vector-based forward: weighted sum of value vectors.

        Args:
            x_flat: Flattened input (bs, C) - used only for corruption noise shape
            indices: Selected slot indices (bs, heads*knn)
            weights: Softmax weights (bs, heads*knn)
            device: Target device

        Returns:
            Aggregated output (bs, v_dim) in float32
        """
        bs = indices.shape[0]
        # When slots are offloaded to CPU, gather the unique retrieved rows on CPU
        # and transfer just the subset to the compute device. Falls back to the
        # original on-GPU path otherwise.
        if self._slots_offloaded and self.values.device.type == "cpu":
            indices_flat = indices.reshape(-1)
            unique_idx_cpu, inverse = torch.unique(indices_flat.cpu(), return_inverse=True)
            values_subset = self.values[unique_idx_cpu].to(device, non_blocking=True).float()
            inverse = inverse.to(device, non_blocking=True).view(bs, self.heads * self.knn)
            retrieved_values = values_subset[inverse]
            if self.training and self.corruption_prob > 0:
                corruption_mask = torch.bernoulli(
                    torch.full((bs, self.heads * self.knn), self.corruption_prob, device=device)
                ).bool()
                noise = torch.randn_like(retrieved_values) * self.corruption_std
                retrieved_values = retrieved_values + corruption_mask.unsqueeze(-1).float() * noise
            out_fp32 = (retrieved_values * weights.float().unsqueeze(-1)).sum(dim=1)
        elif self.training and self.corruption_prob > 0:
            # Manual gather + corruption + weighted sum
            retrieved_values = self.values.float()[indices]  # (bs, heads*knn, v_dim)

            corruption_mask = torch.bernoulli(
                torch.full((bs, self.heads * self.knn), self.corruption_prob, device=device)
            ).bool()
            noise = torch.randn_like(retrieved_values) * self.corruption_std
            retrieved_values = retrieved_values + corruption_mask.unsqueeze(-1).float() * noise

            out_fp32 = (retrieved_values * weights.float().unsqueeze(-1)).sum(dim=1)
        else:
            out_fp32 = F.embedding_bag(
                indices,
                self.values.float(),
                per_sample_weights=weights.float(),
                mode="sum",
            )
        return out_fp32

    def _value_input_noise(self, x32: torch.Tensor) -> torch.Tensor:
        """E57 value-input noise: perturb the fp32 x consumed by the slot transforms.

        Training-only, value-path-only (caller applies this to the down-projection input
        exclusively — router/gate/swilu/MLP all keep the clean x). Per-dim scale is the
        current batch's per-dim std over rows (self-calibrating, detached); sigma is the
        per-layer calibrated ratio; a per-row amplitude draw covers the excursion band.
        x32: (rows, input_dim) fp32. Returns a perturbed copy (or x32 unchanged if off).
        """
        if not self.training or self.value_noise_p <= 0 or self.value_noise_sigma <= 0:
            return x32
        with torch.no_grad():
            std = x32.std(dim=0, keepdim=True) + 1e-6            # (1, D)
            mask = torch.bernoulli(torch.full_like(x32, self.value_noise_p))
            noise = torch.randn_like(x32) * std * self.value_noise_sigma
            if self.value_noise_amp_lo != self.value_noise_amp_hi:
                amp = torch.empty(x32.shape[0], 1, device=x32.device, dtype=x32.dtype).uniform_(
                    self.value_noise_amp_lo, self.value_noise_amp_hi)
                noise = noise * amp
            elif self.value_noise_amp_lo != 1.0:
                noise = noise * self.value_noise_amp_lo
            delta = mask * noise
        return x32 + delta

    def _forward_lora_values(
        self, x_flat: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        LoRA-based forward: run input through selected tiny LoRAs, weighted sum of outputs.

        Each slot is a low-rank transform: output_i = slot_up_i @ SiLU(slot_down_i @ x)
        (+ per-slot bias b_i when cfg.lora_slot_bias — the affine "lora + value" form).
        Final output = sum(weights_i * output_i)

        Args:
            x_flat: Flattened input (bs, input_dim)
            indices: Selected slot indices (bs, heads*knn)
            weights: Softmax weights (bs, heads*knn)
            device: Target device

        Returns:
            Aggregated output (bs, v_dim) in float32
        """
        bs = indices.shape[0]
        k = self.heads * self.knn

        # x_flat: (bs, input_dim), indices: (bs, k)
        # E57: the down-projection consumes the (optionally noised) x; everything
        # downstream that reads x_flat directly (swilu tail) stays clean.
        x_fp32 = self._value_input_noise(x_flat.float())

        # Gather LoRA weights for selected slots
        # slot_down: (n_slots, input_dim, rank) -> (bs, k, input_dim, rank)
        # slot_up: (n_slots, rank, v_dim) -> (bs, k, rank, v_dim)
        bias_weights = None
        if self._slots_offloaded and self.slot_down.device.type == "cpu":
            # CPU-resident slots: dedupe indices, gather on CPU, transfer subset to GPU.
            # `indices` lives on the compute device; flatten + unique to bound the transfer.
            indices_flat = indices.reshape(-1)
            unique_idx_cpu, inverse = torch.unique(indices_flat.cpu(), return_inverse=True)
            down_subset = self.slot_down[unique_idx_cpu].to(device, non_blocking=True)
            up_subset = self.slot_up[unique_idx_cpu].to(device, non_blocking=True)
            inverse = inverse.to(device, non_blocking=True).view(bs, k)
            down_weights = down_subset[inverse]  # (bs, k, input_dim, rank)
            up_weights = up_subset[inverse]  # (bs, k, rank, v_dim)
            if getattr(self, "lora_slot_bias", False):
                bias_subset = self.slot_bias[unique_idx_cpu].to(device, non_blocking=True)
                bias_weights = bias_subset[inverse]  # (bs, k, v_dim)
        else:
            down_weights = self.slot_down[indices]  # (bs, k, input_dim, rank)
            up_weights = self.slot_up[indices]  # (bs, k, rank, v_dim)
            if getattr(self, "lora_slot_bias", False):
                bias_weights = self.slot_bias[indices]  # (bs, k, v_dim)

        # Compute each slot's LoRA output:
        # hidden = SiLU(x @ down) -> (bs, k, rank)
        # output = hidden @ up -> (bs, k, v_dim)

        # Expand x for broadcasting: (bs, 1, input_dim)
        x_expanded = x_fp32.unsqueeze(1)

        # down projection: (bs, 1, input_dim) @ (bs, k, input_dim, rank) -> (bs, k, rank)
        hidden = torch.einsum('bni,bkir->bkr', x_expanded, down_weights)
        hidden = F.silu(hidden)

        # up projection: (bs, k, rank) @ (bs, k, rank, v_dim) -> (bs, k, v_dim)
        slot_outputs = torch.einsum('bkr,bkro->bko', hidden, up_weights)

        # Affine slots: per-slot bias joins the slot output BEFORE corruption and
        # the weighted sum, so it is part of the adapter output proper.
        if bias_weights is not None:
            slot_outputs = slot_outputs + bias_weights

        # Corrupt each adapter output before the shared gating path.
        if self.training and self.corruption_prob > 0:
            corruption_mask = torch.bernoulli(
                torch.full((bs, k), self.corruption_prob, device=device)
            ).bool()
            noise = torch.randn_like(slot_outputs) * self.corruption_std
            slot_outputs = slot_outputs + corruption_mask.unsqueeze(-1).float() * noise

        # Weighted sum: (bs, k, v_dim) * (bs, k, 1) -> sum -> (bs, v_dim)
        out_fp32 = (slot_outputs * weights.float().unsqueeze(-1)).sum(dim=1)

        return out_fp32

    def _compute_contrastive_loss(self, query: torch.Tensor, B: int, T: int, task_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss to push query centroids apart across tasks.

        Args:
            query: Query vectors of shape (B*T*heads, k_dim)
            B: Batch size
            T: Sequence length
            task_ids: Task indices of shape (B,)

        Returns:
            Scalar contrastive loss tensor
        """
        # Reshape query to (B, T*heads, k_dim) and compute per-sample mean
        # query: (B*T*heads, k_dim) -> (B, T*heads, k_dim) -> (B, k_dim)
        query_reshaped = query.view(B, T * self.heads, self.k_dim)
        per_sample_query = query_reshaped.mean(dim=1)  # (B, k_dim)

        # Get unique tasks in this batch
        unique_tasks = torch.unique(task_ids)
        num_tasks = unique_tasks.numel()

        # If only one task, skip loss computation (return 0)
        if num_tasks < 2:
            return torch.tensor(0.0, device=query.device, dtype=query.dtype)

        # Compute per-task centroids
        centroids = []
        for t in unique_tasks:
            mask = (task_ids == t)
            if mask.sum() > 0:
                centroid = per_sample_query[mask].mean(dim=0)  # (k_dim,)
                centroids.append(centroid)

        # Stack centroids: (num_tasks, k_dim)
        centroids = torch.stack(centroids, dim=0)

        # Normalize centroids for cosine similarity
        centroids_norm = F.normalize(centroids, p=2, dim=1)

        # Compute pairwise cosine similarity: (num_tasks, num_tasks)
        cos_sim = torch.mm(centroids_norm, centroids_norm.t())

        # Log intra-task vs inter-task cosine similarity diagnostics (per-sample level)
        with torch.no_grad():
            z_norm = F.normalize(per_sample_query.float(), p=2, dim=1)
            raw_sim = torch.mm(z_norm, z_norm.t())
            self_mask = torch.eye(B, dtype=torch.bool, device=query.device)
            same_task = (task_ids.unsqueeze(0) == task_ids.unsqueeze(1)) & ~self_mask
            diff_task = (task_ids.unsqueeze(0) != task_ids.unsqueeze(1))
            self.last_query_intra_sim = float(raw_sim[same_task].mean().item()) if same_task.any() else None
            self.last_query_inter_sim = float(raw_sim[diff_task].mean().item()) if diff_task.any() else None

        # Extract upper triangle (excluding diagonal) for pairwise loss
        num_pairs = 0
        contrastive_loss = torch.tensor(0.0, device=query.device, dtype=torch.float32)

        for i in range(num_tasks):
            for j in range(i + 1, num_tasks):
                sim = cos_sim[i, j]
                # Hinge-style loss with margin: max(0, cos_sim - margin)
                # If margin=0, this is just the cosine similarity
                pair_loss = torch.clamp(sim - self.contrastive_margin, min=0.0)
                contrastive_loss = contrastive_loss + pair_loss
                num_pairs += 1

        # Normalize by number of pairs
        if num_pairs > 0:
            contrastive_loss = contrastive_loss / num_pairs

        return contrastive_loss

    def _compute_sample_contrastive_loss(
        self, query: torch.Tensor, B: int, T: int, task_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Supervised contrastive loss (SupCon, Khosla et al. 2020) on per-sample
        query vectors.

        For every sample *i* in the batch, all other samples sharing the same
        task_id are positives and the rest are negatives.  The loss pulls
        same-task queries together and pushes cross-task queries apart in the
        normalized cosine-similarity space.

        L = - (1 / |P(i)|) * sum_{p in P(i)} log(
                exp(sim(z_i, z_p) / tau) /
                sum_{a in D(i)} exp(sim(z_i, z_a) / tau)
            )

        averaged over all samples *i* that have at least one positive.

        When ``contrastive_negatives_only=False`` (default, standard SupCon),
        D(i) = all samples except i. When ``contrastive_negatives_only=True``,
        D(i) = only cross-task samples, removing the intra-class uniformity
        pressure that can cause representation collapse at high loss weights.

        `contrastive_margin` is repurposed as a temperature offset: the
        effective temperature is ``max(0.07, contrastive_margin)`` when
        ``contrastive_margin > 0``, otherwise a sensible default of 0.07 is
        used.

        Args:
            query: Query vectors of shape (B*T*heads, k_dim)
            B: Batch size
            T: Sequence length
            task_ids: Task indices of shape (B,)

        Returns:
            Scalar contrastive loss tensor
        """
        # ---- per-sample representation: mean over T*heads -> (B, k_dim) ----
        query_reshaped = query.view(B, T * self.heads, self.k_dim)
        z = query_reshaped.mean(dim=1)  # (B, k_dim)
        labels = task_ids.view(-1).to(device=query.device, dtype=torch.long)

        # Build key set = current batch (+ optional queue entries from previous batches).
        queue_z, queue_labels = self._get_contrastive_queue(device=query.device, dtype=z.dtype)
        if queue_z is not None and queue_labels is not None:
            key_z = torch.cat([z, queue_z], dim=0)
            key_labels = torch.cat([labels, queue_labels], dim=0)
        else:
            key_z = z
            key_labels = labels

        loss = torch.tensor(0.0, device=query.device, dtype=torch.float32)

        # Keep original semantics: require at least 2 in-batch anchors and at least 2 tasks
        # in the contrastive set (batch + queue).
        if B >= 2 and torch.unique(key_labels).numel() >= 2:
            # Temperature
            tau = max(0.07, self.contrastive_margin) if self.contrastive_margin > 0 else 0.07

            # Anchors are current batch only; keys include optional queue.
            anchors = F.normalize(z.float(), p=2, dim=1)  # (B, k_dim)
            keys = F.normalize(key_z.float(), p=2, dim=1)  # (N, k_dim)
            raw_sim = torch.mm(anchors, keys.t())  # (B, N)

            # Positive pairs are same-task keys except the anchor itself (first B keys).
            pos_mask = labels.unsqueeze(1) == key_labels.unsqueeze(0)  # (B, N)
            self_mask = torch.zeros((B, key_z.shape[0]), dtype=torch.bool, device=query.device)
            self_mask[:, :B] = torch.eye(B, dtype=torch.bool, device=query.device)
            pos_mask = pos_mask & ~self_mask
            inter_mask = labels.unsqueeze(1) != key_labels.unsqueeze(0)

            # Log intra-task vs inter-task cosine similarity diagnostics.
            with torch.no_grad():
                if pos_mask.any():
                    self.last_query_intra_sim = float(raw_sim[pos_mask].mean().item())
                else:
                    self.last_query_intra_sim = None
                if inter_mask.any():
                    self.last_query_inter_sim = float(raw_sim[inter_mask].mean().item())
                else:
                    self.last_query_inter_sim = None

            # Apply temperature scaling for loss computation.
            sim_matrix = raw_sim / tau
            logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
            logits = sim_matrix - logits_max.detach()

            # Denominator: negatives-only (cross-task keys) or all non-self keys.
            if self.contrastive_negatives_only:
                denom_mask = inter_mask
            else:
                denom_mask = ~self_mask
            exp_logits = torch.exp(logits) * denom_mask.float()
            log_denom = torch.log(exp_logits.sum(dim=1, keepdim=True).clamp(min=1e-12))

            # Log-probabilities and per-anchor averaging over positive keys.
            log_prob = logits - log_denom
            num_pos = pos_mask.float().sum(dim=1)  # (B,)
            denom_count = denom_mask.float().sum(dim=1)  # (B,)
            valid = (num_pos > 0) & (denom_count > 0)
            mean_log_prob = (log_prob * pos_mask.float()).sum(dim=1) / num_pos.clamp(min=1.0)
            if valid.any():
                loss = -mean_log_prob[valid].mean()
        else:
            self.last_query_intra_sim = None
            self.last_query_inter_sim = None

        # Stage FIFO queue updates and flush them after the optimizer sync step.
        # This keeps the live queue stable across checkpoint recomputation.
        self._stage_contrastive_queries(z=z, labels=labels)

        return loss

    def _compute_subkey_scores(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # query: (bs*heads, k_dim)
        bs = query.shape[0] // self.heads
        query = query.view(bs, self.heads, self.k_dim)
        half = self.k_dim // 2

        keys = self.keys.view(self.heads, 2, self.n_keys, half)
        k1, k2 = keys[:, 0], keys[:, 1]

        q1, q2 = query[..., :half], query[..., half:]
        s1 = torch.einsum("blh,lkh->blk", q1, k1)  # (bs, heads, n_keys)
        s2 = torch.einsum("blh,lkh->blk", q2, k2)
        return s1, s2

    def _compute_routing_losses(
        self,
        s1_full: torch.Tensor,
        s2_full: torch.Tensor,
        B: int,
        T: int,
        task_ids: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, float | None, float | None, float | None]:
        """Compute routing losses on the joint product-key candidate distribution.

        Instead of operating on the two PQ half-marginals independently,
        this takes top-M subkeys per half, forms the M*M Cartesian product
        candidates (matching what retrieval actually does), softmaxes over
        the joint candidate scores, scatters per-task distributions into a
        compact slot histogram, and computes locality/separation losses on
        those full-slot distributions.
        """
        # When a cross-batch routing queue is enabled, use the dense-histogram
        # path that pushes the current (differentiable) per-task distributions
        # away from queued, all-task reference distributions.
        if self.routing_query_queue > 0:
            return self._routing_losses_queued(s1_full, s2_full, B, T, task_ids)

        eps = 1e-12
        M = self.routing_loss_topk if self.routing_loss_topk > 0 else self.knn
        candidate_pool = M * M
        device = s1_full.device

        # Support bounds for the final joint slot distribution
        default_min_support = max(self.knn, 8)
        default_max_support = max(8 * self.knn, candidate_pool // 2)
        min_support = self.routing_intra_task_min_support or default_min_support
        max_support = self.routing_intra_task_max_support or default_max_support
        min_support = max(1, min_support)
        max_support = max(1, max_support)
        if min_support > max_support:
            min_support, max_support = max_support, min_support

        # Normalize entropy by log of the full slot space
        log_norm = math.log(max(self.size, 2))
        min_entropy = math.log(max(min_support, 1)) / log_norm
        max_entropy = math.log(max(max_support, 1)) / log_norm
        score_scale = math.sqrt(max(self.k_dim // 2, 1))

        bs = s1_full.shape[0]  # B * T

        # Top-M subkeys in each half
        s1_top, i1_top = s1_full.topk(M, dim=2)  # (bs, heads, M)
        s2_top, i2_top = s2_full.topk(M, dim=2)  # (bs, heads, M)

        # Joint candidate scores and slot IDs via Cartesian product
        joint_scores = (
            s1_top.unsqueeze(3) + s2_top.unsqueeze(2)
        ).reshape(bs, self.heads, candidate_pool)
        joint_ids = (
            i1_top.unsqueeze(3) * self.n_keys + i2_top.unsqueeze(2)
        ).reshape(bs, self.heads, candidate_pool)

        # Softmax over the M^2 joint candidates per sample per head
        joint_probs = F.softmax(joint_scores.float() / score_scale, dim=-1)

        # Reshape for task grouping: (B, T, heads, M^2)
        joint_probs = joint_probs.view(B, T, self.heads, candidate_pool)
        joint_ids = joint_ids.view(B, T, self.heads, candidate_pool)

        # Compact the slot ID space: map the unique slot IDs that appear
        # in this batch to a dense [0, n_compact) range to save memory.
        all_unique = joint_ids.reshape(-1).unique()  # sorted
        n_compact = all_unique.numel()
        compact_ids = torch.searchsorted(all_unique, joint_ids)  # (B, T, heads, M^2)

        unique_tasks = torch.unique(task_ids)

        task_distributions: list[torch.Tensor] = []
        locality_terms: list[torch.Tensor] = []
        similarity_terms: list[torch.Tensor] = []
        global_terms: list[torch.Tensor] = []
        task_entropy_terms: list[float] = []
        task_support_terms: list[float] = []
        global_entropy_terms: list[float] = []

        # Head offsets for vectorised scatter across heads
        head_offsets = torch.arange(self.heads, device=device).view(1, self.heads, 1) * n_compact

        for task_id in unique_tasks.tolist():
            mask = task_ids == int(task_id)  # (B,)
            if not mask.any():
                continue

            # This task's candidate probs and compact IDs
            t_probs = joint_probs[mask]    # (n_b, T, heads, M^2)
            t_cids = compact_ids[mask]     # (n_b, T, heads, M^2)

            n_samples = t_probs.shape[0] * t_probs.shape[1]
            t_probs_flat = t_probs.reshape(n_samples, self.heads, candidate_pool)
            t_cids_flat = t_cids.reshape(n_samples, self.heads, candidate_pool).long()

            # Scatter into compact histogram, vectorised across heads
            ids_offset = (t_cids_flat + head_offsets).reshape(-1)
            probs_src = t_probs_flat.reshape(-1)

            hist = torch.zeros(
                self.heads * n_compact, device=device, dtype=torch.float32
            ).scatter_add(0, ids_offset, probs_src).view(self.heads, n_compact)

            # Normalise to a probability distribution per head
            hist = hist / hist.sum(dim=-1, keepdim=True).clamp(min=eps)
            task_distributions.append(hist)

            # Entropy over the full slot distribution (per head).
            # Use clamp before log instead of torch.where to avoid NaN
            # gradients from the unused log(0) branch.
            task_H = -(hist * hist.clamp(min=eps).log()).sum(dim=-1)  # (heads,)
            task_H_norm = task_H / log_norm

            task_entropy_terms.append(float(task_H_norm.mean().item()))
            task_support = torch.exp(task_H)
            task_support_terms.append(float(task_support.mean().item()))

            if self.routing_intra_task_locality_weight > 0:
                locality_terms.append(
                    (
                        F.relu(min_entropy - task_H_norm).pow(2)
                        + F.relu(task_H_norm - max_entropy).pow(2)
                    ).mean()
                )

        # Inter-task separation: cosine similarity between task slot distributions
        if len(task_distributions) >= 2 and self.routing_inter_task_separation_weight > 0:
            task_stack = torch.stack(task_distributions)  # (num_tasks, heads, n_compact)
            task_norm = F.normalize(task_stack, p=2, dim=-1)
            pair_sims: list[torch.Tensor] = []
            for i in range(len(task_distributions)):
                for j in range(i + 1, len(task_distributions)):
                    pair_sims.append((task_norm[i] * task_norm[j]).sum(dim=-1).mean())
            if pair_sims:
                similarity_terms.append(torch.stack(pair_sims).mean())

        # Global balance
        if len(task_distributions) >= 2 and self.routing_global_balance_weight > 0:
            task_stack = torch.stack(task_distributions)
            global_prob = task_stack.mean(dim=0)
            global_prob = global_prob / global_prob.sum(dim=-1, keepdim=True).clamp(min=eps)
            global_H = -(global_prob * global_prob.clamp(min=eps).log()).sum(dim=-1) / log_norm
            global_terms.append((1.0 - global_H).mean())
            global_entropy_terms.append(float(global_H.mean().item()))

        locality = torch.stack(locality_terms).mean() if locality_terms else None
        similarity = torch.stack(similarity_terms).mean() if similarity_terms else None
        global_balance = torch.stack(global_terms).mean() if global_terms else None
        task_entropy_mean = (
            float(sum(task_entropy_terms) / len(task_entropy_terms)) if task_entropy_terms else None
        )
        task_support_mean = (
            float(sum(task_support_terms) / len(task_support_terms)) if task_support_terms else None
        )
        global_entropy_mean = (
            float(sum(global_entropy_terms) / len(global_entropy_terms)) if global_entropy_terms else None
        )
        return locality, similarity, global_balance, task_entropy_mean, task_support_mean, global_entropy_mean

    def _routing_losses_queued(
        self,
        s1_full: torch.Tensor,
        s2_full: torch.Tensor,
        B: int,
        T: int,
        task_ids: torch.Tensor,
    ):
        """Routing losses with a cross-batch reference queue.

        Builds per-task slot distributions over the FULL n_keys**2 space (dense;
        numerically identical to the compact path since untouched slots are 0).
        The current batch's per-task histograms are differentiable and carry the
        gradient. The separation term pushes them away from per-task REFERENCE
        histograms recomputed from the queued queries against the CURRENT keys
        (no grad to the stored queries), which cover all recently-seen tasks ->
        fixes both the ~1-sample/task estimate and the partial-task-coverage of
        the in-batch-only loss. Locality / global-balance use the current
        histograms (they need gradient), exactly as before.
        """
        eps = 1e-12
        device = s1_full.device
        size = self.size
        heads = self.heads
        M = self.routing_loss_topk if self.routing_loss_topk > 0 else self.knn
        candidate_pool = M * M
        score_scale = math.sqrt(max(self.k_dim // 2, 1))
        log_norm = math.log(max(size, 2))
        head_off = torch.arange(heads, device=device).view(1, heads, 1) * size

        sep_weight = self.routing_inter_task_separation_weight
        loc_weight = self.routing_intra_task_locality_weight
        gb_weight = self.routing_global_balance_weight

        # Support band -> normalized-entropy band (same definition as compact path)
        default_min_support = max(self.knn, 8)
        default_max_support = max(8 * self.knn, candidate_pool // 2)
        min_support = self.routing_intra_task_min_support or default_min_support
        max_support = self.routing_intra_task_max_support or default_max_support
        min_support = max(1, min_support)
        max_support = max(1, max_support)
        if min_support > max_support:
            min_support, max_support = max_support, min_support
        min_entropy = math.log(max(min_support, 1)) / log_norm
        max_entropy = math.log(max(max_support, 1)) / log_norm

        def _task_hist_raw(probs_rows, ids_rows, task_idx_rows, n_tasks, accum=None):
            # probs_rows/ids_rows: (R, heads, M^2); task_idx_rows: (R,) dense idx.
            # Scatter into a flat (n_tasks*heads*size) buffer; returns raw (unnormalized).
            r = probs_rows.shape[0]
            hi = torch.arange(heads, device=device).view(1, heads, 1)
            off = ((task_idx_rows.view(r, 1, 1) * heads + hi) * size + ids_rows.long()).reshape(-1)
            src = probs_rows.reshape(-1).float()
            if accum is None:
                accum = torch.zeros(n_tasks * heads * size, device=device, dtype=torch.float32)
            return accum.scatter_add(0, off, src)

        def _normalize(raw, n_tasks):
            h = raw.view(n_tasks, heads, size)
            return h / h.sum(dim=-1, keepdim=True).clamp(min=eps)

        # ---- current batch: differentiable per-task histograms (vectorized) ----
        s1_top, i1_top = s1_full.topk(M, dim=2)
        s2_top, i2_top = s2_full.topk(M, dim=2)
        bs = s1_full.shape[0]
        joint_scores = (s1_top.unsqueeze(3) + s2_top.unsqueeze(2)).reshape(bs, heads, candidate_pool)
        joint_ids = (i1_top.unsqueeze(3) * self.n_keys + i2_top.unsqueeze(2)).reshape(bs, heads, candidate_pool)
        joint_probs = F.softmax(joint_scores.float() / score_scale, dim=-1)

        cur_unique = torch.unique(task_ids)
        cur_id_list = cur_unique.tolist()
        n_cur = len(cur_id_list)
        cur_remap = {int(t): i for i, t in enumerate(cur_id_list)}
        sample_idx = torch.tensor([cur_remap[int(t)] for t in task_ids.tolist()], device=device)
        row_idx = sample_idx.view(B, 1).expand(B, T).reshape(B * T)
        cur_raw = _task_hist_raw(joint_probs, joint_ids, row_idx, n_cur)
        cur_h = _normalize(cur_raw, n_cur)  # (n_cur, heads, size) differentiable

        task_H = -(cur_h * cur_h.clamp(min=eps).log()).sum(dim=-1)  # (n_cur, heads)
        task_H_norm = task_H / log_norm
        task_entropy_mean = float(task_H_norm.mean().item())
        task_support_mean = float(torch.exp(task_H).mean().item())
        locality = None
        if loc_weight > 0:
            locality = (F.relu(min_entropy - task_H_norm).pow(2) + F.relu(task_H_norm - max_entropy).pow(2)).mean()

        # ---- queue: detached reference histograms recomputed vs CURRENT keys ----
        ref_h = None
        ref_id_list: list[int] = []
        qz, qlab = self._get_routing_queue(device, self.keys.dtype)
        if qz is not None and sep_weight > 0:
            ref_id_list = torch.unique(qlab).tolist()
            n_ref = len(ref_id_list)
            ref_remap = {int(t): i for i, t in enumerate(ref_id_list)}
            CHUNK = 4096
            with torch.no_grad():
                ref_raw = torch.zeros(n_ref * heads * size, device=device, dtype=torch.float32)
                for c0 in range(0, qz.shape[0], CHUNK):
                    qc = qz[c0:c0 + CHUNK]
                    lc = qlab[c0:c0 + CHUNK]
                    r = qc.shape[0]
                    s1c, s2c = self._compute_subkey_scores(qc.reshape(r * heads, self.k_dim))
                    s1ct, i1ct = s1c.topk(M, dim=2)
                    s2ct, i2ct = s2c.topk(M, dim=2)
                    jsc = (s1ct.unsqueeze(3) + s2ct.unsqueeze(2)).reshape(r, heads, candidate_pool)
                    jic = (i1ct.unsqueeze(3) * self.n_keys + i2ct.unsqueeze(2)).reshape(r, heads, candidate_pool)
                    jpc = F.softmax(jsc.float() / score_scale, dim=-1)
                    lc_idx = torch.tensor([ref_remap[int(t)] for t in lc.tolist()], device=device)
                    ref_raw = _task_hist_raw(jpc, jic, lc_idx, n_ref, accum=ref_raw)
                ref_h = _normalize(ref_raw, n_ref)  # (n_ref, heads, size) detached

        # ---- separation (vectorized einsum; mean cosine over heads, masked i==j) ----
        similarity = None
        if sep_weight > 0:
            if ref_h is not None:
                cur_n = F.normalize(cur_h, p=2, dim=-1)
                ref_n = F.normalize(ref_h, p=2, dim=-1)
                sim = torch.einsum("ihs,jhs->ij", cur_n, ref_n) / heads  # (n_cur, n_ref)
                cur_ids_t = torch.tensor(cur_id_list, device=device).view(n_cur, 1)
                ref_ids_t = torch.tensor(ref_id_list, device=device).view(1, n_ref)
                valid = (cur_ids_t != ref_ids_t).float()
                similarity = (sim * valid).sum() / valid.sum().clamp(min=1.0)
            elif n_cur >= 2:
                # Warmup (queue empty): current-vs-current.
                cur_n = F.normalize(cur_h, p=2, dim=-1)
                sim = torch.einsum("ihs,jhs->ij", cur_n, cur_n) / heads
                valid = 1.0 - torch.eye(n_cur, device=device)
                similarity = (sim * valid).sum() / valid.sum().clamp(min=1.0)

        # ---- global balance (current histograms) ----
        global_balance = None
        global_entropy_mean = None
        if gb_weight > 0 and n_cur >= 2:
            global_prob = cur_h.mean(dim=0)
            global_prob = global_prob / global_prob.sum(dim=-1, keepdim=True).clamp(min=eps)
            global_H = -(global_prob * global_prob.clamp(min=eps).log()).sum(dim=-1) / log_norm
            global_balance = (1.0 - global_H).mean()
            global_entropy_mean = float(global_H.mean().item())

        return locality, similarity, global_balance, task_entropy_mean, task_support_mean, global_entropy_mean

    def _get_indices_from_subkey_scores(self, s1_full: torch.Tensor, s2_full: torch.Tensor):
        bs = s1_full.shape[0]
        s1, i1 = s1_full.topk(self.knn, dim=2, largest=True)
        s2, i2 = s2_full.topk(self.knn, dim=2, largest=True)

        all_s = (s1.unsqueeze(3) + s2.unsqueeze(2)).reshape(bs, self.heads, -1)
        all_i = (i1.unsqueeze(3) * self.n_keys + i2.unsqueeze(2)).reshape(bs, self.heads, -1)

        s, best = torch.topk(all_s, k=self.knn, dim=2, largest=True, sorted=True)
        idx = all_i.gather(2, best)
        return s.view(bs * self.heads, self.knn), idx.view(bs * self.heads, self.knn)


class MLPPlusMemory(nn.Module):
    def __init__(self, base_mlp: nn.Module, dim: int, cfg: MemoryLayerConfig, lang_dim: int = 0,
                 lora_rank_override: int | None = None, value_noise_sigma_override: float | None = None):
        super().__init__()
        self.mlp = base_mlp
        self.mem = HashingMemoryLite(dim, dim, cfg, lang_dim=lang_dim, lora_rank_override=lora_rank_override,
                                     value_noise_sigma_override=value_noise_sigma_override)
        self.memory_only = getattr(cfg, "memory_only", False)
        self.lang_dim = lang_dim
        # Text-span attachment (E44, VLM-side memory): when > 0, memory applies ONLY to the
        # last-N sequence positions (pi05's tokenized language field) — retrieval is computed
        # on that slice alone, the rest of the sequence passes through the plain MLP.
        self.text_span = int(getattr(cfg, "text_span", 0) or 0)
        if self.text_span > 0 and self.memory_only:
            raise ValueError("text_span attachment is incompatible with memory_only")
        # Pooled routing for the state sub-span (E45; see MemoryLayerConfig.vlm_router_pool).
        # Router keys only — the value path always consumes the live per-position hidden.
        self.router_pool = str(getattr(cfg, "vlm_router_pool", "") or "") if self.text_span > 0 else ""
        _w = list(getattr(cfg, "vlm_router_pool_weights", [1.0, 1.0]) or [1.0, 1.0])
        self.router_pool_w = (float(_w[0]), float(_w[1]) if len(_w) > 1 else 1.0)
        # E46: gate on the compact route-once path. False forces the legacy broadcast
        # path (shared key routed at every state position -> losses/queues weight the
        # palette by served positions). See MemoryLayerConfig.vlm_route_once.
        self.route_once = bool(getattr(cfg, "vlm_route_once", True))
        if self.router_pool not in ("", "anchored", "state"):
            raise ValueError(f"unknown vlm_router_pool mode: {self.router_pool!r}")
        # E49 image-span pooled routing: side of the per-camera region grid (0 = off).
        # Requires the anchored pooled mode (the design is instr-anchor + region offset).
        self.image_regions = int(getattr(cfg, "image_regions", 0) or 0) if self.text_span > 0 else 0
        _iw = list(getattr(cfg, "image_pool_weights", [1.0, 0.5]) or [1.0, 0.5])
        self.image_pool_w = (float(_iw[0]), float(_iw[1]) if len(_iw) > 1 else 0.5)
        if self.image_regions > 0 and self.router_pool != "anchored":
            raise ValueError("vlm_image_regions requires vlm_router_pool='anchored'")
        if self.image_regions > 0 and 16 % self.image_regions != 0:
            raise ValueError(f"vlm_image_regions must divide 16, got {self.image_regions}")
        self._img_region_idx = None  # (s*s, (16//s)^2) flat patch ids within one camera
        self._warned_img_fallback = False
        # E52 expert-anchor pooled routing (expert-tower wrappers only; the VLM derived
        # cfg nulls the field and text_span > 0 disables it structurally). The model
        # captures the pooled LM instruction hidden at this wrapper's paired LM layer
        # (detached — routing is a frozen function of the backbone) and hands it in via
        # set_expert_anchor; the mix applies to the ROUTING view only (query + gate).
        self.expert_anchor = (
            str(getattr(cfg, "expert_anchor_pool", "") or "") if self.text_span == 0 else ""
        )
        if self.expert_anchor not in ("", "text"):
            raise ValueError(f"unknown expert_anchor_pool mode: {self.expert_anchor!r}")
        self.expert_anchor_w = float(getattr(cfg, "expert_anchor_weight", 0.5))
        self.anchor_proj = None
        if self.expert_anchor:
            if not (0.0 <= self.expert_anchor_w <= 1.0):
                raise ValueError(
                    f"expert_anchor_weight must be in [0, 1], got {self.expert_anchor_w}"
                )
            src = int(getattr(cfg, "expert_anchor_src_dim", 2048))
            self.anchor_proj = nn.Linear(src, dim, bias=False)
            for p in self.anchor_proj.parameters():
                # Router group: trains in router warm-ups, frozen by freeze_memory_router.
                p.pk_query_proj_param = True
        self._ctx_anchor = None        # (B, src) detached pooled instr hidden
        self._ctx_anchor_valid = None  # (B,) bool — rows with a usable instruction span
        self._warned_anchor_stale = False
        # Frozen-base routing, inference path (suffix-only denoise forward): the
        # dual pass is driven by plain module state because the HF layer stack
        # between PaliGemmaWithExpertModel and this wrapper cannot thread a new
        # kwarg. Pass A (capture) bypasses memory and stashes the mlp input; pass
        # B pops it as router_x. The training path passes router_x explicitly and
        # never touches this state.
        self._frozen_capture = False
        self._frozen_stash: list[torch.Tensor] = []

    def begin_frozen_capture(self):
        if self.memory_only:
            raise RuntimeError("use_frozen_base_input_features is incompatible with memory_only")
        self._frozen_stash = []
        self._frozen_capture = True

    def end_frozen_capture(self):
        self._frozen_capture = False
        if len(self._frozen_stash) != 1:
            raise RuntimeError(
                f"frozen-routing capture expected exactly 1 stashed tensor, got {len(self._frozen_stash)}"
            )

    def assert_frozen_stash_consumed(self):
        if self._frozen_stash:
            raise RuntimeError("frozen-routing stash not consumed by the live pass")

    def drain_frozen_stash(self) -> torch.Tensor:
        """E59 frozen_prepass: extract the single captured routing feature so the joint
        training path can thread it as an explicit router_x argument (checkpoint-safe —
        the arg re-threads through gradient-checkpoint recompute, whereas a stash pop
        would be consumed on the first forward and missing on recompute)."""
        if len(self._frozen_stash) != 1:
            raise RuntimeError(
                f"frozen_prepass drain expected exactly 1 stashed tensor, got {len(self._frozen_stash)}"
            )
        return self._frozen_stash.pop(0)

    def set_expert_anchor(self, pooled: torch.Tensor | None, valid: torch.Tensor | None = None):
        """E52: install the pooled LM instruction hidden for this wrapper's paired LM
        layer. Overwrite semantics (duplicate prefix passes — VLM frozen pass A,
        grad-checkpoint recompute — recompute the identical value at anchor layers,
        which all sit below the VLM memory min). Persists across the suffix-only
        denoise passes at inference; the next prefix pass refreshes it."""
        self._ctx_anchor = pooled
        self._ctx_anchor_valid = valid

    def _mix_expert_anchor(self, base: torch.Tensor) -> torch.Tensor:
        """Anchored composite routing features (E52):
            B*rms_nrm(W_a @ anchor) + (1-B)*rms_nrm(token_p), rescaled to the
        batch-mean token RMS (E45 normalization: both components hard-normalized, so
        W_a's overall scale cancels and B is exactly the mix ratio). base: (B, T, D)
        routing view (frozen router_x when dual-path is on, else the live hidden).
        Rows without a usable instruction pool — and any stale/mismatched anchor —
        fall back to pure per-token features."""
        a = self._ctx_anchor
        if a is None or self.anchor_proj is None or self.expert_anchor_w <= 0.0:
            return base
        if a.shape[0] != base.shape[0]:
            if not self._warned_anchor_stale:
                logging.warning(
                    "expert-anchor: anchor batch %d != routing batch %d — per-token fallback",
                    a.shape[0], base.shape[0],
                )
                self._warned_anchor_stale = True
            return base
        bw = self.expert_anchor_w
        bf = base.float()
        tok_rms = bf.pow(2).mean(dim=-1).sqrt()            # (B, T)
        mean_rms = tok_rms.mean().clamp_min(1e-6)
        ap = self.anchor_proj(a.to(dtype=self.anchor_proj.weight.dtype, device=base.device))
        ap = ap.float()
        ap = ap / ap.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)  # (B, D)
        tn = bf / tok_rms.unsqueeze(-1).clamp_min(1e-6)
        mixed = (bw * ap.unsqueeze(1) + (1.0 - bw) * tn) * mean_rms
        if self._ctx_anchor_valid is not None:
            ok = self._ctx_anchor_valid.to(device=base.device).view(-1, 1, 1)
            mixed = torch.where(ok, mixed, bf)
        return mixed.to(base.dtype)

    def _pooled_router_keys(self, base: torch.Tensor, vm2: torch.Tensor) -> torch.Tensor:
        """Router keys for pooled state-sub-span routing (E45). Router keys ONLY — the
        value path always consumes the live per-position hidden.

        base: (B, T, D) routing features on the batch-max valid slice; vm2: (B, T) bool.
        Instruction positions [0, b_i) keep per-token keys. Every position from the
        boundary on (", State: ..." + tail) shares ONE per-sample key from RMS-normalized
        region means: "anchored" = a*nrm(instr pool) + b*nrm(state pool); "state" =
        nrm(state pool). Instr pool skips the constant "<bos> Task :" prefix (3 tokens);
        state pool skips the ", State :" markers (3) and the ";\\nAction: " tail (5).
        The composite is rescaled to the batch-mean valid-token RMS so keys stay
        in-distribution for the query projection. Rows without a usable boundary
        (marker missing / degenerate spans) fall back to per-token keys.
        """
        comp = self._pooled_components(base, vm2)
        if comp is None:
            return base
        k, il, v, row_ok = comp
        B, T, D = base.shape
        pos = torch.arange(T, device=base.device).unsqueeze(0)
        bmask = (pos >= il.unsqueeze(1)) & vm2 & row_ok.unsqueeze(1)
        out = torch.where(bmask.unsqueeze(-1), k.unsqueeze(1).expand(B, T, D), base.float())
        return out.to(base.dtype)

    def _pooled_components(self, base: torch.Tensor, vm2: torch.Tensor):
        """Shared math for the pooled router modes: returns (k, instr_len, valid, row_ok)
        with k the (B, C) float32 shared state-region key, or None when no row has a
        usable instruction/state boundary (callers fall back to per-token routing)."""
        il = getattr(self, "_ctx_instr_len", None)
        B, T, D = base.shape
        if il is None or il.shape[0] != B:
            return None
        il = il.to(base.device)
        v = vm2.sum(dim=1)
        pos = torch.arange(T, device=base.device).unsqueeze(0)
        bnd = il.unsqueeze(1)
        imask = (pos >= 3) & (pos < bnd) & vm2
        smask = (pos >= bnd + 3) & (pos < (v - 5).unsqueeze(1)) & vm2
        row_ok = (il > 4) & (smask.sum(dim=1) > 0) & (imask.sum(dim=1) > 0)
        if not bool(row_ok.any()):
            return None
        bf = base.float()

        def _pool(mask):
            m = mask.unsqueeze(-1).float()
            return (bf * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)

        def _nrm(u):
            return u / u.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)

        vmf = vm2.float()
        target = (bf.pow(2).mean(-1).sqrt() * vmf).sum(dim=1) / vmf.sum(dim=1).clamp_min(1.0)
        k = _nrm(_pool(smask))
        if self.router_pool == "anchored":
            a, b = self.router_pool_w
            k = a * _nrm(_pool(imask)) + b * k
        k = _nrm(k) * target.unsqueeze(-1)
        return k, il, v, row_ok

    def _route_once_pooled(self, xs, rs, vm2, comp, lang_emb, task_ids):
        """E45 route-once: every state-region position shares one router key, so the
        retrieval is computed ONCE per sample on a compact sequence [state key,
        instruction tokens] (key first so valid tokens stay a contiguous prefix for
        the loss machinery), then the shared palette is applied to each live state
        position via apply_shared_palette (params gathered once per row). Losses and
        queues see each unique routing row once; usage/TF stats keep the served-
        position multiplicity via stat_repeat. Returns the assembled (B, T, out)
        span memory output, zeroed outside valid positions."""
        k, il, v, row_ok = comp
        B, T, C = xs.shape
        bmax = max(int(il.max().item()), 1)
        xc = torch.cat([k.to(xs.dtype).unsqueeze(1), xs[:, :bmax]], dim=1)
        rc = None
        if rs is not None:
            rc = torch.cat([k.to(rs.dtype).unsqueeze(1), rs[:, :bmax]], dim=1)
        pos_i = torch.arange(bmax, device=xs.device).unsqueeze(0)
        instr_valid = (pos_i < il.unsqueeze(1)) & vm2[:, :bmax]
        mc = torch.cat(
            [torch.ones(B, 1, dtype=torch.bool, device=xs.device), instr_valid], dim=1
        )
        n_state = (v - il).clamp(min=1)
        srep = torch.cat(
            [n_state.unsqueeze(1), torch.ones(B, bmax, dtype=torch.long, device=xs.device)],
            dim=1,
        )
        mem_c, idx_v, w_v = self.mem(
            xc, lang_emb=lang_emb, task_ids=task_ids, router_x=rc,
            token_mask=mc, stat_repeat=srep, return_retrieval=True,
        )
        mem_out = xs.new_zeros(B, T, mem_c.shape[-1])
        mem_out[:, :bmax] = mem_c[:, 1:1 + bmax] * instr_valid.to(mem_c.dtype).unsqueeze(-1)
        nmax = max(int((v - il).max().item()), 1)
        offs = il.unsqueeze(1) + torch.arange(nmax, device=xs.device).unsqueeze(0)
        smask_pos = offs < v.unsqueeze(1)
        offs_c = offs.clamp(max=T - 1)
        x_state = xs.gather(1, offs_c.unsqueeze(-1).expand(B, nmax, C))
        pal = self.mem.apply_shared_palette(x_state, smask_pos, idx_v[:, 0], w_v[:, 0], k)
        bidx = torch.arange(B, device=xs.device).unsqueeze(1).expand(B, nmax)[smask_pos]
        mem_out[bidx, offs[smask_pos]] = pal[smask_pos].to(mem_out.dtype)
        return mem_out

    def _image_region_index(self, device):
        """Flat patch ids (within one camera's 256-position block) per spatial region:
        (s*s, (16//s)^2) long tensor, cached. Region order: row-major over the s x s grid."""
        if self._img_region_idx is None or self._img_region_idx.device != device:
            s = self.image_regions
            step = 16 // s
            grid = torch.arange(256, device=device).view(16, 16)
            self._img_region_idx = torch.stack([
                grid[r * step:(r + 1) * step, c * step:(c + 1) * step].reshape(-1)
                for r in range(s) for c in range(s)
            ])
        return self._img_region_idx

    def _image_span_context(self, x: torch.Tensor):
        """Validate the image-span preconditions on this batch. Returns
        (n_img, n_cam, n_act) or None (callers fall back to text-span-only).
        Active cameras are a contiguous prefix of the image block (prepare_images
        appends empty_cameras last with mask False) and activity must be row-constant."""
        ia = getattr(self, "_ctx_img_active", None)
        if ia is None or x.dim() != 3:
            return None
        n_img = x.shape[1] - self.text_span
        if n_img <= 0 or n_img % 256 != 0:
            return None
        n_cam = n_img // 256
        if ia.shape[0] != x.shape[0] or ia.shape[1] != n_cam:
            return None
        n_act_rows = ia.long().sum(dim=1)
        n_act = int(n_act_rows[0].item())
        if n_act == 0 or not bool((n_act_rows == n_act).all()):
            return None
        # contiguous-prefix check: the first n_act slots are the active ones
        if not bool(ia[:, :n_act].all()):
            return None
        return n_img, n_cam, n_act

    def _image_region_keys(self, base: torch.Tensor, n_act: int, instr_pool_nrm: torch.Tensor,
                           target: torch.Tensor):
        """Anchored pooled router keys for the image regions (E49). base: (B, S, D)
        routing features (full prefix); returns keys (B, K_act, D) float32 with
        K_act = n_act * s^2, plus the per-key GLOBAL position ids (K_act, region_size).
        k = rms_nrm( a * instr_pool_nrm + b * rms_nrm(region mean) ) * target — the same
        construction as the state key, so one query projection serves every key type."""
        s2 = self.image_regions ** 2
        ridx = self._image_region_index(base.device)          # (s2, rsize)
        cam_off = (torch.arange(n_act, device=base.device) * 256).view(n_act, 1, 1)
        pos_ids = (ridx.unsqueeze(0) + cam_off).reshape(n_act * s2, -1)  # (K_act, rsize)
        bf = base.float()
        gathered = bf[:, pos_ids.reshape(-1)].view(bf.shape[0], n_act * s2, pos_ids.shape[1], -1)
        region_mean = gathered.mean(dim=2)                     # (B, K_act, D)

        def _nrm(u):
            return u / u.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)

        a, b = self.image_pool_w
        k = a * instr_pool_nrm.unsqueeze(1) + b * _nrm(region_mean)
        k = _nrm(k) * target.view(-1, 1, 1)
        return k, pos_ids

    def _instr_pool_nrm(self, base_lang: torch.Tensor, vm2: torch.Tensor, il: torch.Tensor):
        """RMS-normalized instruction pool (B, D) float32 (positions [3, il), the anchor)."""
        pos = torch.arange(base_lang.shape[1], device=base_lang.device).unsqueeze(0)
        imask = (pos >= 3) & (pos < il.unsqueeze(1)) & vm2
        m = imask.unsqueeze(-1).float()
        pool = (base_lang.float() * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return pool / pool.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)

    def _route_once_pooled_img(self, x_full, r_full, xs, rs, vm2, comp, img_ctx, lang_emb, task_ids):
        """E49 route-once over image + language: ONE retrieval per (region key | state
        key | instruction token) on the compact sequence [K_act image keys, state key,
        instruction tokens], palettes applied per-position via apply_shared_palette.
        Returns the full-prefix (B, S, out) memory output (zeros at unserved positions)."""
        k_state, il, v, row_ok = comp
        n_img, n_cam, n_act = img_ctx
        B, S, C = x_full.shape
        base_full = x_full if r_full is None else r_full
        base_lang = xs if rs is None else rs
        target = (base_lang.float().pow(2).mean(-1).sqrt() * vm2.float()).sum(dim=1) / vm2.float().sum(dim=1).clamp_min(1.0)
        ip = self._instr_pool_nrm(base_lang, vm2, il)
        k_img, pos_ids = self._image_region_keys(base_full, n_act, ip, target)  # (B, K, D), (K, rs)
        K = k_img.shape[1]
        rsize = pos_ids.shape[1]
        bmax = max(int(il.max().item()), 1)
        xc = torch.cat([k_img.to(xs.dtype), k_state.to(xs.dtype).unsqueeze(1), xs[:, :bmax]], dim=1)
        rc = None
        if rs is not None:
            rc = torch.cat([k_img.to(rs.dtype), k_state.to(rs.dtype).unsqueeze(1), rs[:, :bmax]], dim=1)
        pos_i = torch.arange(bmax, device=xs.device).unsqueeze(0)
        instr_valid = (pos_i < il.unsqueeze(1)) & vm2[:, :bmax]
        mc = torch.cat([
            torch.ones(B, K + 1, dtype=torch.bool, device=xs.device), instr_valid,
        ], dim=1)
        n_state = (v - il).clamp(min=1)
        srep = torch.cat([
            torch.full((B, K), rsize, dtype=torch.long, device=xs.device),
            n_state.unsqueeze(1),
            torch.ones(B, bmax, dtype=torch.long, device=xs.device),
        ], dim=1)
        mem_c, idx_v, w_v = self.mem(
            xc, lang_emb=lang_emb, task_ids=task_ids, router_x=rc,
            token_mask=mc, stat_repeat=srep, return_retrieval=True,
        )
        mem_out = x_full.new_zeros(B, S, mem_c.shape[-1])
        lo = S - self.text_span
        # instruction tokens: per-token outputs from the compact call
        mem_out[:, lo:lo + bmax] = mem_c[:, K + 1:K + 1 + bmax] * instr_valid.to(mem_c.dtype).unsqueeze(-1)
        # state palette (row K)
        T_l = xs.shape[1]
        nmax = max(int((v - il).max().item()), 1)
        offs = il.unsqueeze(1) + torch.arange(nmax, device=xs.device).unsqueeze(0)
        smask_pos = offs < v.unsqueeze(1)
        offs_c = offs.clamp(max=T_l - 1)
        x_state = xs.gather(1, offs_c.unsqueeze(-1).expand(B, nmax, C))
        pal = self.mem.apply_shared_palette(x_state, smask_pos, idx_v[:, K], w_v[:, K], k_state)
        bidx = torch.arange(B, device=xs.device).unsqueeze(1).expand(B, nmax)[smask_pos]
        mem_out[bidx, lo + offs[smask_pos]] = pal[smask_pos].to(mem_out.dtype)
        # image region palettes (rows 0..K-1); positions are always valid for active cams
        ones_mask = torch.ones(B, rsize, dtype=torch.bool, device=xs.device)
        for j in range(K):
            x_reg = x_full[:, pos_ids[j]]
            pal_j = self.mem.apply_shared_palette(x_reg, ones_mask, idx_v[:, j], w_v[:, j], k_img[:, j])
            mem_out[:, pos_ids[j]] = pal_j.to(mem_out.dtype)
        return mem_out

    def _broadcast_img(self, x_full, r_full, xs, rs, vm2, comp, img_ctx, lang_emb, task_ids):
        """E49 literal-broadcast path over image + language (warm-ups): every position
        routes with its own key — image positions carry their region key, so the
        losses/queues weight each region key by its served-position count. Inactive
        camera positions are DROPPED from the mem input (valid tokens must be a
        contiguous prefix for the loss machinery). Pair with router_only_fast for VRAM."""
        k_state, il, v, row_ok = comp
        n_img, n_cam, n_act = img_ctx
        B, S, C = x_full.shape
        base_full = x_full if r_full is None else r_full
        base_lang = xs if rs is None else rs
        target = (base_lang.float().pow(2).mean(-1).sqrt() * vm2.float()).sum(dim=1) / vm2.float().sum(dim=1).clamp_min(1.0)
        ip = self._instr_pool_nrm(base_lang, vm2, il)
        k_img, pos_ids = self._image_region_keys(base_full, n_act, ip, target)
        K, rsize = pos_ids.shape
        n_ia = n_act * 256
        # router keys per position: image block <- region keys; language <- pooled state key
        r_img = base_full.new_zeros(B, n_ia, C)
        for j in range(K):
            r_img[:, pos_ids[j]] = k_img[:, j].unsqueeze(1).to(r_img.dtype)
        r_lang = self._pooled_router_keys(base_lang, vm2)
        x_in = torch.cat([x_full[:, :n_ia], xs], dim=1)
        r_in = torch.cat([r_img, r_lang], dim=1)
        m_in = torch.cat([
            torch.ones(B, n_ia, dtype=torch.bool, device=x_full.device), vm2,
        ], dim=1)
        mem_o = self.mem(x_in, lang_emb=lang_emb, task_ids=task_ids, router_x=r_in, token_mask=m_in)
        mem_o = mem_o * m_in.to(dtype=mem_o.dtype).unsqueeze(-1)
        mem_out = x_full.new_zeros(B, S, mem_o.shape[-1])
        mem_out[:, :n_ia] = mem_o[:, :n_ia]
        lo = S - self.text_span
        mem_out[:, lo:lo + xs.shape[1]] = mem_o[:, n_ia:]
        return mem_out

    def forward(self, x: torch.Tensor, lang_emb: torch.Tensor | None = None, task_ids: torch.Tensor | None = None,
                router_x: torch.Tensor | None = None):
        if self._frozen_capture:
            # Pass A of the inference dual forward: memory bypassed, record the
            # frozen (memory-free) routing features for the live pass.
            self._frozen_stash.append(x.detach())
            return self.mlp(x)
        if router_x is None and self._frozen_stash:
            router_x = self._frozen_stash.pop(0)
        if self.text_span > 0:
            n = self.text_span
            if x.dim() != 3 or x.shape[1] < n:
                # Sequence without a full language field (never expected on the pi05 prefix):
                # skip memory rather than misapply it to non-text positions.
                return self.mlp(x)
            # Pad exclusion (E44): the language field pads at the tail. With a valid-token
            # mask present, PADS NEVER REACH THE MODULE — the mem call runs on the batch-max
            # valid prefix of the field only (valid tokens are a contiguous field prefix), so
            # no query projection, routing, or value gather happens at pad positions at all.
            # Within the slice, shorter samples' tails are mask-zeroed (and mask-filtered from
            # stats/losses inside the mem). No mask context -> full-span legacy behavior.
            vm = getattr(self, "_ctx_valid_mask", None)
            if vm is not None and (vm.shape[0] != x.shape[0] or vm.shape[1] != n):
                vm = None  # shape mismatch (e.g. stale context) -> behave unmasked
            S = x.shape[1]
            lo = S - n
            if vm is not None:
                tmax = max(int(vm.sum(dim=1).max().item()), 1)
                hi = lo + tmax
                xs = x[:, lo:hi].contiguous()
                rs = router_x[:, lo:hi].contiguous() if router_x is not None else None
                vm2 = vm[:, :tmax]
                if self.router_pool:
                    base = xs if rs is None else rs
                    comp = self._pooled_components(base, vm2)
                    # E49 image-span dispatch: needs a usable state/instr boundary on
                    # every row (the instr pool anchors the image keys) plus a valid
                    # camera layout; anything missing falls back to text-span-only.
                    if self.image_regions > 0 and comp is not None and bool(comp[3].all()):
                        img_ctx = self._image_span_context(x)
                        if img_ctx is not None:
                            if self.route_once and not self.mem._slots_offloaded:
                                mem_full = self._route_once_pooled_img(
                                    x, router_x, xs, rs, vm2, comp, img_ctx, lang_emb, task_ids)
                            else:
                                mem_full = self._broadcast_img(
                                    x, router_x, xs, rs, vm2, comp, img_ctx, lang_emb, task_ids)
                            return self.mlp(x) + mem_full
                        elif not self._warned_img_fallback:
                            self._warned_img_fallback = True
                            logger.warning(
                                "vlm_image_regions>0 but no usable image context "
                                "(img_active missing / layout mismatch) — text-span-only fallback."
                            )
                    if (self.route_once and comp is not None and bool(comp[3].all())
                            and not self.mem._slots_offloaded):
                        mem_out = self._route_once_pooled(xs, rs, vm2, comp, lang_emb, task_ids)
                        out = self.mlp(x)
                        return torch.cat([out[:, :lo], out[:, lo:hi] + mem_out, out[:, hi:]], dim=1)
                    # Flag off, degenerate rows (missing boundary), or offloaded slots: the
                    # broadcast-key path (identical routing; the shared key is routed at every
                    # state position, so losses/queues carry served-position multiplicity).
                    rs = self._pooled_router_keys(base, vm2)
                mem_out = self.mem(xs, lang_emb=lang_emb, task_ids=task_ids, router_x=rs,
                                   token_mask=vm2)
                mem_out = mem_out * vm2.to(dtype=mem_out.dtype, device=mem_out.device).unsqueeze(-1)
                out = self.mlp(x)
                return torch.cat([out[:, :lo], out[:, lo:hi] + mem_out, out[:, hi:]], dim=1)
            xs = x[:, -n:].contiguous()
            rs = router_x[:, -n:].contiguous() if router_x is not None else None
            mem_out = self.mem(xs, lang_emb=lang_emb, task_ids=task_ids, router_x=rs)
            out = self.mlp(x)
            return torch.cat([out[:, :-n], out[:, -n:] + mem_out], dim=1)
        if self.expert_anchor:
            # E52: the routing view (query + gate) is the anchored composite; the value
            # path below still consumes the live per-position hidden x.
            r_in = self._mix_expert_anchor(router_x if router_x is not None else x)
            mem_out = self.mem(x, lang_emb=lang_emb, task_ids=task_ids, router_x=r_in)
        else:
            mem_out = self.mem(x, lang_emb=lang_emb, task_ids=task_ids, router_x=router_x)
        if self.memory_only:
            return mem_out
        return self.mlp(x) + mem_out


def _resolve_target_layers(num_expert_layers: int, layers: List[int] | str) -> List[int]:
    if layers:
        # Accept list[int] or a string like "[11,13,15]" or "11,13,15"
        parsed: List[int]
        if isinstance(layers, str):
            s = layers.strip()
            parsed = []
            try:
                obj = json.loads(s)
                if isinstance(obj, list):
                    parsed = [int(x) for x in obj]
                else:
                    parsed = []
            except Exception:
                try:
                    obj = ast.literal_eval(s)
                    if isinstance(obj, list):
                        parsed = [int(x) for x in obj]
                    else:
                        # fallback split on comma
                        parsed = [int(x.strip()) for x in s.split(",") if x.strip()]
                except Exception:
                    parsed = [int(x.strip()) for x in s.split(",") if x.strip()]
        else:
            parsed = [int(x) for x in layers]

        # Sanitize provided indices: keep within range and deduplicate while preserving order
        seen = set()
        cleaned: List[int] = []
        for li in parsed:
            if 0 <= li < num_expert_layers and li not in seen:
                cleaned.append(li)
                seen.add(li)
        return cleaned
    # Default: no layers
    return []


def _get_lang_dim(cfg: MemoryLayerConfig) -> int:
    if not getattr(cfg, "lang_to_query", False):
        return 0
    model_name = getattr(cfg, "embedding_model", "all-MiniLM-L6-v2")
    return EMBEDDING_DIM_MAP.get(model_name, 384)


def attach_memory_to_layer_list(
    layers,
    dim: int,
    cfg: MemoryLayerConfig,
    label: str = "EXPERT",
) -> List[int]:
    """
    Replace selected MLPs with MLPPlusMemory in-place on an arbitrary layer list.

    layers: an nn.ModuleList of transformer-style decoder layers, each having a `.mlp` attribute.
    dim: hidden size used for the memory module's input/output dimension.
    cfg: MemoryLayerConfig (enabled must be True at the callsite).
    label: human-readable name printed for debugging.

    Returns the resolved list of target layer indices.
    """
    num_layers = len(layers)
    target_layers = _resolve_target_layers(num_layers, cfg.layers)

    print(f"Target {label} layers for memory: {target_layers}")
    target_set = set(target_layers)

    # Optional per-layer LoRA ranks: matched to target_layers by order, overriding the
    # scalar lora_rank. Empty -> None override everywhere (scalar lora_rank; backward compat).
    _layer_ranks = list(getattr(cfg, "layer_ranks", []) or [])
    rank_by_layer: dict[int, int] = {}
    if _layer_ranks:
        if len(_layer_ranks) != len(target_layers):
            raise ValueError(
                f"layer_ranks has length {len(_layer_ranks)} but there are {len(target_layers)} memory "
                f"{label} layers {target_layers}; lengths must match (matched by order)."
            )
        rank_by_layer = {li: int(r) for li, r in zip(target_layers, _layer_ranks)}
        print(f"Per-layer LoRA ranks for {label}: " + ", ".join(f"L{li}=r{rank_by_layer[li]}" for li in target_layers))

    # Optional per-layer value-input-noise sigmas (E57), matched to target_layers by
    # order like layer_ranks. Noise is enabled on a module ONLY via this explicit
    # per-layer threading — unthreaded attach paths stay noise-free by construction.
    _vnoise = list(getattr(cfg, "value_input_noise_sigma", []) or [])
    vnoise_by_layer: dict[int, float] = {}
    if _vnoise:
        if len(_vnoise) != len(target_layers):
            raise ValueError(
                f"value_input_noise_sigma has length {len(_vnoise)} but there are {len(target_layers)} "
                f"memory {label} layers {target_layers}; lengths must match (matched by order)."
            )
        vnoise_by_layer = {li: float(s) for li, s in zip(target_layers, _vnoise)}
        if float(getattr(cfg, "value_input_noise_p", 0.0) or 0.0) > 0:
            print(f"Value-input noise for {label}: p={cfg.value_input_noise_p}, amp={list(getattr(cfg, 'value_input_noise_amp', [1.0, 1.0]))}, "
                  + ", ".join(f"L{li}=s{vnoise_by_layer[li]:g}" for li in target_layers))

    lang_dim = _get_lang_dim(cfg)
    if lang_dim > 0:
        print(f"Language-conditioned query projection enabled with lang_dim={lang_dim}")

    # First, unwrap any previously wrapped layers that are not in the target set
    for li in range(num_layers):
        layer = layers[li]
        if isinstance(getattr(layer, "mlp", None), MLPPlusMemory) and li not in target_set:
            layer.mlp = layer.mlp.mlp

    # Now, wrap exactly the requested target layers
    for li in target_layers:
        layer = layers[li]
        # Avoid double wrapping if already wrapped
        if isinstance(layer.mlp, MLPPlusMemory):
            continue
        base_dtype = next(layer.mlp.parameters()).dtype
        base_device = next(layer.mlp.parameters()).device
        layer.mlp = MLPPlusMemory(layer.mlp, dim=dim, cfg=cfg, lang_dim=lang_dim,
                                  lora_rank_override=rank_by_layer.get(li),
                                  value_noise_sigma_override=vnoise_by_layer.get(li))
        # Align non-value memory params to base dtype/device; keep value params in float32.
        # When CPU offload is requested, leave value (slot) params on CPU — moving them
        # to GPU here would defeat the whole point and OOM on small cards.
        offload_slots = bool(getattr(cfg, "offload_slots_to_cpu", False))
        for name, p in layer.mlp.mem.named_parameters():
            if getattr(p, "pk_value_param", False):
                if offload_slots:
                    p.data = p.data.to(dtype=torch.float32)  # stay on CPU
                else:
                    p.data = p.data.to(device=base_device, dtype=torch.float32)
            else:
                p.data = p.data.to(device=base_device, dtype=base_dtype)

    return target_layers


def attach_memory_to_expert(smolvla_model, cfg: MemoryLayerConfig):
    """
    Replace selected expert MLPs with MLPPlusMemory in-place.

    smolvla_model: SmolVLMWithExpertModel
    cfg: MemoryLayerConfig (enabled must be True at the callsite)
    """
    target_layers = attach_memory_to_layer_list(
        smolvla_model.lm_expert.layers,
        dim=smolvla_model.expert_hidden_size,
        cfg=cfg,
        label="EXPERT",
    )
    # Record for debugging/metrics
    try:
        smolvla_model.mem_target_layers = target_layers
    except Exception:
        pass


def attach_memory_to_backbones(smolvla_model, cfg: MemoryLayerConfig):
    """
    Replace selected MLPs with MLPPlusMemory for both the action expert and (optionally) the VLM backbone.

    smolvla_model: SmolVLMWithExpertModel
    cfg: MemoryLayerConfig (enabled must be True at the callsite)
    """
    # Expert attachment (reuse existing logic)
    attach_memory_to_expert(smolvla_model, cfg)

    # VLM text backbone attachment
    try:
        vlm_text_model = smolvla_model.get_vlm_model().text_model
    except Exception:
        vlm_text_model = None
    if vlm_text_model is None:
        return

    num_vlm_layers = len(vlm_text_model.layers)
    target_vlm_layers = _resolve_target_layers(num_vlm_layers, getattr(cfg, "vlm_layers", []))
    if not target_vlm_layers:
        return

    print(f"Target VLM layers for memory: {target_vlm_layers}")
    target_vlm_set = set(target_vlm_layers)

    lang_dim = _get_lang_dim(cfg)

    # Unwrap any previously wrapped VLM layers not in the target set
    for li in range(num_vlm_layers):
        layer = vlm_text_model.layers[li]
        if isinstance(getattr(layer, "mlp", None), MLPPlusMemory) and li not in target_vlm_set:
            layer.mlp = layer.mlp.mlp

    # Wrap the requested VLM layers
    for li in target_vlm_layers:
        layer = vlm_text_model.layers[li]
        # Avoid double wrapping if already wrapped
        if isinstance(layer.mlp, MLPPlusMemory):
            continue
        # Determine dimension/dtype/device from the base MLP
        dim = next(layer.mlp.parameters()).shape[-1]
        base_dtype = next(layer.mlp.parameters()).dtype
        base_device = next(layer.mlp.parameters()).device

        layer.mlp = MLPPlusMemory(layer.mlp, dim=dim, cfg=cfg, lang_dim=lang_dim)
        for name, p in layer.mlp.mem.named_parameters():
            if getattr(p, "pk_value_param", False):
                p.data = p.data.to(device=base_device, dtype=torch.float32)
            else:
                p.data = p.data.to(device=base_device, dtype=base_dtype)

    # Record for debugging/metrics
    try:
        smolvla_model.vlm_mem_target_layers = target_vlm_layers
    except Exception:
        pass


def split_memory_params(module: nn.Module):
    mem_vals, others = [], []
    for p in module.parameters():
        (mem_vals if getattr(p, "pk_value_param", False) else others).append(p)
    return mem_vals, others


def iter_memory_layers(root: nn.Module):
    """Yield ``(qualified_name, MLPPlusMemory)`` for every memory wrapper in root.

    Walks ``named_modules`` so this works for any policy regardless of where
    memory layers are attached (action expert, VLM backbone, etc.). Each
    wrapper is yielded once.
    """
    seen: set[int] = set()
    for name, module in root.named_modules():
        if isinstance(module, MLPPlusMemory) and id(module) not in seen:
            seen.add(id(module))
            yield name, module


def aggregate_memory_losses(
    root: nn.Module,
    base_loss: torch.Tensor,
    cfg: MemoryLayerConfig,
    loss_dict: dict | None = None,
    layer_label_fn=None,
) -> torch.Tensor:
    """Aggregate per-layer contrastive / routing / usage stats into the loss.

    Walks every ``MLPPlusMemory`` reachable from ``root`` and adds:
      - the (weighted) mean of per-layer contrastive losses
      - the (weighted) mean of per-layer routing locality / inter-task
        similarity / global balance losses
      - per-layer and aggregate slot usage diagnostics under ``loss_dict``
        when ``cfg.log_usage`` is true

    The returned tensor is ``base_loss`` plus all memory regularizers. When
    ``loss_dict`` is provided it is mutated in place with diagnostic keys.

    ``layer_label_fn(name) -> str`` lets the caller customize the prefix used
    in diagnostic keys (e.g. ``""`` for expert layers, ``"vlm_"`` for VLM text
    layers). The default returns ``""``.
    """
    if loss_dict is None:
        loss_dict = {}

    if layer_label_fn is None:
        def layer_label_fn(_name: str) -> str:
            return ""

    contrastive_loss_weight = float(getattr(cfg, "contrastive_loss_weight", 0.0))
    routing_intra_task_locality_weight = float(getattr(cfg, "routing_intra_task_locality_weight", 0.0))
    if routing_intra_task_locality_weight <= 0:
        routing_intra_task_locality_weight = float(getattr(cfg, "routing_compactness_weight", 0.0))
    routing_inter_task_separation_weight = float(getattr(cfg, "routing_inter_task_separation_weight", 0.0))
    if routing_inter_task_separation_weight <= 0:
        routing_inter_task_separation_weight = float(getattr(cfg, "routing_separation_weight", 0.0))
    routing_global_balance_weight = float(getattr(cfg, "routing_global_balance_weight", 0.0))
    log_usage = bool(getattr(cfg, "log_usage", False))

    contrastive_losses = []
    locality_losses = []
    similarity_losses = []
    global_balance_losses = []
    routing_intra_task_entropies = []
    routing_intra_task_supports = []
    routing_global_entropies = []

    used_fracs: list[float] = []
    perplexities: list[float] = []
    top1_shares: list[float] = []
    eff_nums: list[float] = []
    intra_sims: list[float] = []
    inter_sims: list[float] = []
    gate_means: list[float] = []

    loss = base_loss

    for name, wrapper in iter_memory_layers(root):
        mem = wrapper.mem
        prefix = layer_label_fn(name)
        # parse layer index from "...layers.{i}.mlp"
        import re as _re
        m = _re.search(r"layers\.(\d+)", name)
        li = int(m.group(1)) if m else 0

        # --- contrastive loss ---
        if contrastive_loss_weight > 0 and getattr(mem, "last_contrastive_loss", None) is not None:
            contrastive_losses.append(mem.last_contrastive_loss)
            loss_dict[f"{prefix}contrastive_loss_L{li}"] = mem.last_contrastive_loss.item()

        # --- routing losses ---
        if getattr(mem, "last_routing_intra_task_locality_loss", None) is not None:
            locality_losses.append(mem.last_routing_intra_task_locality_loss)
            v = mem.last_routing_intra_task_locality_loss.item()
            loss_dict[f"{prefix}routing_intra_task_locality_L{li}"] = v
            loss_dict[f"{prefix}routing_compactness_L{li}"] = v
        if getattr(mem, "last_routing_inter_task_similarity_loss", None) is not None:
            similarity_losses.append(mem.last_routing_inter_task_similarity_loss)
            v = mem.last_routing_inter_task_similarity_loss.item()
            loss_dict[f"{prefix}routing_inter_task_similarity_L{li}"] = v
            loss_dict[f"{prefix}routing_inter_task_separation_L{li}"] = 1.0 - v
            loss_dict[f"{prefix}routing_separation_L{li}"] = v
        if getattr(mem, "last_routing_global_balance_loss", None) is not None:
            global_balance_losses.append(mem.last_routing_global_balance_loss)
            loss_dict[f"{prefix}routing_global_balance_L{li}"] = mem.last_routing_global_balance_loss.item()
        if getattr(mem, "last_routing_intra_task_entropy", None) is not None:
            routing_intra_task_entropies.append(mem.last_routing_intra_task_entropy)
            loss_dict[f"{prefix}routing_intra_task_entropy_L{li}"] = mem.last_routing_intra_task_entropy
            loss_dict[f"{prefix}routing_task_entropy_L{li}"] = mem.last_routing_intra_task_entropy
        if getattr(mem, "last_routing_intra_task_support", None) is not None:
            routing_intra_task_supports.append(mem.last_routing_intra_task_support)
            loss_dict[f"{prefix}routing_intra_task_support_L{li}"] = mem.last_routing_intra_task_support
        if getattr(mem, "last_routing_global_entropy", None) is not None:
            routing_global_entropies.append(mem.last_routing_global_entropy)
            loss_dict[f"{prefix}routing_global_entropy_L{li}"] = mem.last_routing_global_entropy

        # --- usage logging ---
        if log_usage and getattr(mem, "last_indices", None) is not None:
            idx = mem.last_indices
            unique_count = torch.unique(idx).numel()
            frac = float(unique_count) / float(mem.size)
            usage_prefix = f"{prefix}mem_" if prefix else "mem_"
            loss_dict[f"{usage_prefix}used_count_L{li}"] = float(unique_count)
            loss_dict[f"{usage_prefix}used_frac_L{li}"] = frac
            used_fracs.append(frac)

            idx_flat = idx.reshape(-1)
            if getattr(mem, "last_weights", None) is not None:
                w_flat = mem.last_weights.reshape(-1).float()
            else:
                w_flat = torch.ones_like(idx_flat, dtype=torch.float32)
            uniq, inv = torch.unique(idx_flat, return_inverse=True)
            usage = torch.zeros(uniq.shape[0], dtype=torch.float32, device=idx_flat.device)
            usage.scatter_add_(0, inv, w_flat)
            total = usage.sum()
            if total > 0:
                p = usage / total
                eps = 1e-12
                entropy = -(p * (p + eps).log()).sum()
                perplexity = float(torch.exp(entropy).item())
                top1 = float(p.max().item())
                hhi = float((p * p).sum().item())
                eff_num = float(1.0 / max(hhi, eps))
                loss_dict[f"{usage_prefix}usage_perplexity_L{li}"] = perplexity
                loss_dict[f"{usage_prefix}usage_top1_share_L{li}"] = top1
                loss_dict[f"{usage_prefix}usage_effnum_L{li}"] = eff_num
                perplexities.append(perplexity)
                top1_shares.append(top1)
                eff_nums.append(eff_num)
            if getattr(mem, "last_query_intra_sim", None) is not None:
                loss_dict[f"{prefix}query_intra_sim_L{li}"] = mem.last_query_intra_sim
                intra_sims.append(mem.last_query_intra_sim)
            if getattr(mem, "last_query_inter_sim", None) is not None:
                loss_dict[f"{prefix}query_inter_sim_L{li}"] = mem.last_query_inter_sim
                inter_sims.append(mem.last_query_inter_sim)
            if getattr(mem, "last_gate_mean", None) is not None:
                loss_dict[f"{prefix}gate_mean_L{li}"] = mem.last_gate_mean
                gate_means.append(mem.last_gate_mean)

    # Add the regularizers to the loss
    if contrastive_loss_weight > 0 and contrastive_losses:
        total_contrastive = sum(contrastive_losses) / len(contrastive_losses)
        loss_dict["contrastive_loss_mean"] = total_contrastive.item()
        loss = loss + contrastive_loss_weight * total_contrastive
    if routing_intra_task_locality_weight > 0 and locality_losses:
        total_locality = sum(locality_losses) / len(locality_losses)
        v = total_locality.item()
        loss_dict["routing_intra_task_locality_mean"] = v
        loss_dict["routing_compactness_mean"] = v
        loss = loss + routing_intra_task_locality_weight * total_locality
    if routing_inter_task_separation_weight > 0 and similarity_losses:
        total_similarity = sum(similarity_losses) / len(similarity_losses)
        v = total_similarity.item()
        loss_dict["routing_inter_task_similarity_mean"] = v
        loss_dict["routing_inter_task_separation_mean"] = 1.0 - v
        loss_dict["routing_separation_mean"] = v
        loss = loss + routing_inter_task_separation_weight * total_similarity
    if routing_global_balance_weight > 0 and global_balance_losses:
        total_global_balance = sum(global_balance_losses) / len(global_balance_losses)
        loss_dict["routing_global_balance_mean"] = total_global_balance.item()
        loss = loss + routing_global_balance_weight * total_global_balance
    if routing_intra_task_entropies:
        loss_dict["routing_intra_task_entropy_mean"] = float(
            sum(routing_intra_task_entropies) / len(routing_intra_task_entropies)
        )
        loss_dict["routing_task_entropy_mean"] = loss_dict["routing_intra_task_entropy_mean"]
    if routing_intra_task_supports:
        loss_dict["routing_intra_task_support_mean"] = float(
            sum(routing_intra_task_supports) / len(routing_intra_task_supports)
        )
    if routing_global_entropies:
        loss_dict["routing_global_entropy_mean"] = float(
            sum(routing_global_entropies) / len(routing_global_entropies)
        )

    if used_fracs:
        loss_dict["mem_used_frac_mean"] = float(sum(used_fracs) / len(used_fracs))
    if perplexities:
        loss_dict["mem_usage_perplexity_mean"] = float(sum(perplexities) / len(perplexities))
    if top1_shares:
        loss_dict["mem_usage_top1_share_mean"] = float(sum(top1_shares) / len(top1_shares))
    if eff_nums:
        loss_dict["mem_usage_effnum_mean"] = float(sum(eff_nums) / len(eff_nums))
    if intra_sims:
        loss_dict["query_intra_sim_mean"] = float(sum(intra_sims) / len(intra_sims))
    if inter_sims:
        loss_dict["query_inter_sim_mean"] = float(sum(inter_sims) / len(inter_sims))
    if gate_means:
        loss_dict["gate_mean"] = float(sum(gate_means) / len(gate_means))

    return loss
