import ast
import contextlib
import json
import math
import threading
from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from .memory_config import MemoryLayerConfig


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


class QueryMLPLite(nn.Module):
    def __init__(self, input_dim: int, heads: int, k_dim: int, bias: bool = True, lang_dim: int = 0, fuse_method: str = "concat"):
        super().__init__()
        self.input_dim = input_dim
        self.heads = heads
        self.k_dim = k_dim
        self.lang_dim = lang_dim
        self.fuse_method = fuse_method

        if fuse_method not in ("concat", "film"):
            raise ValueError(f"Unknown fuse_method: {fuse_method}. Expected 'concat' or 'film'.")

        if fuse_method == "concat":
            proj_input_dim = input_dim + lang_dim
            self.proj = nn.Linear(proj_input_dim, heads * k_dim, bias=bias)
        else:
            self.proj = nn.Linear(input_dim, heads * k_dim, bias=bias)
            if lang_dim > 0:
                hidden_dim = max(lang_dim, heads * k_dim) // 2
                self.film_mlp = nn.Sequential(
                    nn.Linear(lang_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 2 * heads * k_dim),
                )

        try:
            self.proj.weight.pk_query_proj_param = True
            if self.proj.bias is not None:
                self.proj.bias.pk_query_proj_param = True
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

    def __init__(self, input_dim: int, output_dim: int, cfg: MemoryLayerConfig, lang_dim: int = 0):
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
        self.lora_rank = getattr(cfg, "lora_rank", 2)

        # Value corruption parameters
        self.corruption_prob = getattr(cfg, "corruption_prob", 0.0)
        self.corruption_std = getattr(cfg, "corruption_std", 0.1)

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
        self.query_proj = QueryMLPLite(self.input_dim, self.heads, self.k_dim, lang_dim=lang_dim, fuse_method=fuse_method)

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
        if not self._pending_contrastive_batches:
            return
        for z, labels in self._pending_contrastive_batches:
            self._enqueue_contrastive_queries(z=z, labels=labels)
        self._pending_contrastive_batches.clear()

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

        nn.init.xavier_uniform_(self.query_proj.proj.weight)
        if self.v_proj:
            nn.init.normal_(self.value_proj.weight, mean=0, std=self.output_dim ** -0.5)
        if self.swilu_proj:
            nn.init.normal_(self.swilu_projection.weight, mean=0, std=self.output_dim ** -0.5)
        if self.gating is not None:
            nn.init.normal_(self.gating.weight, mean=0, std=self.input_dim ** -0.5)

    def forward(self, x: torch.Tensor, lang_emb: torch.Tensor | None = None, task_ids: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T, C)
        # lang_emb: (B, lang_dim) or None
        # task_ids: (B,) optional task indices for contrastive loss
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

        # Query (with optional language conditioning)
        query = self.query_proj(x, lang_emb=lang_emb)  # (bs*heads, k_dim)

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
        if self.training and self.contrastive_loss_weight > 0 and task_ids is not None:
            if self.contrastive_method == "sample":
                self.last_contrastive_loss = self._compute_sample_contrastive_loss(query, B, T, task_ids)
            else:
                self.last_contrastive_loss = self._compute_contrastive_loss(query, B, T, task_ids)

        s1_full, s2_full = self._compute_subkey_scores(query)
        if self.training and task_ids is not None and (
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
            ) = self._compute_routing_losses(s1_full, s2_full, B, T, task_ids)
            self.last_routing_compactness_loss = self.last_routing_intra_task_locality_loss
            self.last_routing_separation_loss = self.last_routing_inter_task_similarity_loss
            self.last_routing_task_entropy = self.last_routing_intra_task_entropy

        # Indices and scores
        scores, indices = self._get_indices_from_subkey_scores(s1_full, s2_full)  # (bs*heads, knn)

        # Record selected indices/scores for analysis during eval
        if not self.training and self.EVAL_MEMORY:
            self.last_indices = indices.view(bs, self.heads, self.knn).detach().cpu()
            self.last_scores = scores.view(bs, self.heads, self.knn).detach().cpu().float()

        # Softmax in float32 for numerical stability; we will cast as needed later
        weights = F.softmax(scores.float(), dim=-1)
        # Merge heads
        indices = indices.view(bs, self.heads * self.knn)
        weights = weights.view(bs, self.heads * self.knn)

        # Record selected indices/weights during training when log_usage is enabled
        if self.training and self.log_usage:
            self.last_indices = indices.view(bs, self.heads, self.knn).detach()
            self.last_weights = weights.view(bs, self.heads, self.knn).detach()

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
        if self.value_type == "vector":
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
            gate = torch.sigmoid(self.gating(x_flat)).view(B, T, 1)
            if self.training and self.log_usage:
                self.last_gate_mean = float(gate.mean().item())
            out = gate * out
        return out

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

    def _forward_lora_values(
        self, x_flat: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        LoRA-based forward: run input through selected tiny LoRAs, weighted sum of outputs.

        Each slot is a low-rank transform: output_i = slot_up_i @ SiLU(slot_down_i @ x)
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
        x_fp32 = x_flat.float()

        # Gather LoRA weights for selected slots
        # slot_down: (n_slots, input_dim, rank) -> (bs, k, input_dim, rank)
        # slot_up: (n_slots, rank, v_dim) -> (bs, k, rank, v_dim)
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
        else:
            down_weights = self.slot_down[indices]  # (bs, k, input_dim, rank)
            up_weights = self.slot_up[indices]  # (bs, k, rank, v_dim)

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
    def __init__(self, base_mlp: nn.Module, dim: int, cfg: MemoryLayerConfig, lang_dim: int = 0):
        super().__init__()
        self.mlp = base_mlp
        self.mem = HashingMemoryLite(dim, dim, cfg, lang_dim=lang_dim)
        self.memory_only = getattr(cfg, "memory_only", False)
        self.lang_dim = lang_dim

    def forward(self, x: torch.Tensor, lang_emb: torch.Tensor | None = None, task_ids: torch.Tensor | None = None):
        mem_out = self.mem(x, lang_emb=lang_emb, task_ids=task_ids)
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
        layer.mlp = MLPPlusMemory(layer.mlp, dim=dim, cfg=cfg, lang_dim=lang_dim)
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
