import math
import ast
import json
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

        # Keys: (2 * heads * n_keys, k_dim // 2)
        # Keep dtype lightweight (bf16 if default is bf16), otherwise defaults to fp32.
        self.keys = nn.Parameter(torch.empty(2 * self.heads * self.n_keys, self.k_dim // 2))
        # Tag keys param to allow selective training
        try:
            self.keys.pk_keys_param = True
        except Exception:
            pass

        # Values (embedding table) kept in float32 for correct CUDA backward and stability
        self.values = nn.Parameter(torch.empty(self.size, self.v_dim, dtype=torch.float32))
        for p in [self.values]:
            p.pk_value_param = True
            p.fixed_lr = cfg.value_fixed_lr

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

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.k_dim)
        nn.init.uniform_(self.keys, a=-bound, b=bound)
        nn.init.normal_(self.values, mean=0, std=self.v_dim ** -0.5)
        nn.init.xavier_uniform_(self.query_proj.proj.weight)
        if self.v_proj:
            nn.init.normal_(self.value_proj.weight, mean=0, std=self.output_dim ** -0.5)
        if self.swilu_proj:
            nn.init.normal_(self.swilu_projection.weight, mean=0, std=self.output_dim ** -0.5)
        if self.gating is not None:
            nn.init.normal_(self.gating.weight, mean=0, std=self.input_dim ** -0.5)

    def forward(self, x: torch.Tensor, lang_emb: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T, C)
        # lang_emb: (B, lang_dim) or None
        # Ensure module parameters/buffers match the input dtype/device without recreating Parameters
        dtype, device = x.dtype, x.device
        if getattr(self, "_param_dtype", None) is not dtype or getattr(self, "_param_device", None) is not device:
            for p in self.parameters(recurse=True):
                if p is self.values:
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

        # Indices and scores
        scores, indices = self._get_indices(query)  # (bs*heads, knn)

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

        # Weighted aggregation via embedding_bag
        # embedding_bag backward with per_sample_weights is not implemented for bf16 on CUDA.
        # Perform the op in float32 and cast back to the model dtype afterwards.
        out_fp32 = F.embedding_bag(
            indices,
            self.values.float(),
            per_sample_weights=weights.float(),
            mode="sum",
        )
        out = out_fp32.to(dtype)

        if self.v_proj and not self.swilu_proj:
            out = self.value_proj(out)
        if self.swilu_proj:
            out = self.value_proj(out * F.silu(self.swilu_projection(x_flat)))

        out = out.view(B, T, -1)
        if self.gating is not None:
            gate = torch.sigmoid(self.gating(x_flat)).view(B, T, 1)
            out = gate * out
        return out

    def _get_indices(self, query: torch.Tensor):
        # query: (bs*heads, k_dim)
        bs = query.shape[0] // self.heads
        query = query.view(bs, self.heads, self.k_dim)
        half = self.k_dim // 2

        keys = self.keys.view(self.heads, 2, self.n_keys, half)
        k1, k2 = keys[:, 0], keys[:, 1]

        q1, q2 = query[..., :half], query[..., half:]
        s1 = torch.einsum("blh,lkh->blk", q1, k1)  # (bs, heads, n_keys)
        s2 = torch.einsum("blh,lkh->blk", q2, k2)

        s1, i1 = s1.topk(self.knn, dim=2, largest=True)
        s2, i2 = s2.topk(self.knn, dim=2, largest=True)

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

    def forward(self, x: torch.Tensor, lang_emb: torch.Tensor | None = None):
        mem_out = self.mem(x, lang_emb=lang_emb)
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


def attach_memory_to_expert(smolvla_model, cfg: MemoryLayerConfig):
    """
    Replace selected expert MLPs with MLPPlusMemory in-place.

    smolvla_model: SmolVLMWithExpertModel
    cfg: MemoryLayerConfig (enabled must be True at the callsite)
    """
    num_layers = smolvla_model.num_expert_layers
    target_layers = _resolve_target_layers(num_layers, cfg.layers)

    print(f"Target EXPERT layers for memory: {target_layers}")
    target_set = set(target_layers)

    lang_dim = _get_lang_dim(cfg)
    if lang_dim > 0:
        print(f"Language-conditioned query projection enabled with lang_dim={lang_dim}")

    # First, unwrap any previously wrapped layers that are not in the target set
    for li in range(num_layers):
        layer = smolvla_model.lm_expert.layers[li]
        if isinstance(getattr(layer, "mlp", None), MLPPlusMemory) and li not in target_set:
            layer.mlp = layer.mlp.mlp

    # Now, wrap exactly the requested target layers
    for li in target_layers:
        layer = smolvla_model.lm_expert.layers[li]
        dim = smolvla_model.expert_hidden_size
        # Avoid double wrapping if already wrapped
        if isinstance(layer.mlp, MLPPlusMemory):
            continue
        base_dtype = next(layer.mlp.parameters()).dtype
        base_device = next(layer.mlp.parameters()).device
        layer.mlp = MLPPlusMemory(layer.mlp, dim=dim, cfg=cfg, lang_dim=lang_dim)
        # Align non-value memory params to base dtype/device; keep values in float32
        for name, p in layer.mlp.mem.named_parameters():
            if name.startswith("values"):
                p.data = p.data.to(device=base_device, dtype=torch.float32)
            else:
                p.data = p.data.to(device=base_device, dtype=base_dtype)
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
            if name.startswith("values"):
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


