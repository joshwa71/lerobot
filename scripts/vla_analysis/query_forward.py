"""Forward query-decomposition probe. Invoked from lerobot_sequential_train via QUERY_PROBE env.
Captures, per memory layer, the query components on real libero_10 frames and recomputes routing
under 3 query variants:
  full  = proj(x)*(1+gamma) + beta      (the real query)
  scene = proj(x)*(1+gamma)             (beta stripped -> 'upweight scene' limit)
  lang  = beta                          (proj zeroed   -> language-only)
Decisive test: is basket IoU(scene) << IoU(full)? If yes, beta drives the collision (routing-
fixable by down-weighting language). If IoU(scene) ~= IoU(full), the collision is scene-similarity.
Also reports ||scene|| vs ||beta|| magnitude balance.
"""
import os, re
from collections import defaultdict
import torch

NSLOT = 384 * 384
ENV = {0:4,1:6,2:9,3:2,4:7,5:0,6:8,7:1,8:3,9:5}
LBL = {4:"two mugs",6:"mug+pud",9:"mug+micro",2:"stove+moka",7:"soup+cheese",0:"soup+sauce",8:"both mokas",1:"cheese+butter",3:"bowl+drawer",5:"book"}
BASKET = [4, 5, 7]      # task_index env7/env0/env1
CONTROL = [2, 6]        # background controls env9/env8
VARIANTS = ("full", "scene", "lang")

CUR = {"task": None}
QP2INFO = {}
# task -> layer -> variant -> dense count vector (cpu float64)
STATS = defaultdict(lambda: defaultdict(lambda: {v: torch.zeros(NSLOT, dtype=torch.float64) for v in VARIANTS}))
MAG = defaultdict(lambda: defaultdict(lambda: {"scene": [], "beta": []}))


def _retrieve(q, keys, heads, k_dim, n_keys, knn):
    half = k_dim // 2
    bs = q.shape[0] // heads
    qv = q.view(bs, heads, k_dim)
    kv = keys.view(heads, 2, n_keys, half)
    k1, k2 = kv[:, 0], kv[:, 1]
    q1, q2 = qv[..., :half], qv[..., half:]
    s1 = torch.einsum("blh,lkh->blk", q1, k1)
    s2 = torch.einsum("blh,lkh->blk", q2, k2)
    s1t, i1 = s1.topk(knn, dim=2)
    s2t, i2 = s2.topk(knn, dim=2)
    all_s = (s1t.unsqueeze(3) + s2t.unsqueeze(2)).reshape(bs, heads, -1)
    all_i = (i1.unsqueeze(3) * n_keys + i2.unsqueeze(2)).reshape(bs, heads, -1)
    _, best = torch.topk(all_s, knn, dim=2)
    return all_i.gather(2, best).reshape(-1)


def _wiou(a, b):
    return float(torch.minimum(a, b).sum() / torch.maximum(a, b).sum().clamp(min=1e-9))


def run_query_probe(policy, accelerator, dataset, task_index_to_name, preprocessor, device, cfg):
    from lerobot.policies.modules.memory_lite import QueryMLPLite, HashingMemoryLite
    from lerobot.scripts.lerobot_sequential_train import _build_dataloader_for_task
    unwrapped = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
    unwrapped.eval()
    for name, mod in unwrapped.named_modules():
        if isinstance(mod, HashingMemoryLite):
            m = re.search(r"layers\.(\d+)\.", name)
            QP2INFO[id(mod.query_proj)] = dict(
                keys=mod.keys.detach().float(), heads=mod.heads, k_dim=mod.k_dim,
                n_keys=mod.n_keys, knn=mod.knn, L=int(m.group(1)) if m else -1)
    print(f"[probe] found {len(QP2INFO)} memory modules")

    orig = QueryMLPLite.forward

    @torch.no_grad()
    def patched(self, x, lang_emb=None):
        out = orig(self, x, lang_emb)
        info = QP2INFO.get(id(self)); t = CUR["task"]
        if info is not None and t is not None and lang_emb is not None and self.lang_dim > 0:
            B_T = x.shape[0] if x.dim() == 2 else x.shape[0] * x.shape[1]
            x_flat = x.view(-1, self.input_dim)
            proj = self.proj(x_flat).float()
            B = lang_emb.shape[0]; T = B_T // B
            film = self.film_mlp(lang_emb.to(out.dtype)).float()
            g = film[:, : self.heads * self.k_dim]; b = film[:, self.heads * self.k_dim:]
            g = g.unsqueeze(1).expand(B, T, -1).reshape(B_T, -1)
            b = b.unsqueeze(1).expand(B, T, -1).reshape(B_T, -1)
            scene = proj * (1 + g); full = scene + b; lang = b
            MAG[t][info["L"]]["scene"].append(scene.norm(dim=1).mean().item())
            MAG[t][info["L"]]["beta"].append(b.norm(dim=1).mean().item())
            keys = info["keys"].to(proj.device)
            for var, qv in (("full", full), ("scene", scene), ("lang", lang)):
                idx = _retrieve(qv.reshape(B_T * self.heads, self.k_dim), keys,
                                self.heads, self.k_dim, info["n_keys"], info["knn"])
                cnt = torch.bincount(idx, minlength=NSLOT).double().cpu()
                STATS[t][info["L"]][var] += cnt
        return out

    QueryMLPLite.forward = patched
    nb = int(os.environ.get("QUERY_PROBE_NB", "25"))
    cam_keys = list(dataset.meta.camera_keys)
    for t in BASKET + CONTROL:
        name = task_index_to_name.get(t, "")
        dl = _build_dataloader_for_task(dataset, task_index_to_name, t, batch_size=32,
                                        num_workers=4, device_type=device.type)
        CUR["task"] = t; it = iter(dl); n = 0
        for _ in range(nb):
            try: batch = next(it)
            except StopIteration: break
            for ck in cam_keys:
                if ck in batch and batch[ck].dtype == torch.uint8:
                    batch[ck] = batch[ck].to(torch.float32) / 255.0
            batch = preprocessor(batch)
            B = batch[next(iter(batch))].shape[0]
            te = unwrapped.get_task_embeddings([name] * B)
            if te is not None: te = te.to(device)
            with torch.no_grad(), accelerator.autocast():
                unwrapped.forward(batch, task_emb=te)
            n += 1
        CUR["task"] = None
        print(f"[probe] task{t} env{ENV[t]} {LBL[ENV[t]]}: {n} batches")
    QueryMLPLite.forward = orig

    LAYERS = sorted({i["L"] for i in QP2INFO.values()})
    print("\n" + "=" * 80); print("MAGNITUDE: ||proj(x)*(1+gamma)|| (scene)  vs  ||beta|| (language)"); print("=" * 80)
    for L in LAYERS:
        sc = [v for t in BASKET + CONTROL for v in MAG[t][L]["scene"]]
        be = [v for t in BASKET + CONTROL for v in MAG[t][L]["beta"]]
        sc = sum(sc) / len(sc); be = sum(be) / len(be)
        print(f"  L{L}: ||scene||={sc:7.2f}   ||beta||={be:6.2f}   ratio scene/beta = {sc/be:5.1f}x")

    def pair_iou(tasks, var, L):
        out = []
        for i in range(len(tasks)):
            for j in range(i + 1, len(tasks)):
                a, b = tasks[i], tasks[j]
                out.append((a, b, _wiou(STATS[a][L][var], STATS[b][L][var])))
        return out

    print("\n" + "=" * 80); print("BASKET pairwise weighted routing IoU under each query variant"); print("=" * 80)
    for L in LAYERS:
        print(f"\n  L{L}:")
        for var in VARIANTS:
            ps = pair_iou(BASKET, var, L)
            mean = sum(p[2] for p in ps) / len(ps)
            detail = "  ".join(f"e{ENV[a]}~e{ENV[b]}={v:.3f}" for a, b, v in ps)
            print(f"    {var:5s}: mean={mean:.3f}   [{detail}]")
        # background = basket-vs-control pairs, full query
        bg = [_wiou(STATS[a][L]["full"], STATS[c][L]["full"]) for a in BASKET for c in CONTROL]
        print(f"    background(full, basket~control) mean={sum(bg)/len(bg):.3f}")
    print("\n[probe] DONE")
