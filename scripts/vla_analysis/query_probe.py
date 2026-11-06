#!/usr/bin/env python3
"""Language-anchor routing probe (no model forward).
For each memory layer, compute each task's FiLM language anchor beta = film_mlp(mpnet(instr)),
route the scene-zeroed query q=beta (proj(x)=0 -> q=beta) through the FROZEN product keys, and
measure whether the basket family (task_index 4/5/7 = env7/env0/env1) routes to the SAME slots
*more than background pairs*. If language-only basket IoU >> background, the additive language
bias is the collapsing force -> upweighting proj(x)/down-weighting beta should separate them.
"""
import os, re, glob
import numpy as np, torch
import pandas as pd
from safetensors import safe_open

CKPT="/home/josh/lerobot/outputs/train/libero_90_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.05_sep_5.0_noloc_knn_36_rq512_40k/checkpoints/last/pretrained_model/model.safetensors"
LAYERS=[8,10,12,14]; HEADS=4; KDIM=512; NK=384; KNN=36; HALF=KDIM//2
ENV={0:4,1:6,2:9,3:2,4:7,5:0,6:8,7:1,8:3,9:5}
BASKET=[4,5,7]  # task_index of env7/env0/env1
LBL={4:"two mugs",6:"mug+pud",9:"mug+micro",2:"stove+moka",7:"soup+cheese",0:"soup+sauce",8:"both mokas",1:"cheese+butter",3:"bowl+drawer",5:"book"}

# 1) instructions
tp=pd.read_parquet("/home/josh/lerobot/outputs/libero_10/meta/tasks.parquet")
print("tasks.parquet cols:", list(tp.columns), "| shape", tp.shape)
# tasks.parquet: instruction string is the INDEX, 'task_index' is the int id column
idx2instr={}
for s,row in tp.iterrows():
    ti=int(row["task_index"])
    if ti not in idx2instr: idx2instr[ti]=str(s)
instrs=[idx2instr[i] for i in range(10)]
for i in range(10): print(f"  task{i} (env{ENV[i]}): {instrs[i]}")

# 2) mpnet embeddings (faithful: SentenceTransformer.encode, no normalize)
from sentence_transformers import SentenceTransformer
enc=SentenceTransformer("all-mpnet-base-v2", device="cpu")
lang=enc.encode(instrs, convert_to_tensor=True, show_progress_bar=False).float()  # (10,768)
print("lang emb:", tuple(lang.shape))

def silu(x): return x*torch.sigmoid(x)

def retrieve(q, k1, k2):
    # q (heads, k_dim) ; k1,k2 (heads, n_keys, half) -> 144 slot ids
    q=q.view(1,HEADS,KDIM); q1=q[...,:HALF]; q2=q[...,HALF:]
    s1=torch.einsum("blh,lkh->blk", q1, k1); s2=torch.einsum("blh,lkh->blk", q2, k2)
    s1t,i1=s1.topk(KNN,dim=2); s2t,i2=s2.topk(KNN,dim=2)
    all_s=(s1t.unsqueeze(3)+s2t.unsqueeze(2)).reshape(1,HEADS,-1)
    all_i=(i1.unsqueeze(3)*NK+i2.unsqueeze(2)).reshape(1,HEADS,-1)
    _,best=torch.topk(all_s,KNN,dim=2); idx=all_i.gather(2,best)
    return set(idx.reshape(-1).tolist())

def iou(a,b): return len(a&b)/len(a|b) if (a|b) else 0.0

f=safe_open(CKPT,framework="pt")
allkeys=list(f.keys())
def find(L,suffix):
    pat=re.compile(rf"layers\.{L}\..*{re.escape(suffix)}$")
    m=[k for k in allkeys if pat.search(k)]
    return m[0] if m else None

per_layer_sets={}; per_layer_beta={}
for L in LAYERS:
    w0=find(L,"query_proj.film_mlp.0.weight"); b0=find(L,"query_proj.film_mlp.0.bias")
    w2=find(L,"query_proj.film_mlp.2.weight"); b2=find(L,"query_proj.film_mlp.2.bias")
    kk=find(L,"mem.keys")
    if not all([w0,b0,w2,b2,kk]):
        print(f"L{L}: MISSING keys -> w0={w0} kk={kk}; sample layer-{L} keys:",
              [k for k in allkeys if f"layers.{L}." in k and "mem" in k][:6]); continue
    W0=f.get_tensor(w0).float(); B0=f.get_tensor(b0).float()
    W2=f.get_tensor(w2).float(); B2=f.get_tensor(b2).float()
    keys=f.get_tensor(kk).float().view(HEADS,2,NK,HALF); k1=keys[:,0]; k2=keys[:,1]
    film=silu(lang@W0.T+B0)@W2.T+B2          # (10, 4096)
    gamma=film[:,:HEADS*KDIM]; beta=film[:,HEADS*KDIM:]   # (10,2048) each
    per_layer_beta[L]=beta
    sets=[retrieve(beta[i].view(HEADS,KDIM),k1,k2) for i in range(10)]
    per_layer_sets[L]=sets
    # magnitudes
    print(f"\nL{L}: ||beta|| mean={beta.norm(dim=1).mean():.2f}  ||gamma|| mean={gamma.norm(dim=1).mean():.2f}  ||1+gamma|| mean={(1+gamma).norm(dim=1).mean():.2f}")

# 3) language-only routing IoU: basket pairs vs background
print("\n"+"="*78); print("LANGUAGE-ONLY routing IoU (q=beta), basket family vs background"); print("="*78)
for L in LAYERS:
    if L not in per_layer_sets: continue
    sets=per_layer_sets[L]; beta=per_layer_beta[L]
    bask=[]; bg=[]
    for i in range(10):
        for j in range(i+1,10):
            v=iou(sets[i],sets[j])
            if i in BASKET and j in BASKET: bask.append((i,j,v))
            else: bg.append(v)
    bm=np.mean([v for _,_,v in bask]); bgm=np.mean(bg)
    print(f"\nL{L}: basket-pair lang-IoU mean={bm:.3f}  | background mean={bgm:.3f}  | ratio={bm/max(bgm,1e-9):.1f}x")
    for i,j,v in bask:
        cs=torch.cosine_similarity(beta[i],beta[j],dim=0).item()
        print(f"    env{ENV[i]}({LBL[ENV[i]]}) ~ env{ENV[j]}({LBL[ENV[j]]}): lang-IoU={v:.3f}  cos(beta)={cs:.3f}")

# 4) beta cosine: basket vs background (query-space, supporting)
print("\n"+"="*78); print("beta cosine (mean over layers): basket vs background"); print("="*78)
bask_cos=[]; bg_cos=[]
for i in range(10):
    for j in range(i+1,10):
        cs=np.mean([torch.cosine_similarity(per_layer_beta[L][i],per_layer_beta[L][j],dim=0).item() for L in per_layer_beta])
        (bask_cos if (i in BASKET and j in BASKET) else bg_cos).append(cs)
print(f"  basket-pair cos(beta) mean = {np.mean(bask_cos):.3f}")
print(f"  background  cos(beta) mean = {np.mean(bg_cos):.3f}")
print(f"\nACTUAL full-query routing (from sequential JSONs, weighted L14): basket ~0.25 vs background ~0.05")
