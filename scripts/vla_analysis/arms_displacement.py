#!/usr/bin/env python3
"""Per-block value displacement + e6-perceived field change, per arm, L14 (+L8 for bleed).
||delta theta_s|| over slots updated in each block; and mass-weighted field change seen by
e6's read distribution across e9's block (t2)."""
import os, sys, json, gc
import numpy as np, torch
from safetensors import safe_open
sys.path.insert(0,"/home/josh/lerobot/scripts/vla_analysis")
from slots import PREFIX
BASE="/home/josh/lerobot/outputs/train"
A_B=f"{BASE}/libero_90_pi05_8_10_12_14_frozenroute_rwarmupB_values10k_c0.05_sep5.0_noloc_rq512/checkpoints/last/pretrained_model"
A_AFF=f"{BASE}/libero_90_pi05_8_10_12_14_frozenroute_affine_nogate_values10k_c0.05_sep5.0_noloc_rq512/checkpoints/last/pretrained_model"
RUNS={

 "affine": (f"{BASE}/libero_10_sequential_pi05_8_10_12_14_frozenroute_affine_nogate_c0.05_sep5.0_noloc_rq512_top_t_1536_protect_beta4_steps5k_tasks5", A_AFF, ["005000","010000","015000","020000","025000"], True),
 "lr2x":   (f"{BASE}/libero_10_sequential_pi05_8_10_12_14_frozenroute_rwarmupB_c0.05_sep5.0_noloc_rq512_top_t_1536_protect_beta4_lr2x_steps5k_tasks5", A_B, ["005000","010000","015000","020000","025000"], False),
 "steps7k":(f"{BASE}/libero_10_sequential_pi05_8_10_12_14_frozenroute_rwarmupB_c0.05_sep5.0_noloc_rq512_top_t_1536_protect_beta4_steps7k_tasks5", A_B, ["007000","014000","021000","028000","035000"], False),
}
ENV={0:4,1:6,2:9,3:2,4:7}
L=14
def key(L,t): return f"model.paligemma_with_expert.gemma_expert.model.layers.{L}.mlp.mem.{t}"
def load_slot(path, L, bias=False):
    with safe_open(os.path.join(path,"model.safetensors"), framework="pt") as f:
        d=f.get_tensor(key(L,"slot_down")).float().reshape(147456,-1)
        u=f.get_tensor(key(L,"slot_up")).float().reshape(147456,-1)
        parts=[d,u]
        if bias:
            try: parts.append(f.get_tensor(key(L,"slot_bias")).float())
            except Exception: pass
    return torch.cat(parts,dim=1)  # (147456, D)
def read_dist(run_dir, t, L):
    d=json.load(open(os.path.join(run_dir,"memory_by_task",f"memory_usage_task_{t}.json")))
    node=d["per_module"][PREFIX+str(L)]; slots=node[next(iter(node))]
    ids=[];acc=[]
    for sk,st in slots.items():
        if st["total_accesses"]: ids.append(int(sk.rsplit("_",1)[1])); acc.append(st["total_accesses"])
    del d; gc.collect()
    ids=np.array(ids); acc=np.array(acc,dtype=np.float64)
    w=np.zeros(147456); w[ids]=acc/acc.sum(); return torch.tensor(w,dtype=torch.float32)
def upd_ids(run_dir,t,L):
    d=json.load(open(os.path.join(run_dir,"memory_by_task",f"memory_usage_task_{t}.json")))
    node=d["per_module"][PREFIX+str(L)]; slots=node[next(iter(node))]
    out=[int(sk.rsplit("_",1)[1]) for sk,st in slots.items() if st["total_updates"]]
    del d; gc.collect(); return np.array(sorted(out))

for name,(rd,a_ckpt,steps,bias) in RUNS.items():
    print("="*100); print(f"RUN: {name}  (L{L})"); print("="*100)
    paths=[a_ckpt]+[os.path.join(rd,"checkpoints",s,"pretrained_model") for s in steps]
    prev=load_slot(paths[0],L,bias)
    w6=read_dist(rd,1,L); w4=read_dist(rd,0,L)
    base_norm=prev.norm(dim=1)  # per-slot norm before each block (updated below)
    for b in range(5):
        cur=load_slot(paths[b+1],L,bias)
        delta=(cur-prev).norm(dim=1)
        uids=upd_ids(rd,b,L)
        du=delta[uids]
        rel=du/ (base_norm[uids]+1e-8)
        # own-block displacement stats over updated slots
        print(f"  block t{b} (e{ENV[b]}): upd={len(uids):6d}  ||d|| mean={du.mean():.4f} p50={du.median():.4f} p90={du.kthvalue(int(0.9*len(uids))).values:.4f} max={du.max():.3f}   rel mean={rel.mean():.3f}")
        if b==2:  # e9's block: field change perceived by e6 and e4
            for wname,w in [("e6",w6),("e4",w4)]:
                num=(w*delta).sum(); den=(w*base_norm).sum()
                # also core-restricted: top slots carrying 50% of w
                order=torch.argsort(w,descending=True); cw=torch.cumsum(w[order],0)
                core=order[:int(torch.searchsorted(cw,0.5))+1]
                numc=(w[core]*delta[core]).sum(); denc=(w[core]*base_norm[core]).sum()
                print(f"      e9-block field change seen by {wname}: full={100*num/den:.2f}%  core50={100*numc/denc:.2f}%")
        prev=cur; base_norm=prev.norm(dim=1)
    del prev,cur; gc.collect()
