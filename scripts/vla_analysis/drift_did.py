"""Test 2 (Entry 32): within-run r2-vs-r4 contested-slot value-drift DiD.
Drift of env7's (t4) core slots across env0's (t5) block, at L8 (r2) vs L14 (r4),
in the [2,2,4,4] run vs the all-r2 protectB4 baseline.
d(s) = ||theta_after(s)-theta_before(s)|| / ||theta_before(s)||, theta = concat(down,up).
"""
import sys, os, json, gc
import numpy as np
from safetensors import safe_open
sys.path.insert(0, "/home/josh/lerobot/scripts/vla_analysis")
from slots import load_task

BASE="/home/josh/lerobot/outputs/train"
RUNS={
 "r2244":  (f"{BASE}/libero_10_sequential_pi05_8_10_12_14_film_lora_2244_sample_contrastive_0.05_sep_5.0_noloc_knn_36_rq512_40k_top_t_1536_protect_beta4_steps5k","025000","030000"),
 "protB4": (f"{BASE}/libero_10_sequential_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.05_sep_5.0_noloc_knn_36_rq512_40k_top_t_1536_protect_beta4","015000","018000"),
}
PRE="model.paligemma_with_expert.gemma_expert.model.layers.%d.mlp.mem."
LAYERS=[8,14]

def slot_drift(ckb, cka, layer):
    num=den=None
    for name in ("slot_down","slot_up"):
        with safe_open(ckb, framework="pt") as fb, safe_open(cka, framework="pt") as fa:
            sb=fb.get_slice(PRE%layer+name); sa=fa.get_slice(PRE%layer+name)
            N=sb.get_shape()[0]
            if num is None:
                num=np.zeros(N); den=np.zeros(N)
            for lo in range(0,N,16384):
                hi=min(lo+16384,N)
                tb=sb[lo:hi].reshape(hi-lo,-1).double()
                ta=sa[lo:hi].reshape(hi-lo,-1).double()
                num[lo:hi]+=((ta-tb)**2).sum(1).numpy()
                den[lo:hi]+=(tb**2).sum(1).numpy()
    return np.sqrt(num)/np.sqrt(np.maximum(den,1e-24))

for run,(d,cb,ca) in RUNS.items():
    ckb=f"{d}/checkpoints/{cb}/pretrained_model/model.safetensors"
    cka=f"{d}/checkpoints/{ca}/pretrained_model/model.safetensors"
    mbt=f"{d}/memory_by_task"
    e7=load_task(f"{mbt}/memory_usage_task_4.json")   # env7 reads
    e0=load_task(f"{mbt}/memory_usage_task_5.json")   # env0 updates
    print("="*90); print(f"RUN {run}: drift across env0's block on env7's core slots"); print("="*90)
    rho={}
    for L in LAYERS:
        rids,racc,_=e7[L]
        o=np.argsort(-racc); rids,racc=rids[o],racc[o]
        c=np.cumsum(racc); core=rids[:int(np.searchsorted(c,0.5*c[-1])+1)]
        corew=racc[:len(core)]
        upd=set(e0[L][2].tolist())
        d_all=slot_drift(ckb,cka,L)
        # background: random non-core slots for scale
        rng=np.random.default_rng(0); bg=rng.choice(147456,20000,replace=False)
        coreset=set(core.tolist()); bg=np.array([s for s in bg if s not in coreset])
        m_upd=np.array([s in upd for s in core])
        w=corew/corew.sum()
        wm_all=float((d_all[core]*w).sum())
        wm_upd=float((d_all[core[m_upd]]*(w[m_upd]/w[m_upd].sum())).sum()) if m_upd.any() else float('nan')
        expo=float(w[m_upd].sum())
        med_bg=float(np.median(d_all[bg]))
        rho[L]=(wm_all,wm_upd)
        print(f"  L{L}: env7 core50={len(core)} slots | mass hit by env0={expo:.1%} | "
              f"wmean drift core={wm_all:.4f} | drift core&updated={wm_upd:.4f} | bg median={med_bg:.5f}")
        del d_all; gc.collect()
    print(f"  rho(all core)      = d(L14)/d(L8) = {rho[14][0]/rho[8][0]:.2f}")
    print(f"  rho(core&updated)  = d(L14)/d(L8) = {rho[14][1]/rho[8][1]:.2f}")
    del e7,e0; gc.collect()
