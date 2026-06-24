#!/usr/bin/env python3
"""Graded protect-vs-starve curve matching the user's rule:
P(update s) decreases with usefulness(s) = prior task's read weight on s.
For collision (prior P, writer W): protect P's top-X% read-mass core.
  saved(X) = P read-mass on (P_core_X ∩ W_updates) / P_total   (env7 damage we prevent)
  cost(X)  = W read-mass on  P_core_X / W_total                 (env0 adaptation we block)
If saved rises faster than cost at small X -> headroom. If saved~=cost -> zero-sum."""
import sys, os, json, gc
import numpy as np
LAYERS=[8,10,12,14]; PREFIX="model.paligemma_with_expert.gemma_expert.model.layers."
ORDER=list(range(10))
def load_task(path):
    d=json.load(open(path)); pm=d["per_module"]; out={}
    for L in LAYERS:
        node=pm[PREFIX+str(L)]; slots=node[next(iter(node))]
        rid=[];rac=[];uid=[]
        for sk,st in slots.items():
            sid=int(sk.rsplit("_",1)[1]); a=st["total_accesses"]; u=st["total_updates"]
            if a: rid.append(sid); rac.append(a)
            if u: uid.append(sid)
        out[L]=(np.array(rid),np.array(rac,dtype=np.float64),np.array(sorted(uid)))
    del d; gc.collect(); return out
def curve(P,W,data,L,Xs=(10,25,50,75,90)):
    pid,pac,_=data[P][L]; wid,wac,_=data[W][L]; _,_,wu=data[W][L]
    ptot=pac.sum(); wtot=wac.sum(); wuset=set(wu.tolist())
    wmap=dict(zip(wid.tolist(),wac.tolist()))
    order=np.argsort(pac)[::-1]; pid_s=pid[order]; pac_s=pac[order]; cum=np.cumsum(pac_s)
    rows=[]
    for X in Xs:
        ncore=int(np.searchsorted(cum, X/100.0*ptot)+1)
        core=pid_s[:ncore]; coreset=set(core.tolist())
        saved=sum(a for s,a in zip(pid_s[:ncore].tolist(),pac_s[:ncore].tolist()) if s in wuset)/ptot
        cost =sum(wmap.get(s,0.0) for s in coreset)/wtot
        rows.append((X,ncore,saved,cost))
    return rows
def main(run,name,chans):
    data={t:load_task(os.path.join(run,"memory_by_task",f"memory_usage_task_{t}.json")) for t in ORDER}
    print("="*92); print(f"{name}: graded protect-vs-starve at L14  (protect prior P's top-X% read-mass core)"); print("="*92)
    for (P,W,lbl) in chans:
        print(f"\n  {lbl}   [protect P={P}'s core from writer W={W}]")
        print(f"    {'X%core':>7} {'n_slots':>8} {'saved(P dmg)':>12} {'cost(W block)':>13} {'saved/cost':>11}")
        for X,n,s,c in curve(P,W,data,14):
            r=s/c if c>0 else float('inf')
            print(f"    {X:>6}% {n:>8} {s:>11.1%} {c:>12.1%} {r:>11.2f}")
    del data; gc.collect()
if __name__=="__main__":
    runs={"sep5":"/home/josh/lerobot/outputs/train/libero_10_sequential_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.05_sep_5.0_noloc_knn_36_rq512_40k_top_t_1536",
          "ctrl":"/home/josh/lerobot/outputs/train/libero_10_sequential_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.01_sep_0.25_loc_0.25_sup_128_2048_knn_36_40k_top_t_1536"}
    # (prior, writer, label): env7=t4, env0=t5, env1=t7, env3=t8, env9=t2, env4=t0
    chans=[(4,5,"env7<-env0  GENUINE (soup shared)"),
           (5,7,"env0<-env1  basket-frame only (env0 SURVIVED)"),
           (2,8,"env9<-env3  close-container shared"),
           (0,4,"env4<-env7  cross-family-ish")]
    main(runs[sys.argv[1]], sys.argv[1], chans)
