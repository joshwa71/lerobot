#!/usr/bin/env python3
"""Offline first-order simulation of usefulness-gated writes.

Store:   u_before[W](s) = max over prior tasks P<W of  r_P(s)   (peak-normalized read)
Gate:    pi_W(s) = (1 - u_before[W](s))^beta
Saving:  per prior P, read-through-overwrite recomputed with later writers' updates
         attenuated by their gate:  destroyed(s)=max_{W>P, s in U_W} pi_W(s)
         RTO_b(P) = sum_s pread_P(s)*destroyed(s);  saving = RTO_0 - RTO_b
Cost:    per writer W, suppressed write-demand = 1 - sum_s pread_W(s)*pi_W(s)
beta=0 must reproduce the measured read-through (pi=1 => destroyed=1 on any later update).
Static-footprint approximation: footprints fixed; no closed-loop, mass!=success.
"""
import sys, os, json, gc
import numpy as np
LAYERS=[8,10,12,14]; PREFIX="model.paligemma_with_expert.gemma_expert.model.layers."
NSLOT=384*384
ENV={0:4,1:6,2:9,3:2,4:7,5:0,6:8,7:1,8:3,9:5}
LBL={4:"2mugs",6:"mug+pud",9:"mug+micro",2:"stove+moka",7:"soup+cheese",0:"soup+sauce",8:"2mokas",1:"cheese+butter",3:"bowl+drawer",5:"book"}
ORDER=list(range(10))

def build(run):
    P={}  # (t,L)->dict(pread,rnorm,upd)
    for t in ORDER:
        d=json.load(open(os.path.join(run,"memory_by_task",f"memory_usage_task_{t}.json")))
        pm=d["per_module"]
        for L in LAYERS:
            node=pm[PREFIX+str(L)]; slots=node[next(iter(node))]
            acc=np.zeros(NSLOT,dtype=np.float64); upd=np.zeros(NSLOT,dtype=bool)
            for sk,st in slots.items():
                sid=int(sk.rsplit("_",1)[1]); a=st["total_accesses"]; u=st["total_updates"]
                if a: acc[sid]=a
                if u: upd[sid]=True
            tot=acc.sum(); mx=acc.max()
            P[(t,L)]=(acc/tot, acc/mx, upd)   # pread (sum1), rnorm (peak1), updmask
        del d; gc.collect()
    return P

def ubefore(P):
    U={}
    for L in LAYERS:
        cum=np.zeros(NSLOT,dtype=np.float64)
        for t in ORDER:
            U[(t,L)]=cum.copy()
            cum=np.maximum(cum, P[(t,L)][1])  # fold rnorm of task t
    return U

def sweep(P,U,betas):
    out={}  # beta -> (rto_task[10], cost_task[10])
    for b in betas:
        rto=np.zeros((10,len(LAYERS))); cost=np.zeros((10,len(LAYERS)))
        for li,L in enumerate(LAYERS):
            # precompute writer gates
            pis={t:(1.0-U[(t,L)])**b for t in ORDER}
            # suffix max of (updmask*pi) over writers, descending
            suffix=np.zeros(NSLOT)
            for Pr in range(9,-1,-1):
                destroyed=suffix  # writers > Pr
                pread=P[(Pr,L)][0]
                rto[Pr,li]=float((pread*destroyed).sum())
                # fold writer Pr
                suffix=np.maximum(suffix, P[(Pr,L)][2]*pis[Pr])
            for W in ORDER:
                pread=P[(W,L)][0]
                cost[W,li]=1.0-float((pread*pis[W]).sum())
        out[b]=(rto.mean(1), cost.mean(1))
    return out

def main(run,name):
    P=build(run); U=ubefore(P)
    betas=[0,0.5,1,2,4,8,16,32]
    out=sweep(P,U,betas)
    rto0=out[0][0]
    print("="*92); print(f"RUN {name}: offline usefulness-gate beta-sweep (4-layer mean, peak-norm usefulness)"); print("="*92)
    print(f"  baseline mean read-through (P0..P8) @beta=0 = {rto0[:9].mean():.1%}  (sanity vs measured 37.3%/55.4%)")
    print(f"\n  {'beta':>5} | {'mean saving':>11} {'mean cost':>10} {'NET':>7} | {'env7 RTO':>9} {'env0 cost':>9} {'env1 cost':>9}")
    for b in betas:
        rto,cost=out[b]
        saving=(rto0-rto)
        msav=saving[:9].mean(); mcost=cost.mean(); net=msav-mcost
        # env7=task4, env0=task5, env1=task7
        e7=rto[4]; c0=cost[5]; c1=cost[7]
        print(f"  {b:>5} | {msav:>10.1%} {mcost:>9.1%} {net:>+7.1%} | {e7:>8.1%} {c0:>8.1%} {c1:>8.1%}")
    # per-task breakdown at a mid beta (pick best net)
    nets={b: (rto0-out[b][0])[:9].mean()-out[b][1].mean() for b in betas}
    bstar=max(nets,key=nets.get)
    rto,cost=out[bstar]
    print(f"\n  PER-TASK at beta*={bstar} (max net):")
    print(f"  {'ord':>3} {'env':>3} {'task':<14} {'RTO_0':>6} {'RTO_b':>6} {'saved':>6} | {'cost(writer)':>12}")
    for t in ORDER:
        print(f"  {t:>3} {ENV[t]:>3} {LBL[ENV[t]]:<14} {rto0[t]:>6.1%} {rto[t]:>6.1%} {rto0[t]-rto[t]:>6.1%} | {cost[t]:>12.1%}")
    del P,U; gc.collect()

if __name__=="__main__":
    runs={"sep5":"/home/josh/lerobot/outputs/train/libero_10_sequential_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.05_sep_5.0_noloc_knn_36_rq512_40k_top_t_1536",
          "ctrl":"/home/josh/lerobot/outputs/train/libero_10_sequential_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.01_sep_0.25_loc_0.25_sup_128_2048_knn_36_40k_top_t_1536"}
    main(runs[sys.argv[1]], sys.argv[1])
