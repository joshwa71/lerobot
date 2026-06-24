#!/usr/bin/env python3
"""Slot-usage deep dive for the libero_10 (LIBERO-Long) sequential runs.
Computes per-task capacity (effnum=1/sum(p^2), core50), read-IoU (validates logged),
read-through-overwrite, and the pairwise overwrite matrix. One run per invocation.
"""
import sys, os, json, gc
import numpy as np

LAYERS = [8,10,12,14]
PREFIX = "model.paligemma_with_expert.gemma_expert.model.layers."
# libero_10: file index == dataset task_index == train order; env map below
ENV = {0:4,1:6,2:9,3:2,4:7,5:0,6:8,7:1,8:3,9:5}
ORDER = list(range(10))
ORD = {t:i for i,t in enumerate(ORDER)}
NK = 384  # n_keys; slot = i1*384 + i2

def load_task(path):
    d = json.load(open(path))
    pm = d["per_module"]
    out = {}
    for L in LAYERS:
        node = pm[PREFIX+str(L)]
        slots = node[next(iter(node))]
        r_ids=[]; r_acc=[]; u_ids=[]
        for sk,st in slots.items():
            sid=int(sk.rsplit("_",1)[1])
            acc=st["total_accesses"]; upd=st["total_updates"]
            if acc: r_ids.append(sid); r_acc.append(acc)
            if upd: u_ids.append(sid)
        out[L]=(np.array(r_ids,dtype=np.int64), np.array(r_acc,dtype=np.float64),
                np.array(sorted(u_ids),dtype=np.int64))
    del d; gc.collect()
    return out

def effnum(acc):
    if acc.size==0: return 0.0
    p=acc/acc.sum(); return float(1.0/np.square(p).sum())
def core_frac(acc, frac=0.5):
    if acc.size==0: return 0
    s=np.sort(acc)[::-1]; c=np.cumsum(s); tot=c[-1]
    return int(np.searchsorted(c, frac*tot)+1)
def subkey_decomp(ids, acc):
    """effective subkeys/half and joint/product ratio at a layer (mass-weighted)."""
    if ids.size==0: return (0,0,0.0)
    p=acc/acc.sum()
    i1=ids//NK; i2=ids%NK
    p1=np.bincount(i1,weights=p,minlength=NK); p2=np.bincount(i2,weights=p,minlength=NK)
    e1=1.0/np.square(p1).sum(); e2=1.0/np.square(p2).sum()
    ej=1.0/np.square(p).sum()
    jp=ej/(e1*e2) if e1*e2>0 else 0.0
    return (float(e1),float(e2),float(jp))

def overwrite_frac(reads_ids, reads_acc, upd_ids_sorted):
    """frac of read weight on updated slots."""
    tot=reads_acc.sum()
    if tot==0 or upd_ids_sorted.size==0: return 0.0
    mask=np.isin(reads_ids, upd_ids_sorted, assume_unique=False)
    return float(reads_acc[mask].sum()/tot)

def wjacc(a_ids,a_acc,b_ids,b_acc):
    # weighted jaccard: sum min / sum max over union (using access counts)
    amap=dict(zip(a_ids.tolist(),a_acc.tolist()))
    bmap=dict(zip(b_ids.tolist(),b_acc.tolist()))
    keys=set(amap)|set(bmap)
    inter=0.0; union=0.0
    for k in keys:
        x=amap.get(k,0.0); y=bmap.get(k,0.0)
        if x<y: inter+=x; union+=y
        else: inter+=y; union+=x
    return inter/union if union else 0.0

def main(run_dir, name):
    mbt=os.path.join(run_dir,"memory_by_task")
    data={t:load_task(os.path.join(mbt,f"memory_usage_task_{t}.json")) for t in ORDER}
    print("="*100); print(f"RUN: {name}"); print("="*100)

    print("\n[CAPACITY] per-task L14 read footprint / effnum / core50  +  4-layer sums")
    print(f"  {'ord':>3} {'env':>3} | {'L14_slots':>9} {'L14_eff':>8} {'L14_core50':>10} {'L14_effK/h':>10} {'L14_j/p':>7} | {'4L_readslots':>12} {'4L_updslots':>11}")
    cap={}
    for t in ORDER:
        rids,racc,_=data[t][14]
        e=effnum(racc); c=core_frac(racc); (k1,k2,jp)=subkey_decomp(rids,racc)
        rs4=sum(data[t][L][0].size for L in LAYERS)
        us4=sum(data[t][L][2].size for L in LAYERS)
        cap[t]=(e,c)
        print(f"  {ORD[t]:>3} {ENV[t]:>3} | {rids.size:>9} {e:>8.0f} {c:>10} {0.5*(k1+k2):>10.0f} {jp:>7.2f} | {rs4:>12} {us4:>11}")
    eL=np.mean([cap[t][0] for t in ORDER]); cL=np.mean([cap[t][1] for t in ORDER])
    print(f"  MEAN L14 effnum={eL:.0f}  core50={cL:.0f}")

    print("\n[READ-THROUGH-OVERWRITE] frac of task READ weight on slots updated by LATER tasks")
    print(f"  {'ord':>3} {'env':>3} | {'L8':>6} {'L10':>6} {'L12':>6} {'L14':>6} | {'mean4':>6}")
    rtos=[]
    for t in ORDER:
        later=[u for u in ORDER if ORD[u]>ORD[t]]
        pl=[]
        for L in LAYERS:
            rids,racc,_=data[t][L]
            upd=np.array(sorted(set().union(*[set(data[u][L][2].tolist()) for u in later])),dtype=np.int64) if later else np.array([],dtype=np.int64)
            pl.append(overwrite_frac(rids,racc,upd))
        m=float(np.mean(pl)); rtos.append((t,m))
        print(f"  {ORD[t]:>3} {ENV[t]:>3} | "+" ".join(f"{x:6.1%}" for x in pl)+f" | {m:6.1%}")
    print(f"  MEAN over tasks (excl last): {np.mean([m for t,m in rtos[:-1]]):.1%}")

    print("\n[PAIRWISE OVERWRITE] M[X<-Y] = frac of X(row) read weight on Y(col later) updates, mean over 4 layers")
    hdr="          "+"".join(f"e{ENV[u]:<4}" for u in ORDER)
    print(hdr)
    big=[]
    for t in ORDER:
        row=""
        for u in ORDER:
            if ORD[u]<=ORD[t]: row+="    ."; continue
            vals=[]
            for L in LAYERS:
                rids,racc,_=data[t][L]
                vals.append(overwrite_frac(rids,racc,data[u][L][2]))
            mv=float(np.mean(vals)); row+=f"{mv:5.0%}"
            if mv>=0.12: big.append((ENV[t],ENV[u],mv,ORD[t],ORD[u]))
        print(f"  e{ENV[t]:<2}o{ORD[t]:<2} {row}")
    print("\n  Biggest channels (>=12%, X<-Y, Y later):")
    for x,y,v,ox,oy in sorted(big,key=lambda z:-z[2]):
        print(f"    env{x} (t{ox}) <- env{y} (t{oy}):  {v:.1%}")

    # basket family focus: env7(t4),env0(t5),env1(t7)
    print("\n[BASKET FAMILY] L14 pairwise weighted-read IoU + overwrite (t4=env7 soup+cheese, t5=env0 soup+sauce, t7=env1 cheese+butter)")
    fam=[(4,7),(5,0),(7,1)]  # (ord,env)
    for (oi,ei) in fam:
        for (oj,ej) in fam:
            if oj<=oi: continue
            ti=ORDER[oi]; tj=ORDER[oj]
            iou=wjacc(*data[ti][14][:2],*data[tj][14][:2])
            # overwrite in both directions among the pair (later updates earlier)
            ow=overwrite_frac(*data[ti][14][:2], data[tj][14][2])
            print(f"    L14 IoU env{ei}(t{oi}) ~ env{ej}(t{oj}) = {iou:.3f} ; overwrite env{ei}<-env{ej} = {ow:.1%}")

    # validate logged read IoU (mean pairwise weighted, 4-layer)
    jac=[]
    for i,t in enumerate(ORDER):
        for u in ORDER[i+1:]:
            jac.append(np.mean([wjacc(*data[t][L][:2],*data[u][L][:2]) for L in LAYERS]))
    print(f"\n[VALIDATE] mean pairwise weighted-read IoU (4-layer) = {np.mean(jac):.4f}  (compare logged memory_iou/all_modules_mean)")
    # binary write-set IoU
    wj=[]
    for i,t in enumerate(ORDER):
        for u in ORDER[i+1:]:
            ll=[]
            for L in LAYERS:
                A=set(data[t][L][2].tolist()); B=set(data[u][L][2].tolist())
                ll.append(len(A&B)/len(A|B) if (A|B) else 0.0)
            wj.append(np.mean(ll))
    print(f"[VALIDATE] mean pairwise write-set binary IoU (4-layer) = {np.mean(wj):.4f}")
    del data; gc.collect()

if __name__=="__main__":
    runs={
      "sep5": "/home/josh/lerobot/outputs/train/libero_10_sequential_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.05_sep_5.0_noloc_knn_36_rq512_40k_top_t_1536",
      "ctrl": "/home/josh/lerobot/outputs/train/libero_10_sequential_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.01_sep_0.25_loc_0.25_sup_128_2048_knn_36_40k_top_t_1536",
      "protect": "/home/josh/lerobot/outputs/train/libero_10_sequential_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.05_sep_5.0_noloc_knn_36_rq512_40k_top_t_1536_protect_beta4",
    }
    which=sys.argv[1] if len(sys.argv)>1 else "sep5"
    main(runs[which], which)
