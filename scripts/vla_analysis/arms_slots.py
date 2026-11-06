#!/usr/bin/env python3
import sys, os, json, gc
import numpy as np
sys.path.insert(0,"/home/josh/lerobot/scripts/vla_analysis")
from slots import load_task, effnum, core_frac, overwrite_frac, LAYERS, PREFIX

BASE="/home/josh/lerobot/outputs/train"
RUNS={
 "stageB":  f"{BASE}/libero_10_sequential_pi05_8_10_12_14_frozenroute_rwarmupB_c0.05_sep5.0_noloc_rq512_top_t_1536_protect_beta4_steps5k_tasks5",
 "affine":  f"{BASE}/libero_10_sequential_pi05_8_10_12_14_frozenroute_affine_nogate_c0.05_sep5.0_noloc_rq512_top_t_1536_protect_beta4_steps5k_tasks5",
 "lr2x":    f"{BASE}/libero_10_sequential_pi05_8_10_12_14_frozenroute_rwarmupB_c0.05_sep5.0_noloc_rq512_top_t_1536_protect_beta4_lr2x_steps5k_tasks5",
 "steps7k": f"{BASE}/libero_10_sequential_pi05_8_10_12_14_frozenroute_rwarmupB_c0.05_sep5.0_noloc_rq512_top_t_1536_protect_beta4_steps7k_tasks5",
}
ENV={0:4,1:6,2:9,3:2,4:7}
ORDER=[0,1,2,3,4]

def load_task_with_updates(path):
    """like slots.load_task but also keeps update counts per slot."""
    d=json.load(open(path)); pm=d["per_module"]; out={}
    for L in LAYERS:
        node=pm[PREFIX+str(L)]; slots=node[next(iter(node))]
        r_ids=[];r_acc=[];u_ids=[];u_cnt=[]
        for sk,st in slots.items():
            sid=int(sk.rsplit("_",1)[1]); acc=st["total_accesses"]; upd=st["total_updates"]
            if acc: r_ids.append(sid); r_acc.append(acc)
            if upd: u_ids.append(sid); u_cnt.append(upd)
        o=np.argsort(u_ids)
        out[L]=(np.array(r_ids,dtype=np.int64),np.array(r_acc,dtype=np.float64),
                np.array(u_ids,dtype=np.int64)[o] if u_ids else np.array([],dtype=np.int64),
                np.array(u_cnt,dtype=np.float64)[o] if u_ids else np.array([],dtype=np.float64))
    del d; gc.collect(); return out

for name,rd in RUNS.items():
    mbt=os.path.join(rd,"memory_by_task")
    data={t:load_task_with_updates(os.path.join(mbt,f"memory_usage_task_{t}.json")) for t in ORDER}
    print("="*110); print(f"RUN: {name}"); print("="*110)
    print(f"  {'t':>2} {'env':>3} | L14: {'readslots':>9} {'effnum':>7} {'core50':>7} | {'updslots':>8} {'ev/slot p50':>11} {'p90':>7} {'max':>8} | {'4L upd':>8}")
    for t in ORDER:
        r_ids,r_acc,u_ids,u_cnt=data[t][14]
        upd4=sum(data[t][L][2].size for L in LAYERS)
        p50=np.percentile(u_cnt,50) if u_cnt.size else 0; p90=np.percentile(u_cnt,90) if u_cnt.size else 0
        print(f"  {t:>2} {ENV[t]:>3} |      {r_ids.size:>9} {effnum(r_acc):>7.0f} {core_frac(r_acc):>7.0f} | {u_ids.size:>8} {p50:>11.0f} {p90:>7.0f} {u_cnt.max() if u_cnt.size else 0:>8.0f} | {upd4:>8}")
    # RTO: task t's read mass on slots updated by LATER tasks (4-layer mean)
    print(f"\n  RTO (read-thru-overwrite by later tasks, 4L mean) + still-mine (self-adapted & not-later-hit, L14)")
    for t in ORDER[:-1]:
        rt=[]
        for L in LAYERS:
            r_ids,r_acc,_,_=data[t][L]
            later=np.unique(np.concatenate([data[u][L][2] for u in ORDER if u>t]))
            rt.append(overwrite_frac(r_ids,r_acc,later))
        # still-mine at L14
        r_ids,r_acc,u_ids,_=data[t][14]
        later14=np.unique(np.concatenate([data[u][14][2] for u in ORDER if u>t]))
        mine=np.isin(r_ids,u_ids)&~np.isin(r_ids,later14)
        sm=float(r_acc[mine].sum()/r_acc.sum())
        selfad=float(r_acc[np.isin(r_ids,u_ids)].sum()/r_acc.sum())
        print(f"    t{t} e{ENV[t]}: RTO={np.mean(rt)*100:5.1f}%  (per-L {' '.join(f'{x*100:.0f}' for x in rt)})   self-adapted(L14)={selfad*100:.0f}%  still-mine(L14)={sm*100:.0f}%")
    # The bleed channels: e6(t1) and e4(t0) read mass on e9(t2)'s updated slots, per layer
    print(f"\n  BLEED channels (read mass of X on slots task-2/e9 updated), per layer:")
    for vt,lab in [(1,"e6"),(0,"e4")]:
        fr=[overwrite_frac(data[vt][L][0],data[vt][L][1],data[2][L][2]) for L in LAYERS]
        print(f"    {lab}<-e9: {' '.join(f'L{L}:{x*100:4.1f}%' for L,x in zip(LAYERS,fr))}")
    for vt,lab in [(1,"e6"),(0,"e4")]:
        fr=[overwrite_frac(data[vt][L][0],data[vt][L][1],data[3][L][2]) for L in LAYERS]
        print(f"    {lab}<-e2: {' '.join(f'L{L}:{x*100:4.1f}%' for L,x in zip(LAYERS,fr))}")
    print()
