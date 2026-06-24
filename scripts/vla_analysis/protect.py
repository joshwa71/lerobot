#!/usr/bin/env python3
"""Collision-aware protectability autopsy (extends Entry 19 to sep5).
For each (prior P, later writer W) collision: of the slots where W overwrites P's reads,
how much of P's read-mass do they carry (=damage) vs how much of W's read-mass (=cost to
protect). damage>>cost => incidental, a task-aware IDF/veto protects it cheaply.
damage~=cost (both high) => genuine contention, write-mask is zero-sum (rank/prior territory)."""
import sys, os, json, gc
import numpy as np
LAYERS=[8,10,12,14]
PREFIX="model.paligemma_with_expert.gemma_expert.model.layers."
ENV={0:4,1:6,2:9,3:2,4:7,5:0,6:8,7:1,8:3,9:5}
ORDER=list(range(10)); ORD={t:i for i,t in enumerate(ORDER)}
LBL={4:"2mugs",6:"mug+pud",9:"mug+micro",2:"stove+moka",7:"soup+cheese",0:"soup+sauce",8:"2mokas",1:"cheese+butter",3:"bowl+drawer",5:"book"}

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

def collision(P,W,data,L):
    pr_id,pr_ac,_=data[P][L]; wr_id,wr_ac,_=data[W][L]; _,_,wu=data[W][L]
    pmap=dict(zip(pr_id.tolist(),pr_ac.tolist())); ptot=pr_ac.sum()
    wmap=dict(zip(wr_id.tolist(),wr_ac.tolist())); wtot=wr_ac.sum()
    wuset=set(wu.tolist())
    # hit = P's read slots that W updated
    dmg=0.0; costW=0.0
    for s,a in pmap.items():
        if s in wuset:
            dmg+=a; costW+=wmap.get(s,0.0)
    return dmg/ptot if ptot else 0, costW/wtot if wtot else 0

def main(run_dir,name):
    data={t:load_task(os.path.join(run_dir,"memory_by_task",f"memory_usage_task_{t}.json")) for t in ORDER}
    print("="*100); print(f"RUN: {name}  — collision protectability (L14)"); print("="*100)
    # channels of interest: basket family (genuine) + env-X<-env3 (mixed) + a few
    chans=[(4,5,"env7<-env0 SOUP shared"),(4,7,"env7<-env1 CHEESE shared"),(5,7,"env0<-env1 basket-frame only"),
           (2,8,"env9<-env3 CLOSE shared"),(3,8,"env2<-env3 ?"),(1,8,"env6<-env3 ?"),(6,8,"env8<-env3 ?"),
           (0,5,"env4<-env0 ?"),(2,3,"env9<-env2 ?")]
    print(f"  {'channel':<30} {'L14 dmg%':>9} {'L14 costW%':>10} {'dmg/cost':>9}   {'4L dmg%':>8} {'4L costW%':>9} {'4L ratio':>8}  verdict")
    for (op,ow,lbl) in chans:
        P=ORDER[op]; W=ORDER[ow]
        d14,c14=collision(P,W,data,14)
        # 4-layer
        ds=[];cs=[]
        for L in LAYERS:
            dd,cc=collision(P,W,data,L); ds.append(dd); cs.append(cc)
        d4=np.mean(ds); c4=np.mean(cs)
        r14=d14/c14 if c14>0 else float('inf'); r4=d4/c4 if c4>0 else float('inf')
        verdict = "GENUINE (zero-sum)" if r4<1.5 else ("incidental (protectable)" if r4>2.2 else "mixed")
        print(f"  {lbl:<30} {d14:9.1%} {c14:10.1%} {r14:9.2f}   {d4:8.1%} {c4:9.1%} {r4:8.2f}  {verdict}")
    del data; gc.collect()

if __name__=="__main__":
    runs={
      "sep5":"/home/josh/lerobot/outputs/train/libero_10_sequential_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.05_sep_5.0_noloc_knn_36_rq512_40k_top_t_1536",
      "ctrl":"/home/josh/lerobot/outputs/train/libero_10_sequential_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.01_sep_0.25_loc_0.25_sup_128_2048_knn_36_40k_top_t_1536",
    }
    main(runs[sys.argv[1]], sys.argv[1])
