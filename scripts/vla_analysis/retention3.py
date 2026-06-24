#!/usr/bin/env python3
import json, os
BASE="/home/josh/lerobot/outputs/train"
RUNS={
 "control(50ep)": f"{BASE}/libero_10_sequential_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.01_sep_0.25_loc_0.25_sup_128_2048_knn_36_40k_top_t_1536",
 "sep5(50ep)":    f"{BASE}/libero_10_sequential_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.05_sep_5.0_noloc_knn_36_rq512_40k_top_t_1536",
 "protect_b4(20ep)": f"{BASE}/libero_10_sequential_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.05_sep_5.0_noloc_knn_36_rq512_40k_top_t_1536_protect_beta4",
}
ENV={0:4,1:6,2:9,3:2,4:7,5:0,6:8,7:1,8:3,9:5}
ORDER=list(range(10)); STEPS=[3000*(k+1) for k in range(10)]
LBL={4:"two mugs",6:"mug+pud",9:"mug+micro",2:"stove+moka",7:"soup+cheese",0:"soup+sauce",8:"both mokas",1:"cheese+butter",3:"bowl+drawer",5:"book"}
def load(run):
    M={}
    for line in open(os.path.join(run,"eval/results.jsonl")):
        d=json.loads(line); M[d["step"]]=d
    return M
data={n:load(r) for n,r in RUNS.items()}
# per-task init/peak/final per run
print(f"{'ord/env':>8} {'task':<13}", end="")
for n in RUNS: print(f"| {n:>17}", end="")
print()
print(f"{'':>8} {'(init>peak>final ret%)':<13}", end=""); [print(f"| {'':>17}",end="") for n in RUNS]; print()
agg={n:{"init":[],"peak":[],"final":[]} for n in RUNS}
for ord_k,t in enumerate(ORDER):
    env=ENV[t]; st0=STEPS[ord_k]
    print(f"t{ord_k} e{env:<4} {LBL[env]:<13}", end="")
    for n in RUNS:
        M=data[n]; traj=[M[st].get(f"task_{env}") for st in STEPS if st>=st0 and f"task_{env}" in M.get(st,{})]
        if traj:
            i,p,f=traj[0],max(traj),traj[-1]; r=100*f/i if i>0 else 0
            agg[n]["init"].append(i); agg[n]["peak"].append(p); agg[n]["final"].append(f)
            print(f"| {i:3.0f}>{p:3.0f}>{f:3.0f} {r:4.0f}%", end="")
        else: print(f"| {'--':>17}", end="")
    print()
print("-"*80)
print(f"{'MEAN':>8} {'':<13}", end="")
for n in RUNS:
    a=agg[n]; mi=sum(a['init'])/len(a['init']); mp=sum(a['peak'])/len(a['peak']); mf=sum(a['final'])/len(a['final'])
    print(f"| init{mi:4.1f} fin{mf:4.1f}", end="")
print()
for n in RUNS:
    a=agg[n]; mf=sum(a['final'])/len(a['final'])
    print(f"  {n:>18}: FINAL AVG = {mf:.1f}")
