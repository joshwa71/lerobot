#!/usr/bin/env python3
import os, sys
sys.path.insert(0,"/home/josh/lerobot/scripts")
from parse_wandb import WandbRun
BASE="/home/josh/lerobot/outputs/train"
SEP5=f"{BASE}/libero_10_sequential_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.05_sep_5.0_noloc_knn_36_rq512_40k_top_t_1536"
CTL =f"{BASE}/libero_10_sequential_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.01_sep_0.25_loc_0.25_sup_128_2048_knn_36_40k_top_t_1536"
ENV={0:4,1:6,2:9,3:2,4:7,5:0,6:8,7:1,8:3,9:5}
LBL={4:"2mugs",6:"mug+pud",9:"mug+micro",2:"stove+moka",7:"soup+cheese",0:"soup+sauce",8:"2mokas",1:"cheese+butter",3:"bowl+drawer",5:"book"}
for name,run in [("sep5",SEP5),("ctrl",CTL)]:
    r=WandbRun.from_wandb_dir(os.path.join(run,"wandb"))
    pts=r.get_metric("train/mse_loss")
    print(f"\n=== {name}: per-task block MSE (min within block / value at block end) ===")
    print(f"  {'ord':>3} {'env':>3} {'task':<14} {'block_min':>9} {'block_end':>9}")
    for k in range(10):
        lo,hi=3000*k+1,3000*(k+1)
        block=[v for s,v in pts if lo<=s<=hi]
        if not block: continue
        end=[v for s,v in pts if s<=hi][-1]
        print(f"  {k:>3} {ENV[k]:>3} {LBL[ENV[k]]:<14} {min(block):9.4f} {end:9.4f}")
