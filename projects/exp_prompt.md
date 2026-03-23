Hi.

Project docs: @projects/vla-memory.md

Experiment history and conclusions: @projects/research_log.md

I have downloaded the most recent set of experiments (tfidf_top_t sweep, 4 layers, and lora=4) to here: /home/josh/phddev/
lerobot/outputs/22_3_26/

Note the current best config has been saved to baseline_pretrain and baseline_sequential in the above folder. Full runs can be found at:
/home/josh/phddev/lerobot/outputs/16_3_26/libero_95_10_12_14_film_lora_2_sample_contrastive_1_sep_0.25_loc_0.25_sup_128_2048
/home/josh/phddev/lerobot/outputs/16_3_26/sequential_libero_95_10_12_14_film_lora_2_sample_contrastive_1_sep_0.25_loc_0.25_sup_128_2048

Analyse the memory slot jsons and the wandb logs using this utility:

@scripts/parse_wandb.py

Then help me understand the following:
1. How is the pretrain and sequential train memory usage for the runs compared to baseline?
2. How is perf compared to baseline?
3. What is driving perf compared to baseline in terms of internal dynamics of the model and training?
4. What should we try next?

Please be thorough, read the wandb logs and metrics carefully, as well as the memory slot jsons to understand the internal
dynamics of the model and training.