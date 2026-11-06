@lerobot/projects/research_log.md
@lerobot/projects/realworld_research_log.md

Read these above files in full. They provide full context on the project, research trajectory, and results and findings for both sim training and realworld training.
Today we are focused on sim training, though some realworld findings may be useful.

Last time we ran this script:
/home/josh/lerobot/job_scripts/nebius/libero_90/combined/pi05_libero_10_4_layer_film_lora2_knn36_40k_c0.01_topt1536.sh

Which produced these outputs:
/home/josh/lerobot/outputs/train/libero_90_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.01_sep_0.25_loc_0.25_sup_128_2048_knn_36_40k

/home/josh/lerobot/outputs/train/libero_10_sequential_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.01_sep_0.25_loc_0.25_sup_128_2048_knn_36_40k_top_t_1536

Put simply, what we did was swap the previous use of the suite datasets for the use of the libero 90 datasets.

You can read the evolution of a given task's performace over a run by reading the raw logs (not metrics) for sequential runs. The relevant lines look like this:                                                        
                                                                                                                                                                                                         17:39                                                                                                                                                                                                    INFO 2026-03-31 01:17:39 l_train.py:1635 Checkpoint policy after task 4 | step 12000                                                                                                                                    
2026-03-31 01:18:28                                                                                                                                                                                                     
INFO 2026-03-31 01:18:28 l_train.py:1672 Evaluate on env tasks: [8, 1, 3, 5]                                                                                                                                            
2026-03-31 01:18:28                                                                                                                                                                                                     
Stepping through eval batches: 100%|██████████| 50/50 [12:05<00:00, 14.52s/it, running_success_rate=10.0%]                                                                                                              
2026-03-31 01:18:32                                                                                                                                                                                                     
Stepping through eval batches: 100%|██████████| 50/50 [11:51<00:00, 14.23s/it, running_success_rate=46.0%]                                                                                                              
2026-03-31 01:30:36                                                                                                                                                                                                     
Stepping through eval batches: 100%|██████████| 50/50 [08:55<00:00, 10.72s/it, running_success_rate=78.0%]                                                                                                              
2026-03-31 01:42:27                                                                                                                                                                                                     
Stepping through eval batches: 100%|██████████| 50/50 [09:20<00:00, 11.21s/it, running_success_rate=68.0%]                                                                                                              
                                                                                                                                                                                                                        
By finding these blocks in the logs you can see how the model does on a task when it has just been trained, then how it performs later when other tasks have also been trained.

Analyse the memory slot jsons and the wandb logs using this utility:
@scripts/parse_wandb.py

Your task is as follows:
0. As mentioned, read the docs and the run scripts so you understand the context and the configs we used in the last run.
1. Review the wandb logs and metrics, and the memory slot usage files. This should be the bulk of your effort.
2. Identify the failure modes of this run and explain them to me.
3. Given the failure modes, identify which knobs or changes we need to make to achieve strong performance across all 10 sequential tasks without forgetting.
Remember to be scientific here, meaning you should be thorough, identify concrete issues, and propose concrete solutions. Note that pretraining and sequential runs are expensive in terms of compute so you should explain the priority of the next steps.