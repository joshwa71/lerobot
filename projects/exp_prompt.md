Hi.

Project docs: @projects/vla-memory.md

Experiment history and conclusions: @projects/research_log.md

Make sure to read these document in full, every line. It traces the research decisions, expirements and results.

Make special attention to the most recent entires as they are the most informative about the current experiment set.

I have downloaded the most recent set of experiments (routing balance loss weight sweep) to here: /home/josh/phddev/lerobot/outputs/15_4_26/

You can read the evolution of a given task's performace over a run by reading the raw logs (not metrics) for sequential runs. The relevant lines look like this:

17:39
INFO 2026-03-31 01:17:39 l_train.py:1635 Checkpoint policy after task 4 | step 12000
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

Note the current best config has been saved to baseline_pretrain and baseline_sequential in the above folder.
Analyse the memory slot jsons and the wandb logs using this utility:

@scripts/parse_wandb.py

Your task is primarily to help me understand the findings and results of these experiments in the context of the research thus far, then help me reason about a good next set of experiments to run.
Some pointer questions:
1. How is the pretrain and sequential train memory usage for the runs compared to baseline?
2. How is perf compared to baseline?
3. What is driving perf compared to baseline in terms of internal dynamics of the model and training?
4. What should we try next?

Please be thorough, read the wandb logs and metrics carefully, as well as the memory slot jsons to understand the internal dynamics of the model and training. Last time we discussed read time overlap between slots and noticed that increased knn seemed to help performance, so we tested higher knn with aligned routing. You can see the full research trajectory from the research log.