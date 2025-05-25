
---
## Overall Implementation Strategy

1.  **Dataset Handling (Done partially):**
    *   We'll need a way to load both standard successful trajectories and paired (failure, success) trajectories.
    *   The `LeRobotDataset` will need to be adapted or wrapped to provide these two types of samples to the policy.

2.  **Failure Classifier ($f_\phi$) (Done):**
    *   This can be a separate model, trained first or jointly. For simplicity, let's assume it's trained first and then used during policy inference.
    *   It will take a trajectory (sequence of obs, states/actions) as input and output a failure probability.

3.  **FCIL Policy (transformer-based, $\pi_\theta$):**
    *   The transformer architecture is a good candidate. We'll need to modify its input processing to accept the conditional information ($\tau_{fail}$, $z_{fail}$) during recovery mode.
    *   `z_{fail}` will be a learnable embedding.
    *   $\tau_{fail}$ will be encoded into a fixed-size representation.

4.  **Training Script:**
    *   A new training script (`train_fcil.py`) will be the cleanest approach.
    *   It will manage loading the two types of data (standard success, paired failure-success).
    *   It will compute the combined loss $\mathcal{L}_{total} = \mathcal{L}_{standard} + \lambda \mathcal{L}_{recovery}$.

5.  **Inference/Evaluation:**
    *   The evaluation script (`eval.py`) will need to incorporate the failure classifier and the policy's mode-switching logic.
