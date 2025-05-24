
## Project 

### Abstract

We propose Failure-Conditioned Imitation Learning (FCIL), a method that leverages failed demonstrations to improve policy robustness through explicit failure-aware recovery mechanisms. Unlike traditional imitation learning that discards negative samples, FCIL trains policies to recognize and recover from failures by conditioning on previous failed attempts.

### Method Overview

#### Problem Setting

Given a dataset D containing:

- Successful trajectories: $\tau_s = {(s_t, a_t)}_{t=0}^T$ where terminal reward $r_T = 1$
- Failed trajectories: $\tau_f = {(s_t, a_t)}_{t=0}^T$ where terminal reward $r_T = 0$
- Paired trajectories: $(\tau_f, \tau_s)$ representing failure followed by success on the same task

#### Core Approach

We train a policy $\pi_\theta$ with two operational modes:

1. **Standard Mode**: $\pi_\theta(a_t | s_t)$ - predicts actions given current state
2. **Recovery Mode**: $\pi_\theta(a_t | s_t, \tau_{fail}, z_{fail})$ - predicts actions conditioned on previous failure

Where:

- $\tau_{fail}$ is the failed trajectory context
- $z_{fail}$ is a learned failure token indicating recovery mode

#### Training Procedure

**1. Failure Classifier Training** $$f_\phi: \tau \rightarrow {0, 1}$$ Train binary classifier to detect trajectory failures using collected demonstrations.

**2. Policy Training**

For paired data $(\tau_f, \tau_s)$: $$\mathcal{L}_{recovery} = \mathbb{E}_{(s,a) \sim \tau_s} \left[ |a - \pi_\theta(a|s, \tau_f, z_{fail})|^2 \right]$$

For standard successful trajectories: $$\mathcal{L}_{standard} = \mathbb{E}_{(s,a) \sim \tau_s} \left[ |a - \pi_\theta(a|s)|^2 \right]$$

Total loss: $$\mathcal{L}_{total} = \mathcal{L}_{standard} + \lambda \mathcal{L}_{recovery}$$

#### Inference Procedure

```
1. Execute policy: τ = rollout(π_θ(a|s))
2. Detect failure: is_failure = f_φ(τ) > threshold
3. If is_failure:
   - Switch to recovery mode
   - Execute: π_θ(a|s, τ, z_fail)
4. Else: continue with standard mode
```

### Implementation Details

#### Architecture Considerations

- **Trajectory Encoding**: Use causal transformer to encode failed trajectory into fixed-size representation
- **Context Window**: Limit to last N steps to manage computational cost
- **Failure Token**: Learned embedding concatenated to state encoding

#### Data Collection Strategy

1. Collect initial demonstrations with natural failure rate
2. For each failure, immediately collect recovery demonstration
3. Augment with unpaired successful trajectories for standard mode training

#### Key Hyperparameters

- $\lambda$: Weight balancing standard vs recovery loss (typically 0.5-1.0)
- $N$: Context window size for failure trajectory (task-dependent, typically 10-50 steps)
- Failure detection threshold: Tune based on classifier calibration

### Advantages

1. **Efficient use of negative data**: Failed attempts directly improve policy robustness
2. **Natural recovery behavior**: Mimics human trial-and-error learning
3. **Single model solution**: No need for separate recovery policies
4. **Compatibility with VLAs**: Leverages pretrained representations and in-context learning capabilities

### Evaluation Metrics

- **First-attempt success rate**: Performance in standard mode
- **Recovery success rate**: Success rate after initial failure
- **Overall success rate**: Combined performance with retry
- **Failure mode coverage**: Percentage of failure types that can be recovered

### Open Questions

1. Generalization to out-of-distribution failures not seen during training
2. Optimal pairing strategy for (failure, success) trajectories
3. Scaling to multiple retry attempts with hierarchical recovery strategies