# Memory Layers for Fast VLA Task Adaptation
## 1) Motivation & Intuition
Vision-language-action (VLA) models trained on large robotic datasets struggle with **continual learning**: fine-tuning on new tasks causes catastrophic forgetting or requires retraining the entire model. **Memory layers** offer a principled solution: augment the policy with a large, sparse external memory that can be rapidly updated for new tasks while keeping the base model frozen.

Key idea: attach **Product-Key-like memory modules** alongside select MLP layers in the action expert. During pretraining, learn both the base model $\phi$ and memory keys/values $\theta$. For online adaptation on novel tasks, **freeze $\phi$ and update only memory values $\theta$**—a small fraction of parameters that provides high capacity without interference. 

The approach scales memory capacity independently of compute (via sparse retrieval), enables **fast online updates** (only touched memory slots gradient-flow), and avoids forgetting (frozen backbone). Prototype in simulation with **SmolVLA** on **LIBERO**, then transition to **π₀** on a **Franka** arm with real-world data.

---
## 2) Problem Setup
### Task Distribution
Let $\mathcal{T}$ denote a distribution over robotic manipulation tasks with natural-language instructions. Each task $T$ has observation-action-instruction tuples $(o, a, c)$. We assume access to a large **pretraining distribution** $\mathcal{T}_{\text{pre}}$ (e.g., LIBERO-90) and a disjoint **online adaptation** set $\mathcal{T}_{\text{online}}$ (e.g., LIBERO-10).
### VLA Policy with Memory
A policy $f_{\phi,\theta}(o,c) \to a$ predicts actions given observations and language, where:
- $\phi$: frozen base weights (vision encoder, VLM decoder, action expert MLPs)
- $\theta = \{\mathbf{K}, \mathbf{V}\}$: trainable memory parameters (keys and values)

**Memory architecture.** For $L$ selected transformer layers in the action expert, replace the MLP $\mathrm{MLP}(x)$ with:
$$
\mathrm{MLP}_{\text{mem}}(x) = \mathrm{MLP}(x) + \mathrm{Memory}(x),
$$
where the memory module performs **product-key retrieval**:
1. **Query projection:** $q = W_q x \in \mathbb{R}^{h \times k}$ (multi-head, dimension $k$)
2. **Product quantization:** split $q$ into $(q_1, q_2) \in \mathbb{R}^{h \times k/2} \times \mathbb{R}^{h \times k/2}$
3. **Sparse k-NN:** compute top-$m$ keys in each subspace:
$$
\text{TopK}(q_1^\top K_1), \quad \text{TopK}(q_2^\top K_2), \quad K_1,K_2 \in \mathbb{R}^{n \times k/2}
$$
4. **Cartesian product:** combine to get $m^2$ candidates; select final top-$m$ indices $\{i_1,\ldots,i_m\}$
5. **Weighted aggregation:** retrieve values with softmax scores:
$$
\mathrm{Memory}(x) = \sum_{j=1}^m w_j \, V[i_j], \quad w_j \propto \exp(s_j)
$$
6. **Optional gating:** $\sigma(W_g x) \cdot \mathrm{Memory}(x)$ to control memory contribution
**Key properties:**
- Memory size: $n^2$ entries (e.g., $n=128 \Rightarrow 16\text{K}$ slots)
- Sparse access: only $m$ (e.g., $16$) slots accessed per forward pass
- Value table $\mathbf{V} \in \mathbb{R}^{n^2 \times d}$ kept in `float32` for stable gradients
- Keys $\mathbf{K}$ and query projections follow model dtype (e.g., `bfloat16`)
---
## 3) Training Protocol
### Phase 1: Joint Pretraining
Train the full model $(\phi, \theta)$ end-to-end on $\mathcal{T}_{\text{pre}}$ using standard supervised learning:
$$
\mathcal{L}(\phi,\theta; D) = \sum_{(o,a,c) \in D} \ell(f_{\phi,\theta}(o,c), a),
$$
where $\ell$ is MSE for flow-matching losses (or cross-entropy for discrete actions).
**Optimizer configuration:**
- Base parameters $\phi$: learning rate $\eta_{\phi} \in [10^{-4}, 5\times10^{-4}]$
- Memory values $\mathbf{V}$: separate learning rate $\eta_{\theta} \in [10^{-3}, 5\times10^{-3}]$ with no weight decay
- Gradient clipping: clip $\mathbf{V}$ and other params separately
**Memory placement:** attach to the last 1–2 expert layers (e.g., layers $[L-2, L-1]$ of action expert transformer).
### Phase 2: Online Task Adaptation
For each new task $T \in \mathcal{T}_{\text{online}}$ with support demos $D_T^{\text{support}}$:
1. **Freeze backbone:** set `requires_grad=False` for all $\phi$ parameters
2. **Update memory only:** fine-tune $\theta$ (memory values $\mathbf{V}$) for $K$ steps:
$$
\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}(\phi, \theta; D_T^{\text{support}})
$$
with higher learning rate $\alpha \in [5\times10^{-3}, 10^{-2}]$ and small batches
3. **Evaluate:** test adapted policy on $D_T^{\text{query}}$
**Key advantages:**
- Fast adaptation: only $|\mathbf{V}| \ll |\phi|$ parameters updated (e.g., $<1\%$ of total)
- Minimal forgetting: frozen $\phi$ retains pretraining knowledge
- Scalable memory: increase $n$ for more capacity without adding compute
---
## 4) Practical MVP Plan (Simulation First)

### Model & Configuration
Start with **SmolVLA-500M** on LIBERO. Memory configuration:
- **Placement:** last 2 expert transformer layers
- **Size:** $n=128$ keys ($16\text{K}$ total slots), rank $k=256$, heads $h=4$
- **Retrieval:** top-$m=16$ nearest neighbors per head
- **Gating:** enabled (`mem_gated=True`) with sigmoid modulation
- **Projection:** SwiLU-gated output projection
Estimated memory footprint: $\sim32\text{MB}$ in `float32` for values (cheap on a 4090).
### Pretraining Data & Tasks
- **Dataset:** LIBERO-90 (90 tasks across multiple suites)
- **Episodes:** use all available demonstrations per task
- **Batch size:** 8–16, sequence length 2048 tokens
- **Training:** 100K–200K steps with cosine decay scheduler
- **Validation:** hold out 10% of tasks for monitoring generalization
### Online Adaptation Setup
- **Held-out tasks:** LIBERO-10 (disjoint from pretraining)
- **Support set size:** 1, 3, 5, or 10 episodes per task
- **Adaptation steps:** $K \in \{50, 100, 200, 500\}$ gradient updates
- **Query evaluation:** measure success rate on 10–20 fresh rollouts per task

---
## 5) Evaluation Protocol
### Metrics
1. **Success rate** on query episodes vs. adaptation steps (0-shot, 1-shot, 5-shot, 10-shot)
2. **Sample efficiency:** episodes needed to reach 75% success threshold
3. **Forgetting:** measure pretraining task performance degradation after online updates
4. **Adaptation speed:** wall-clock time per adaptation step (should be fast with frozen backbone)
5. **Memory usage:** % of memory slots accessed per task; diversity of retrieved indices
### Baselines
1. **Fine-tune all:** unfreeze $\phi$ and $\theta$ for online adaptation (catastrophic forgetting baseline)
2. **Fine-tune head only:** update only action expert, freeze VLM (no memory)
3. **LoRA adapters:** attach low-rank adapters instead of memory, update adapters online
4. **Frozen baseline:** zero-shot performance (no adaptation)
### Ablations
- Memory size: $n \in \{64, 128, 256\}$
- Placement: single layer vs. multiple layers; earlier vs. later layers
- Retrieval: top-$m \in \{8, 16, 32\}$
- Gating: enabled vs. disabled
- Support set size: 1, 3, 5, 10 episodes
- Pretraining vs. no pretraining (random init memory)
---
## 6) Transition to Real (π₀ + Franka)
If simulation results validate fast, stable adaptation:
### Phase 1: Real-World Data Collection
1. **Robot platform:** Franka Emika Panda arm with wrist-mounted RGB-D camera
2. **Dataset:** use existing **DROID** multi-task demonstrations or collect custom teleoperation data
3. **Tasks:** 10–20 tabletop manipulation tasks with natural language (pick, place, pour, wipe, etc.)
4. **Domain calibration:** record camera intrinsics/extrinsics; apply minor image preprocessing to match pretraining distribution
### Phase 2: Model Port
1. Apply same memory layer architecture to **π₀** (or π₀.₅) expert transformer
2. Initialize memory from sim-pretrained checkpoint or train from scratch on DROID
3. Verify dtype/device compatibility; ensure `float32` memory values for gradient stability
### Phase 3: Online Adaptation on Robot
1. Collect 1–5 support demonstrations via teleoperation for a new task
2. Freeze π₀ backbone $\phi$; update memory values $\theta$ for $K=100$–$500$ steps
3. Deploy adapted policy in closed-loop with:
- Low initial velocity limits (safety)
- Collision detection / emergency stop
- Gradual trust increase as success rate improves
### Phase 4: Continual Learning Loop
- After each task, optionally retain a small replay buffer of previous tasks
- Periodically interleave replay samples during new task adaptation to mitigate forgetting
- Track memory slot usage across tasks (detect when capacity saturates)
---
## 7) Expected Outcomes
### Scientific Contributions
- **Efficient continual learning:** demonstrate that sparse memory layers enable $5$–$10\times$ faster adaptation than full fine-tuning with minimal forgetting
- **Scalable capacity:** show that increasing memory size $n$ improves multi-task capacity without proportional compute cost
- **Portability:** validate that memory-augmented VLAs transfer from sim (LIBERO) to real robot (Franka + DROID) with modest domain adaptation

## 8) Implementation Notes

### Code Structure
```
lerobot/src/lerobot/policies/modules/
├── memory_config.py # MemoryLayerConfig dataclass
├── memory_lite.py # HashingMemoryLite, MLPPlusMemory, attach/split helpers
```
### Training Command (with memory enabled)
```bash
lerobot-train \
--policy.path=outputs/smolvla_base \
--dataset.repo_id=outputs/libero_90 \
--env.type=libero \
--env.task=libero_90 \
--steps=100000 \
--batch_size=8 \
--policy.memory_layers=true \
--policy.memory_layer.enabled=true \
--policy.memory_layer.mem_n_keys=128 \
--policy.memory_layer.mem_heads=4 \
--policy.memory_layer.mem_knn=16 \
--policy.memory_layer.mem_k_dim=256 \
--policy.memory_layer.memory_lr=0.001
```
