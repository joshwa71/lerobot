# Meta-Learning Lora for Fast VLA Adaptation


## 1) Motivation & Intuition
Modern vision-language-action (VLA) policies adapt slowly when fine-tuned on new, task-specific demonstrations. Full meta-learning is compute-intensive, but **meta-learning only LoRA adapter parameters** offers a cheap path to **one/few-shot adaptation**. The plan: prototype in simulation with **SmolVLA** on **LIBERO** tasks, then—if successful—collect real demonstrations and carry the same recipe to **π₀** and a **Franka** arm.

Key idea: freeze backbone weights $\phi$ (vision encoder, language encoder, most of the policy), and **meta-learn** a small set of LoRA parameters $\theta$ that provide a good initialization for rapid task-specific tuning.

---

## 2) Problem Setup
Let $\mathcal{T}$ denote a distribution over tasks (e.g., LIBERO tasks with natural-language instructions). Each task $T$ has a dataset $D_T = \{(o, a, c)\}$ with observations $o$, actions $a$, and instruction $c$. Split $D_T$ into **support** $S_T$ (few demos) and **query** $Q_T$.

A VLA policy $f_{\phi,\theta}$ predicts actions given $(o,c)$, where $\phi$ are frozen base weights and $\theta$ are trainable **LoRA** parameters inserted in selected modules.

**LoRA parameterization.** For a target linear map with pretrained weight $W_0 \in \mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$, LoRA adds a low-rank update
$$
W = W_0 + \Delta W,\quad \Delta W = B A,\quad A \in \mathbb{R}^{r \times d_{\text{in}}},\; B \in \mathbb{R}^{d_{\text{out}} \times r},
$$
with small rank $r$ (e.g., $r \in \{4,8,16\}$). We place LoRA on attention projections ($q,k,v,o$) and MLPs ($\mathrm{fc1},\mathrm{fc2}$) in the VLM decoder and in the action head.

**Supervised objective.** For a batch $D$, use a behavior-cloning loss
$$
\mathcal{L}(\theta; D) \;=\; \sum_{(o,a,c)\in D} \ell\!\left(f_{\phi,\theta}(o,c),\, a\right),
$$
where $\ell$ is e.g., mean-squared error for continuous actions (extendable to distributional/flow-matching variants).

---

## 3) Meta-Learning Objective
We seek a LoRA initialization $\theta^\star$ that adapts quickly on new tasks via a few gradient steps on $S_T$.

**Inner adaptation (per task).**
$$
\theta'_T \;=\; \theta \;-\; \alpha\, \nabla_{\theta}\,\mathcal{L}(\theta;\, S_T), \quad \text{(1–5 steps, small $\alpha$).}
$$

Two compute-friendly outer-loop choices:

- **First-Order MAML (FOMAML):**
  $$
  \theta \;\leftarrow\; \theta \;-\; \eta\, \frac{1}{B}\sum_{T\sim\mathcal{T}} \nabla_{\theta}\,\mathcal{L}\!\left(\theta'_T;\, Q_T\right).
  $$

- **Reptile (no second-order terms):**
  $$
  \theta \;\leftarrow\; \theta \;+\; \beta\, \frac{1}{B}\sum_{T\sim\mathcal{T}} \left(\theta'_T - \theta\right).
  $$

In both cases, only $\theta$ (LoRA) is updated; $\phi$ stays frozen. An ANIL-style variant adapts **only** action-head LoRA in the inner loop.

---

## 4) Practical MVP Plan (Simulation First)
**Model & hooks.** Start with **SmolVLA**. Insert LoRA into:
- VLM decoder: $q,k,v,o,\mathrm{fc1},\mathrm{fc2}$
- Action head: same set of projections/MLPs

Use small ranks ($r{=}8$ action head, $r{=}4$ decoder) and light dropout (e.g., $0.05$). Freeze everything else.

**Data & task sampling.** Use **LIBERO** suites as $\mathcal{T}$. For each meta-batch:
- Sample $B$ tasks.
- For each task, pick $K\in\{1,3,5\}$ support demos and a query set (e.g., $10$–$20$ episodes).
- Apply mild domain randomization in sim to encourage generalization.

**Optimization.**  
Inner loop: $K$ steps, learning rate $\alpha \in [10^{-4},5\!\times\!10^{-4}]$.  
Outer loop: FOMAML with step size $\eta$ or Reptile with $\beta \in [0.05,0.2]$; meta-batch $B\in[4,16]$.

---

## 5) Evaluation Protocol
- **Few-shot adaptation curves** on held-out LIBERO tasks: 0-shot, 1-shot, 3-shot, 5-shot (report success rate and final return vs. adaptation steps).  
- **Baselines:** (i) plain LoRA fine-tuning from random init $\theta_0$, (ii) action-head-only LoRA, (iii) full head fine-tune (no LoRA).  
- **Metrics:** success rate on query episodes, adaptation steps to threshold, sample efficiency, and wall-clock per adaptation.  
- **Ablations:** rank $r$, where-to-LoRA (decoder vs. head), inner steps $K$, support size, Reptile vs. FOMAML.

---

## 6) Transition to Real (π₀ + Franka)
If sim results are promising:
1. **Data collection:** teleop a Franka to gather a small, multi-task dataset with instructions.
2. **Model port:** apply the same LoRA placement on **π₀** (or π₀.₅) and initialize with the meta-learned $\theta^\star$ from sim.
3. **Few-shot adaptation on robot:** run $K$ inner steps on $S_T$ (few demos) and evaluate on $Q_T$; add light camera/domain calibration.  
4. **Safety:** start with low-velocity constraints and collision checks; gradually relax as success improves.

---

## 7) Expected Outcomes
- A compact, reusable **LoRA prior** $\theta^\star$ that enables **1–5-shot** adaptation on novel VLAs.  
- Clear evidence that **meta-learning only adapters** yields faster/safer on-robot adaptation than plain fine-tuning, while staying within a modest compute budget.

