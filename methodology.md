# Methodology

## 3.1 System Overview

This work extends a virtual hospital framework designed to support safe, naturalistic communication between pediatric patients, caregivers, and clinical decision-makers. The platform instantiates a digital recreation of Victoria Hospital (London, Ontario) in which users interact with an AI-powered guide through natural language queries spanning procedural, logistical, and informational domains. Emotional-support queries are in scope only to the extent of age-appropriate reassurance and redirection (e.g., acknowledging nervousness, normalizing common fears, directing the user to a nurse or guardian); clinical assessment, therapeutic guidance, and crisis intervention are explicitly out of scope. The system's goal is to provide constrained, clinically grounded responses that are both age-appropriate and consistent with institutional guidelines.

The conversational interface routes each incoming query to one of two age-targeted response modules: one serving younger children (ages 5–11) and one serving adolescents (ages 12–18). Routing is determined by the established user profile. Input safety filtering — rejecting queries that fall outside the system's clinical scope before they reach the language model — is not implemented in the current system. Integration of a learned guard model (such as LlamaGuard) to provide this filtering layer is identified as planned future work. This report focuses on the fine-tuning methodology, data pipeline, and evaluation framework that underpin the age-targeted response modules.

## 3.2 Base Models and Infrastructure

We use the Llama 3.2 Instruct model family [Dubey et al., 2024] as our foundation, specifically the 1B and 3B parameter variants. Both models are loaded in 4-bit group quantized form via the MLX-community pre-quantized weights (`mlx-community/Llama-3.2-3B-Instruct-4bit` and `mlx-community/Llama-3.2-1B-Instruct-4bit`), using the `mlx_lm` framework (version 0.30.7) [Apple, 2023] on Apple Silicon. The training and evaluation environment runs Python 3.12, MLX 0.30.6, transformers 5.2.0, datasets 4.6.0, textstat 0.7.13, and scikit-learn ≥ 1.4.0 on macOS with an Apple Mac Mini M4 (16 GB unified memory). This hardware choice is a deliberate design constraint: the system is intended to be deployable in under-resourced clinical or educational settings without access to cloud infrastructure or GPU clusters. The 16 GB unified memory ceiling makes full-parameter fine-tuning impractical at both model scales. A theoretical lower-bound calculation for full FT — accounting for model weights, gradients, and Adam optimizer states (m and v), all in bfloat16 — gives approximately 12.4 GB for the 1B model (1.50B parameters) and 29.9 GB for the 3B model (3.61B parameters), before activations or allocator overhead. If optimizer states are maintained in float32 (standard mixed-precision practice), the figures rise to approximately 18.0 GB (1B) and 43.3 GB (3B). Under either assumption, the 3B model's total full-FT footprint exceeds the 16 GB ceiling by a factor of nearly 2× (bfloat16) to approximately 2.7× (fp32 optimizer states); the 1B model is at best borderline with no headroom for activations or framework overhead. Parameter-efficient adaptation via LoRA is therefore the only viable fine-tuning strategy on this hardware for both model scales.

## 3.3 LoRA as Declarative Programming

We employ Low-Rank Adaptation (LoRA) to specialize the base models for clinical deployment without modifying full model weights. In the standard LoRA formulation, a pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$ is held frozen during training. A low-rank update is expressed as:

$$\Delta W = BA$$

where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$, with rank $r \ll \min(d, k)$. During fine-tuning, only $B$ and $A$ are updated; during inference, the effective weight becomes $W_0 + \Delta W$. We insert LoRA adapters into all linear projection modules within the targeted transformer layers. For each adapted layer, this covers seven weight matrices: the four attention projections (`self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`, `self_attn.o_proj`) and the three MLP projections (`mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`). This is the mlx-lm default: it targets all linear layers within the selected blocks, unlike the commonly documented example in HuggingFace PEFT tutorials [Mangrulkar et al., 2022] (which typically targets only query and value projections for Llama-family models). The update magnitude is controlled by a scalar multiplier (`scale`). Note that mlx-lm uses a raw scalar rather than the $\frac{\alpha}{r}$ convention used in Hu et al. (2022) and HuggingFace PEFT; in this implementation, the adapter output is multiplied directly by `scale` without rank division. We use `scale = 20.0` (the mlx-lm validated default) held constant across all runs.

*The following framing is offered as a conceptual analogy to motivate the design philosophy of this work; it is not a falsifiable scientific claim.*

We conceptualize LoRA adapters as a form of declarative programming over foundation models. In conventional software, declarative programming specifies *what* a system should do rather than prescribing *how* it should do it. We argue that LoRA-based clinical adaptation has the same structure: hospital guidelines, care pathways, patient information packages, and age-appropriate communication standards collectively define a specification of allowable system behavior, while the adapted model realizes this specification during inference. Knowledge is not encoded as explicit procedural rules but as a learned bias over the model's output distribution — an approximation that shares the spirit of constraint-based specification while remaining an empirical, learned phenomenon rather than a formal guarantee. This perspective facilitates interpretability and governance by making the locus of specialization explicit and modular rather than entangled with base model weights.

**Adapter configurations evaluated.** Two adapter configurations were trained and compared in this work:

- **Standard adapter** (rank $r = 8$, `num_layers = 16`): adapters cover the upper 16 transformer layers — 16/28 (57%) of the 3B model and 16/16 (100%) of the 1B model. This configuration provides stronger adaptation capacity. Note that for the 1B model, `num_layers = 16` saturates the entire model depth, making the Standard 1B configuration qualitatively closer to full-layer LoRA coverage than to a typical partial-depth deployment; this limits direct comparability with the Standard 3B configuration on the layer-coverage dimension.
- **Fast adapter** (rank $r = 4$, `num_layers = 8`): adapters cover the upper 8 layers — 8/28 (29%) of the 3B model and 8/16 (50%) of the 1B model. This configuration prioritizes efficiency: fewer trainable parameters, faster training, and smaller adapter size on disk.

All other hyperparameters — including `scale = 20.0` — are identical across both configurations. We note that holding `scale` constant while varying rank means the two configurations are not matched on effective update magnitude: because mlx-lm applies the raw scalar without rank normalization, lower-rank adapters receive the same per-step scaling as higher-rank adapters, whereas the Hu et al. convention would increase $\alpha$ proportionally with rank to normalize per-step updates. This is an additional confounder in the comparison, distinct from the rank-and-layer-count co-variation noted below. A broader rank sweep ($r \in \{2, 4, 8, 16\}$) with multiple seeds per configuration is identified as planned future work; the current study provides two comparison points that bracket the practical range for this dataset scale and hardware constraint.

## 3.4 Synthetic Data Generation

Following the LIMA hypothesis [Zhou et al., 2023] — which holds that a small number of carefully curated, high-quality examples is sufficient to align a capable base model with a target behavior — we construct synthetic training corpora rather than collecting naturalistic patient data. We note that the LIMA hypothesis was originally demonstrated on models considerably larger than the sub-3B models used here; whether the same alignment efficiency holds at this scale is an empirical question and a limitation of the current design. This choice is further motivated by the practical impossibility of collecting real pediatric clinical dialogues at scale and the ethical constraints on using such data in research systems.

Training data was generated by Claude Opus 4.6 (Anthropic; API snapshot March 12, 2026) with Victoria Hospital's publicly available patient and family reference materials attached as grounding artifacts. All hospital-specific details (visiting hours, departmental procedures, staff roles, and facility layout) are sourced directly from these documents; the model was instructed not to fabricate facility-specific information not present in the reference materials. Each example is a JSON object with fields for instruction, response, age group role (`5-11` or `12-18`), and topic category. Data is organized into five topic categories reflecting the communication needs of pediatric patients and caregivers:

- **What to Expect** — descriptions of medical procedures, equipment, and sensory experiences
- **Who Are These People?** — age-appropriate explanations of clinical staff roles
- **Hospital Rules & Routines** — visiting hours, daily schedules, and preparation information
- **Emotional Reassurance** — responses to fear, nervousness, or confusion (strictly non-diagnostic; excludes therapeutic guidance or crisis intervention)
- **FAQs / General Curiosity** — open-ended exploratory questions a child or caregiver might ask

Each adapter serves a single age group. Each topic category targets 100 training examples per adapter; the actual post-review count is approximately 475 per adapter (some examples were filtered during spot-check review), for a nominal total of up to 500 per adapter (5 categories × 100 target) and up to 1,000 across both adapters. Safety constraints were enforced at the generation prompt level: the generator was instructed never to imply or produce a diagnosis, never to recommend medications or treatments, and to redirect any medical or emergency question to a nurse, doctor, or parent/guardian. Generated examples were spot-checked by the first author for adherence to safety rules, age-appropriateness, and factual grounding before inclusion. We acknowledge that a spot-check of a sample — rather than exhaustive independent review — is a limitation of the data pipeline.

A held-out validation set of 100 examples was generated in a separate pass with explicit independence instructions: 10 examples per category per age group across 5 categories and 2 age groups = 100 examples total, split evenly as 50 per age group. Each adapter is evaluated on its age-matched 50-example subset during training (loss monitoring) and for the primary perplexity evaluation, ensuring perplexity reflects adapter fit on the target age group's distribution rather than a pooled cross-age score. Because both the training set and this validation set are drawn from the same generative process and share the same grounding documents and generation prompts, they are not statistically independent in the sense required for unbiased generalization estimation; perplexity results should therefore be interpreted as a measure of in-distribution coherence rather than out-of-distribution generalization.

We treat the synthetic corpus as a programmable object in the sense that its properties — including correctness, completeness, and internal consistency — can be varied systematically by modifying generation prompts and constraints. This design supports study of how corpus quality propagates through LoRA-based adaptation into downstream system behavior.

## 3.5 Training Protocol

Adapters are trained using the Adam optimizer with a cosine decay learning rate schedule: peak learning rate $1 \times 10^{-5}$, linear warmup over the first 100 steps, decaying toward zero over 600 total steps. Batch size is 4; maximum sequence length is 768 tokens; LoRA dropout is 0.0. Loss is computed only over assistant response tokens (`mask_prompt: true`); user-turn tokens contribute zero gradient. The `scale` parameter (20.0) is held constant across both configurations. Each configuration is trained for a fixed budget of 600 gradient steps — equivalent to approximately 4.8 passes at the nominal 500-example count, or approximately 5.1 passes at the post-review count of ~475 (fewer examples means more passes per 600 steps at fixed batch size). Training is step-based (mlx_lm.lora `iters` parameter), not epoch-based; there is no guarantee of clean epoch boundaries. All runs use seed 42 for reproducibility.

The standard and fast adapters differ only in rank ($r = 8$ vs. $r = 4$) and the number of adapted layers (`num_layers = 16` vs. `num_layers = 8`). Both configurations are trained across both model sizes (1B and 3B) and both age groups (5–11, 12–18), yielding 8 total training runs (2 configurations × 2 model sizes × 2 age groups). Training is conducted locally on the Mac Mini M4 hardware described in Section 3.2. We acknowledge that the two configurations differ in both rank and layer count simultaneously, making it impossible to attribute performance differences to either factor in isolation; this is a limitation of the current comparison design. A controlled ablation that varies rank and layer depth independently is planned as future work.

Model checkpoints are saved every 100 steps (`save_every: 100`), yielding six checkpoints per run at steps 100, 200, 300, 400, 500, and 600 (approximately 16.7% intervals). Validation loss is computed every 60 steps (`steps_per_eval: 60`) using the full validation set (`val_batches: -1`), providing a dense loss trajectory for detecting overfitting. The fully trained adapter at step 600 is saved as the final artifact and constitutes the model evaluated in the results; earlier checkpoints are auxiliary.

## 3.6 Evaluation Framework

Evaluation is organized into two tiers: metrics that have been computed and reported in this work, and metrics identified as planned extensions for future work.

### 3.6.1 Completed Evaluation

**Readability.** Each adapter's outputs are evaluated on five corroborating readability metrics computed over model-generated responses: Flesch-Kincaid (FK) grade level [Kincaid et al., 1975], SMOG index [McLaughlin, 1969], Gunning Fog index [Gunning, 1952], Coleman-Liau index [Coleman and Liau, 1975], and lexical diversity (type-token ratio). For the ages 5–11 adapter, a pass/fail target of FK ≤ 7.0 is applied (corresponding to grade 6, the upper bound of the target age range); no hard ceiling is applied for the 12–18 adapter. Readability is evaluated on outputs generated from the held-out validation prompts using the `textstat` library (version 0.7.13) [Bansal, 2023].

**Latency and throughput.** For each model–adapter configuration, inference is run on the evaluation prompt set with latency, token count, and tokens-per-second recorded per response. Average latency, minimum, maximum, average tokens per response, and average tokens per second are reported. A 1.0-second target, reflecting real-time VR interaction requirements, is used as a deployment feasibility threshold.

**Inter-role style separation.** To verify that the two age-targeted adapters produce genuinely distinct communication registers, a TF-IDF + logistic regression classifier is trained on the combined outputs of both adapters using 5-fold cross-validation (scikit-learn [Pedregosa et al., 2011]). Each adapter is run on its own age-matched held-out set with no system prompt (`system_prompt=None` in `src/generate_outputs.py`), matching training conditions (`data/age_5_11/valid.jsonl` for the 5–11 adapter; `data/age_12_18/valid.jsonl` for the 12–18 adapter). This design eliminates system-prompt vocabulary as a confound, but because each adapter is evaluated on its own age-matched prompts rather than a shared neutral prompt set, age-matched instruction vocabulary remains a potential signal source for the classifier in addition to adapter-encoded register. A stronger design would run both adapters on an identical shared prompt set; the current design is a practical constraint of age-matched validation data. Classification accuracy is interpreted as: ≈ 0.50 = chance (no separation), ≈ 0.70 = moderate, ≥ 0.90 = strong register difference.

### 3.6.2 Planned Evaluation (Future Work)

The following evaluation components are identified as important extensions but have not been run in this work:

**Reference-based metrics.** ROUGE-L [Lin, 2004] and BERTScore [Zhang et al., 2020] against human-written reference responses would provide a reference-dependent measure of response quality. This requires construction of a human reference set — a set of responses to evaluation prompts written by independent evaluators — which has not yet been completed. Comparison against Claude-generated references is also planned, to characterize the degree to which synthetic evaluation aligns with human judgment.

**LLM-as-judge evaluation.** Systematic evaluation of response quality along the dimensions of correctness, completeness, lack of ambiguity, and minimization of extraneous content using an LLM judge from a different model family (e.g., GPT-4 or Gemini) to avoid circularity with the Claude-generated training data is planned. The judge would rate responses on age-appropriateness (1–5), helpfulness (1–5), and safety (pass/fail) using a structured prompt.

**Safety evaluation.** Safety evaluation is not conducted in this work. The current system does not implement input-level safety filtering; integration of a learned guard model (e.g., LlamaGuard [Inan et al., 2023]) to screen queries and responses is planned as the primary safety layer. Adversarial probe testing — a structured set of prompts designed to elicit diagnostic language, medication recommendations, or age-inappropriate content — is also planned, and will become meaningful once the guard model layer is in place.

**Multi-seed and rank ablation.** A controlled rank sweep ($r \in \{2, 4, 8, 16\}$) with multiple independent seeds per configuration, holding all other hyperparameters constant, is planned to provide statistically grounded characterization of rank-performance relationships across model scales.

---

## References

**[Apple, 2023]** Apple. *MLX: Efficient and Flexible Machine Learning on Apple Silicon*. 2023. Available: https://github.com/ml-explore/mlx

**[Bansal, 2023]** S. Bansal. *textstat: Python library to calculate readability statistics of a text object* (v0.7.13). 2023. Available: https://pypi.org/project/textstat/

**[Coleman and Liau, 1975]** M. Coleman and T. L. Liau, "A Computer Readability Formula Designed for Machine Scoring," *Journal of Applied Psychology*, vol. 60, no. 2, pp. 283–284, 1975.

**[Dubey et al., 2024]** A. Dubey et al. (Meta AI), "The Llama 3 Herd of Models," *arXiv preprint arXiv:2407.21783*, 2024.

**[Gunning, 1952]** R. Gunning, *The Technique of Clear Writing*. New York: McGraw-Hill, 1952.

**[Hu et al., 2022]** E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, "LoRA: Low-Rank Adaptation of Large Language Models," in *Proc. ICLR*, 2022.

**[Inan et al., 2023]** H. Inan, K. Upasani, J. Chi, R. Rungta, K. Iyer, Y. Mao, M. Tontchev, Q. Hu, B. Fuller, D. Testuggine, and M. Khabsa, "Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations," *arXiv preprint arXiv:2312.06674*, Meta AI, 2023.

**[Kincaid et al., 1975]** J. P. Kincaid, R. P. Fishburne, R. L. Rogers, and B. S. Chissom, "Derivation of New Readability Formulas for Navy Enlisted Personnel," *Research Branch Report 8-75*, Naval Technical Training Command, 1975. (The Flesch-Kincaid grade-level formula.)

**[Lin, 2004]** C. Y. Lin, "ROUGE: A Package for Automatic Evaluation of Summaries," in *Proc. Workshop on Text Summarization Branches Out (ACL 2004)*, pp. 74–81, 2004.

**[Mangrulkar et al., 2022]** S. Mangrulkar, S. Gugger, L. Debut, Y. Belkada, S. Paul, and B. Bossan, "PEFT: State-of-the-art Parameter-Efficient Fine-Tuning Methods," 2022. Available: https://github.com/huggingface/peft

**[McLaughlin, 1969]** G. H. McLaughlin, "SMOG Grading — A New Readability Formula," *Journal of Reading*, vol. 12, no. 8, pp. 639–646, 1969.

**[Pedregosa et al., 2011]** F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

**[Zhang et al., 2020]** T. Zhang, V. Kishore, F. Wu, K. Q. Weinberger, and Y. Artzi, "BERTScore: Evaluating Text Generation with BERT," in *Proc. ICLR*, 2020.

**[Zhou et al., 2023]** C. Zhou, P. Liu, P. Xu, S. Iyer, J. Sun, R. Maddela, X. Peng, S. Shrivastava, M. Lewis, L. Zettlemoyer, and O. Levy, "LIMA: Less Is More for Alignment," in *Advances in Neural Information Processing Systems* (NeurIPS), 2023.
