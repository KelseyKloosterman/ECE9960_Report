# Paper Outline v4: Dr. Beary Good
## Don't Transfer Your LoRA Config: Rank-Performance Crossovers Across Model Scales

**Target:** arXiv preprint (short technical report, ~5–6 pages + references)
**Authors:** Barros, Kelsey, Dr. Sandrine *(confirm order with collaborators)*
**arXiv Category:** `cs.CL` (primary), `cs.HC` (cross-list)

---

## Framing Strategy

**Lead with the ML finding. Propose a practical fix. Use the pediatric VR system as the motivating application.**

The paper makes three moves:
1. **Demonstrate the problem:** LoRA adapter config performance ordering reverses between Llama 3.2 1B and 3B
2. **Show why it matters:** default practitioner advice assumes stable ordering — following it produces silently suboptimal results
3. **Propose a cheap fix:** a lightweight profiling protocol that identifies the right rank for a given model size in a fraction of the full training budget

This structure elevates the paper from "here's an observation" to "here's a problem and a tool."

---

## Paper Structure

### 1. Introduction (~0.75 page)

**Opening:** LoRA is the dominant method for adapting LLMs to downstream tasks. Practitioners select adapter hyperparameters — rank, target layers — using rules of thumb, then commonly reuse those choices across model sizes. This is especially prevalent in resource-constrained settings where developers test configurations on a smaller model before scaling up, or vice versa.

**The assumption:** Default LoRA configuration advice implicitly treats adapter config ordering as stable across model sizes:
- HuggingFace PEFT documentation: recommends rank 8-16 without model-size qualification
- Unsloth fine-tuning guide: frames rank selection as task-dependent, not model-size-dependent
- QLoRA (Dettmers et al., 2023): uses consistent LoRA configs across 7B-65B
- Biderman et al. (2024): recommends rank 16 without model-size conditioning

**The problem:** We show this assumption can fail. On Llama 3.2, a rank-4 adapter outperforms rank-8 on the 1B model but underperforms on the 3B model. A practitioner following standard advice would select the wrong configuration for one of these model sizes.

**The fix:** We propose a lightweight profiling protocol — training each candidate rank for a small fraction of the full budget — that correctly predicts the optimal configuration at a fraction of the cost.

**Motivating application:** We study this in the context of a role-adaptive pediatric VR hospital guide deployed on consumer hardware (~$600 desktop), where model size selection is dictated by memory constraints, making config transfer across model scales a practical necessity.

**Contributions:**
1. Empirical evidence that LoRA rank-performance ordering reverses between Llama 3.2 1B and 3B, validated with multiple seeds and statistical significance testing
2. Demonstration that this reversal generalizes beyond a single task domain
3. A cheap profiling protocol that identifies the optimal rank for a given model size using ≤10% of the full training compute
4. A complete system for age-adaptive pediatric hospital communication, with analysis of evaluation methodology for synthetic-data-trained models

---

### 2. Related Work (~0.75 page)

**2.1 LoRA Rank Selection and Scaling Behavior**
- Hu et al. (2022): original LoRA; rank-deficiency analysis showed low ranks often sufficient
- Dettmers et al. (2023): QLoRA; consistent configs across 7B-65B scales
- Biderman et al. (2024): LoRA Learns Less and Forgets Less; recommends rank 16, all modules, ≥4 epochs without model-size conditioning; full FT learns 10-100× higher rank perturbations
- Schulman et al. (2025): LoRA Without Regret; when LoRA matches full FT; MLP layers critical; capacity-dependent dynamics
- Hayou et al. (2024, 2026): LoRA+, μA framework; LR scaling depends on rank and init but analysis is within fixed model size
- Brenndoerfer (2025): qualitative note that larger models may need relatively smaller ranks — no controlled experiment
- Scaling Law for LoRA (arXiv 2501.03152): MIUB scaling across model size and rank
- **Positioning:** Existing work studies rank selection within a fixed model size or makes qualitative observations. We provide controlled experimental evidence of a performance *crossover* across model sizes and propose a cheap protocol to handle it.

**2.2 LLMs in Healthcare Communication**
- VAPS (Wan et al., 2025): VR + LLM-powered ECAs for clinical communication — general-purpose LLMs via prompting
- LLM-based virtual patient chatbots (Laverde et al., 2025; Holderried et al., 2024)
- Healthcare chatbot personalization (ACM GoodIT 2024): personality/disease-type adaptation via prompting
- LLMs in Medical Chatbots review (MDPI, 2025): deployment concerns
- **Gap:** No prior work uses fine-tuned, age-targeted adapters for pediatric communication.

**2.3 VR for Pediatric Care**
- Meta-analyses: VR reduces pain (SMD = −0.67) and anxiety (SMD = −0.74) in children (Bai et al., 2022 update)
- VR for preoperative anxiety: affects 50-75% of pediatric surgical patients (Fonseca et al., 2024)
- VR feasibility ages 4-16 (Kouijzer et al., 2022); Surgery Prep VR (Journal of Child Life, 2024)
- **Gap:** Existing VR interventions are passive distraction or scripted. None deploy conversational AI adapting to developmental stage.

---

### 3. System Design (~0.5 page)

**3.1 Architecture Overview**
- Diagram: VR frontend → inference server → LoRA adapter routing by age group → response
- Base model: Llama 3.2 Instruct (1B and 3B), 4-bit quantization
- The system includes a keyword-based input filter that rejects off-topic or inappropriate queries.
- Hardware reported in Section 4.1.

**3.2 Age-Targeted LoRA Adapters**
- Two adapters: younger children (5–11) and adolescents (12–18)
- Design rationale: vocabulary complexity, explanation depth, emotional register
- Adapter configurations: see Section 4 for the full experimental space

**3.3 Training Data**
- LIMA-style: small, curated synthetic dataset generated by Claude (Anthropic)
- Dataset statistics: number of examples, scenario categories, splits
- "We use synthetic data throughout. We address circularity risks in Section 4.5."

---

### 4. Experiments & Evaluation (~2 pages)

**4.1 Experimental Setup**

Hardware and software (reported factually, not as a contribution):
- Apple Mac Mini M4, 16GB unified memory
- macOS version, framework (PyTorch/MLX), quantization library, versions
- Cost: ~$600 consumer desktop

Training details:
- Quantization method: 4-bit (specify NF4/MLX native)
- Optimizer, learning rate, batch size, alpha, target modules
- **Compute equalization:** State explicitly how training budget is controlled across configs. Options:
  - (a) Fixed number of epochs (simplest; acknowledge the confound that higher-rank configs update more parameters per step)
  - (b) Fixed number of gradient steps (fairer per-step comparison; acknowledge configs see different amounts of data)
  - Pick one, justify it, state the confound.
- Report per-run: training time, peak memory

**4.2 Baselines**

These are essential. Without them, the reversal is relative to itself — you need anchors.

| Baseline | Model | What It Shows |
|----------|-------|---------------|
| **Zero-shot (no adapter)** | 1B | Lower bound — does fine-tuning help at all? |
| **Zero-shot (no adapter)** | 3B | Same |
| **Few-shot prompting (3-5 examples)** | 1B | Does LIMA-style FT beat in-context learning? |
| **Few-shot prompting (3-5 examples)** | 3B | Same |
| **Full fine-tuning** | 1B only | Upper bound for 1B (feasible on 16GB with gradient checkpointing) |

The full FT baseline on 1B is optional but extremely valuable. It lets you say: "rank 4 LoRA achieves X% of full fine-tuning performance on 1B, while rank 8 achieves Y%." That makes the reversal meaningful in absolute terms, not just relative.

Total additional runs for baselines: ~5 (zero-shot is free, few-shot is nearly free, full FT on 1B is 1 run).

**4.3 The Reversal: Ablation Design**

**Step 0: Estimate compute budget** (before committing — see next steps doc)

**Primary ablation (rank sweep, layers held constant):**
- Ranks: {2, 4, 8, 16}
- Models: {1B, 3B}
- Seeds: 3 per config
- Total: 24 runs
- Target layers: fixed at the set used in the original "fast" adapter

**Secondary ablation (layer sweep, rank held constant) — if compute allows:**
- Target layers: {4, 8, 16, all}
- Rank: fixed at 4
- Models: {1B, 3B}, 3 seeds
- Total: 24 additional runs

**4.4 Second Task Domain**

To demonstrate the reversal is not a dataset artifact, run a subset of configs on a second task.

**Recommended:** Alpaca-style general instruction following
- Use a small subset of the Alpaca dataset (1000 examples)
- Run your 4 ranks × 2 models × 1 seed = 8 runs
- Evaluate on a standard instruction-following metric or held-out perplexity
- You don't need full seeds or the complete ablation — you just need to show the crossover replicates on different data

**Alternative:** Evaluate all your pediatric-trained adapters on a standard benchmark like a subset of MMLU or HellaSwag
- This tests general capability retention, not task-specific performance
- It's cheaper (no additional training) and connects to Biderman et al.'s "forgetting" analysis

Total additional runs: 8 (if training on Alpaca) or 0 (if evaluating existing adapters on a benchmark).

**4.5 Profiling Protocol**

This is the prescriptive contribution that elevates the paper.

**Method:**
1. For each candidate rank in {2, 4, 8, 16}, train for T_short steps (e.g., 10% of the full training budget)
2. Record validation loss at T_short
3. Rank the candidates by validation loss
4. Compare this ranking to the final ranking after full training

**Evaluation:**
- Does the T_short ranking match the full-training ranking? (Kendall's tau or Spearman correlation between short-run ranking and full-run ranking)
- Test at multiple profile budgets: {5%, 10%, 20%} of full training
- Report: "A profiling run using X% of the training budget correctly identified the optimal rank in Y/Z cases"

**If this works:** it's a practical tool. A practitioner can spend 30 minutes profiling instead of committing to a 6-hour run with the wrong config. The cost of the profiling protocol is bounded and the benefit is avoiding the reversal trap.

**If this doesn't work (short-run rankings don't predict full-run rankings):** that's also a result. Report it honestly — it means the reversal emerges late in training and cheap profiling isn't sufficient. This informs future work on early-stopping-based rank selection.

Total additional runs: 0 — you already have the training curves from your full runs. Just evaluate checkpoints saved at 5%/10%/20% of training.

**4.6 Evaluation Metrics**

**Primary metrics (model-agnostic, non-circular):**
- Perplexity on held-out set — core reversal metric
- Flesch-Kincaid Grade Level / Coleman-Liau Index — age-appropriateness
- Latency (TTFT, tokens/sec) and peak memory — deployment feasibility

**Secondary metrics (reference-dependent, with caveats):**
- ROUGE-L / BERTScore vs. human-written references (30-50 prompts)
- ROUGE-L / BERTScore vs. Claude-generated references (for comparison only)
- LLM-as-judge (GPT-4 or Gemini): age-appropriateness (1-5), helpfulness (1-5), safety (pass/fail)

Report human-ref and Claude-ref metrics side by side.

**Statistical reporting:**
- Mean ± std across seeds for every metric × config × model
- Two-way ANOVA: model_size × rank → metric; report interaction F-statistic and p-value
- **Effect sizes:** eta-squared (η²) from ANOVA and Cohen's d for key pairwise comparisons
- State whether the reversal is visible in every individual seed or only in the mean
- If η² is small: "statistically significant but practically marginal" — be honest

**4.7 Addressing Synthetic Data Circularity**

Frame as a methodological discussion, not a defensive section.

**The problem:** Training on model-A-generated data, evaluating against model-A-generated references, risks measuring stylistic imitation rather than task quality.

**Mitigations applied:**
1. Primary evidence uses perplexity and readability (no reference dependency)
2. Human-written references (30-50 prompts, written by authors — ML grad students, not clinicians) break the generator-evaluator loop
3. Cross-family LLM judge (GPT-4/Gemini) diversifies evaluation

**What we do not claim:** High ROUGE-L against Claude references ≠ clinical quality.

**4.8 Human Evaluation**

- 50 prompts × 2 age groups × 2 key adapter configs = 200 outputs
- 2-3 evaluators, blinded
- Dimensions: age-appropriateness (1-5), helpfulness (1-5), safety (pass/fail), pairwise preference
- Inter-annotator agreement: Cohen's kappa or Krippendorff's alpha
- Optional: if a clinician reviews even a subset, report separately

---

### 5. Results & Discussion (~0.75 page)

**Ordered by strength:**

1. **The reversal:** Crossing-lines plot. ANOVA interaction effect with F, p, and η². Individual seed consistency.
2. **Generalization:** Does the reversal replicate on the second task?
3. **Profiling protocol:** Does cheap profiling predict the optimal rank? Kendall's tau across budget fractions.
4. **Absolute positioning:** How do the best adapters compare to zero-shot, few-shot, and full FT baselines? (This contextualizes the reversal's practical magnitude.)
5. **Age differentiation:** Readability scores confirm adapters produce age-appropriate output.
6. **Deployment viability:** Latency and memory on consumer hardware.
7. **Evaluation methodology:** Human-ref vs. Claude-ref metric agreement/divergence.

**Interpretation of the reversal:**
- Capacity-regularization hypothesis: 1B model's limited capacity benefits from implicit regularization of lower-rank adapters; 3B model exploits additional degrees of freedom
- Connection to Biderman et al. (full FT rank 10-100× higher than typical LoRA) and LoRA Without Regret (capacity-dependent dynamics)
- Honest scope: "We observe this on two model sizes in one family. We hypothesize a general capacity interaction but confirming this requires additional model families and scales."

---

### 6. Limitations (~0.25 page)

- **Two model sizes, one family.** Llama 3.2 1B and 3B only. The reversal may not generalize to other families (Qwen, Mistral, Gemma) or larger scales.
- **Two task domains.** Pediatric communication + one additional. Limited generalization evidence.
- **Synthetic data.** Mitigated but not fully eliminated.
- **Human eval scale.** 2-3 grad students, not clinicians.
- **No patient testing.** Research prototype only.
- **Profiling protocol tested on same data.** The protocol needs validation on independent tasks/models.

---

### 7. Ethics (~1 paragraph)

No patient data. LLM hallucination risk in pediatric context. Research prototype, not for unsupervised clinical use. Supplementary tool, not a replacement for human providers.

---

### 8. Reproducibility (~1 paragraph)

- Release: training scripts, evaluation code, adapter weights, LLM-judge prompts, experimental logs
- Release: dataset generation prompts (and the dataset itself if Anthropic ToS permit)
- Report: exact hardware, software versions, library versions, all random seeds
- All results reproducible from released materials on equivalent hardware

---

### 9. Conclusion (~0.25 page)

- LoRA config ordering is not stable across model sizes; default transfer practices produce suboptimal results
- A cheap profiling protocol (≤10% of training budget) can identify the right rank per model size
- Age-adaptive pediatric hospital communication is feasible on consumer hardware with LIMA-style synthetic fine-tuning
- Future work: additional model families/scales, clinical pilot, automated rank selection

---

## Figures

1. **Reversal plot (key figure):** Rank (x) vs. perplexity (y), lines for 1B and 3B, error bars, visible crossing
2. **Profiling protocol figure:** Short-run validation loss ranking vs. full-run ranking, across budget fractions
3. **Architecture diagram:** VR → inference → adapter routing → response
4. **Baselines + ablation table:** All configs + baselines, all metrics, mean ± std
5. **Readability comparison:** FK/CLI by age-group adapter
6. **Latency table:** TTFT, tokens/sec, peak memory per config

---

## References

### LoRA & PEFT
1. Hu et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models." ICLR.
2. Dettmers et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS.
3. Hayou et al. (2024). "LoRA+: Efficient Low Rank Adaptation."
4. Biderman et al. (2024). "LoRA Learns Less and Forgets Less." TMLR.
5. Schulman et al. (2025). "LoRA Without Regret." Thinking Machines Lab.
6. Hayou et al. (2026). "Learning Rate Scaling across LoRA Ranks." arXiv 2602.06204.
7. "Scaling Law for LoRA Based on MIUB." arXiv 2501.03152.

### Practitioner Defaults
8. HuggingFace PEFT docs — default rank recommendations
9. Unsloth fine-tuning guide — rank recommendations
10. Brenndoerfer (2025). LoRA Hyperparameters guide.

### Multi-LoRA / Foundation
11. Sheng et al. (2023). "S-LoRA."
12. Meta (2024). Llama 3 Technical Report.
13. Zhou et al. (2023). "LIMA: Less Is More for Alignment."

### Healthcare / VR
14. Wan et al. (2025). VAPS. CHI EA.
15. Laverde et al. (2025). LLM virtual patient chatbot. PMC.
16. Healthcare chatbot personalization. ACM GoodIT 2024.
17. LLMs in Medical Chatbots review. MDPI 2025.
18. Bai et al. (2022). VR pediatric pain/anxiety meta-analysis. PMC.
19. Fonseca et al. (2024). VR preoperative caregiver anxiety.
20. Kouijzer et al. (2022). VR distraction/relaxation pediatric. Frontiers Digital Health.
21. Nature-based mindfulness VR. Frontiers Pediatrics 2024.
22. Surgery Prep VR. Journal of Child Life 2024.

### Evaluation
23. Readability formula references (Flesch, Kincaid, Coleman-Liau)
24. LLM-as-judge methodology (cite specific framework used)
