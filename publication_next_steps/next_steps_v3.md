# Dr. Beary Good — Concrete Next Steps (v3)

Key changes from v2: compute estimation still first, added baselines, second task domain, profiling protocol (zero additional cost), effect size reporting, compute equalization decision, hardware framing cleaned up.

---

## Phase 0: Compute Budget Estimation (Day 1)

### Step 0: Time your training runs
- [ ] Train ONE adapter: 1B model, rank 8, your standard layer config, 1 seed → record wall-clock time and peak memory
- [ ] Train ONE adapter: 3B model, rank 8, same layer config, 1 seed → same
- [ ] Check: was macOS swap engaged? (Activity Monitor → Memory Pressure)
- [ ] Also test: can you do full fine-tuning on the 1B model with gradient checkpointing? Try it. If it works, record time and memory. If it OOMs, note that.

### Step 0b: Calculate total budget and commit

**Primary ablation:** 4 ranks × 2 models × 3 seeds = 24 runs
**Baselines:** ~5 runs (zero-shot free, few-shot free, full FT on 1B = 1 run)
**Second task:** 4 ranks × 2 models × 1 seed = 8 runs (on Alpaca subset)
**Total:** ~37 training runs

Estimated time = (time_1B × ~17) + (time_3B × ~16) + (full_FT_1B × 1) = ___ hours

**Decision rules:**
- If total ≤ 96 hours (~4 days): run everything above
- If total ≤ 150 hours (~6 days): cut second task seeds or reduce to 3 ranks {4, 8, 16}
- If total > 150 hours: cut second task to evaluation-only (run existing adapters on MMLU/HellaSwag, no additional training)

**Important:** Save training checkpoints at 5%, 10%, and 20% of total steps. This costs nothing (just disk space) and gives you the profiling protocol data for free.

**Deliverable:** Time estimates, chosen design, estimated completion date — written down before any runs start.

---

## Phase 1: Coordination & Setup (Week 1, parallel with Phase 0)

### Step 1: Co-author alignment
- [ ] Meet with Kelsey and Dr. Sandrine
- [ ] Share outline v4
- [ ] Agree on: authorship order, timeline, sign-off process
- [ ] arXiv endorsement from Dr. Sandrine
- [ ] Publication/IP policies for your program
- [ ] Open-source release plan (adapters, scripts, generation prompts, data)

### Step 2: arXiv account
- [ ] Register at https://arxiv.org/user/register
- [ ] Request cs.CL endorsement

### Step 3: Experiment tracking
- [ ] Set up W&B / MLflow / structured CSV
- [ ] Naming: `{model}_{rank}_{layers}_{seed}` (e.g., `3B_r8_L8_seed42`)
- [ ] Decide on compute equalization:
  - **Option A — Fixed epochs:** Simpler. Every config trains for the same number of epochs (e.g., 4, per Biderman et al. recommendation). Acknowledge confound: higher-rank configs update more parameters per step.
  - **Option B — Fixed steps:** Fairer per-gradient-update comparison. Acknowledge confound: configs see different fractions of the dataset.
  - Pick one. Document it. Stick with it.

**Deliverable:** Tracking set up, compute equalization decision documented.

---

## Phase 2: Human References & Eval Prep (Week 1 — start immediately)

### Step 4: Build human-written reference set
- [ ] Select 50 diverse prompts from test set (25 per age group, varied scenarios)
- [ ] You write answers for all 50
- [ ] Kelsey independently writes answers for all 50 (no coordination until both done)
- [ ] Document: "References written by two ML graduate students, not clinicians."

**Deliverable:** 50 prompts × 2 authors = 100 reference responses.

### Step 5: Set up LLM-as-judge
- [ ] Choose: GPT-4 or Gemini (not Claude)
- [ ] Write judge prompt for: age-appropriateness (1-5), helpfulness (1-5), safety (pass/fail)
- [ ] Test on 5-10 examples
- [ ] Save exact prompt for appendix

### Step 6: Recruit human evaluators
- [ ] 2-3 grad students, committed deadline, ~7 hours each
- [ ] Prepare blinded evaluation materials (outputs labeled A/B, no config info)
- [ ] Optional: ask Dr. Sandrine about a clinician for a subset

---

## Phase 3: Run Experiments (Weeks 1-2)

### Step 7: Run baselines (do these first — they're cheap and essential)
- [ ] **Zero-shot:** Run 1B and 3B base models (no adapter) on your test set. Record perplexity, readability, and generate outputs for the 50 human-ref prompts.
- [ ] **Few-shot:** Same, but with 3-5 in-context examples prepended to each prompt.
- [ ] **Full fine-tuning on 1B** (if feasible from Step 0): Train 1B model with full parameter updates using gradient checkpointing. This is your upper bound.

These take ~1 day total and completely transform the paper's interpretive power.

### Step 8: Write training loop script
- [ ] Parameterize: model_size, rank, target_layers, seed
- [ ] Auto-log: hyperparameters, loss curve, training time, peak memory
- [ ] **Save intermediate checkpoints** at 5%, 10%, and 20% of total steps (for profiling protocol — this is free)
- [ ] Save final adapter checkpoints

### Step 9: Run primary ablation
- [ ] Ranks {2, 4, 8, 16} × {1B, 3B} × 3 seeds = 24 runs
- [ ] Target layers held constant
- [ ] Consecutive integer seeds (42, 43, 44) — no cherry-picking

### Step 10: Run secondary ablation (if compute allows)
- [ ] Layers {4, 8, 16, all} × {1B, 3B} × 3 seeds = 24 runs
- [ ] Rank held constant at 4

### Step 11: Second task domain
**Option A (additional training — 8 runs):**
- [ ] Prepare Alpaca subset (1000 examples)
- [ ] Train ranks {2, 4, 8, 16} × {1B, 3B} × 1 seed = 8 runs
- [ ] Evaluate on held-out Alpaca perplexity

**Option B (no additional training — 0 runs):**
- [ ] Take your existing pediatric-trained adapters
- [ ] Evaluate all of them on a subset of MMLU or HellaSwag
- [ ] This tests general capability retention and connects to Biderman et al.'s forgetting analysis
- [ ] The reversal may or may not appear on general benchmarks — either result is informative

**Option A is stronger but costs 8 training runs. Option B is free but less direct.**

### Step 12: Generate outputs for evaluation
- [ ] Every trained adapter: generate responses to full test set + 50 human-ref prompts
- [ ] Save all outputs with config metadata

**Deliverable:** All checkpoints (including intermediate), all logs, all generated outputs.

---

## Phase 4: Evaluation (Weeks 2-3)

### Step 13: Primary metrics
- [ ] **Perplexity** on held-out set — per config, per model, per seed
- [ ] **Readability:** Flesch-Kincaid Grade Level, Coleman-Liau Index on generated outputs
- [ ] **Latency:** TTFT (median and p95 over 20+ runs), tokens/sec, peak memory during inference

### Step 14: Secondary metrics
- [ ] ROUGE-L / BERTScore vs. human references
- [ ] ROUGE-L / BERTScore vs. Claude references (for comparison)
- [ ] LLM-as-judge scores on 200 outputs

### Step 15: Statistical analysis
- [ ] Compile: config × model_size × seed → all metrics
- [ ] Mean ± std per config × model
- [ ] **Two-way ANOVA:** model_size × rank → perplexity. Report:
  - Main effect of rank: F, p
  - Main effect of model_size: F, p
  - **Interaction term: F, p** (this is the reversal test)
  - **Effect size: η² (eta-squared)** for the interaction
- [ ] **Cohen's d** for key pairwise comparisons (e.g., rank 4 vs. rank 8 within each model size)
- [ ] Check: reversal visible in every seed, or only in the mean?
- [ ] Repeat for second task domain metrics
- [ ] Create reversal plot: rank (x) vs. perplexity (y), 1B and 3B lines, error bars

### Step 16: Profiling protocol analysis (zero additional runs needed)
- [ ] From saved intermediate checkpoints (5%, 10%, 20% of training):
  - Compute validation loss at each checkpoint for every config
  - Rank configs by validation loss at each checkpoint
  - Compare to final ranking
- [ ] **Kendall's tau** (or Spearman's rho) between short-run ranking and full-run ranking
- [ ] Do this separately for 1B and 3B
- [ ] Report: "Profiling at X% of training budget correctly identified the optimal rank in Y/Z model-size conditions"
- [ ] Create figure: profiling budget (x) vs. rank correlation with final result (y)

### Step 17: Human evaluation
- [ ] Distribute blinded outputs to evaluators
- [ ] Collect ratings
- [ ] Inter-annotator agreement (Cohen's kappa / Krippendorff's alpha)
- [ ] If clinician reviewed subset, report separately

**Deliverable:** All metrics computed, statistical tests done, figures created.

---

## Phase 5: Writing (Weeks 3-4)

### Step 18: LaTeX setup
- [ ] NeurIPS 2024 style file
- [ ] BibTeX with all references
- [ ] Placeholder sections

### Step 19: Write (in this order)
1. **Section 4** (Experiments): you have the data, write this first
2. **Section 5** (Results): include reversal plot, profiling figure, baselines comparison
3. **Section 3** (System Design): adapt from presentations, compress
4. **Section 2** (Related Work): tight, cite list from outline
5. **Section 1** (Introduction): the "so what?" paragraph citing practitioner defaults is critical
6. **Section 6** (Limitations): thorough and honest
7. **Sections 7-9** (Ethics, Reproducibility, Conclusion): short, direct
8. **Abstract**: dead last, 150-200 words

### Step 20: Create figures
Priority order:
1. [ ] Reversal plot (the paper's key visual)
2. [ ] Profiling protocol figure
3. [ ] Baselines + ablation table
4. [ ] Architecture diagram
5. [ ] Readability comparison
6. [ ] Latency table

### Step 21: Co-author review
- [ ] Draft to Kelsey + Dr. Sandrine, 3-5 days for feedback
- [ ] Revise, final sign-off

---

## Phase 6: Submission (Week 5)

### Step 22: Pre-submission
- [ ] All numbers verified against logs
- [ ] Figures legible in color and grayscale
- [ ] BibTeX complete
- [ ] Reproducibility statement written
- [ ] Repository prepared (scripts, prompts, adapter weights, logs)

### Step 23: Submit
- [ ] arXiv: PDF + source, cs.CL primary, cs.HC cross-list, CC BY 4.0

### Step 24: Post-submission
- [ ] Share link
- [ ] Post on Twitter/X, LinkedIn
- [ ] Open-source repo
- [ ] Check workshop deadlines for expanded version

---

## Contingency Plans

**Reversal disappears with seeds:**
Reframe as negative result: "config ordering is stable, contrary to initial observations." Still publishable with the profiling protocol and evaluation methodology.

**Profiling protocol doesn't predict final ranking:**
Report honestly. It means the reversal emerges late in training. That's an informative finding about training dynamics. Discuss implications for early stopping and rank selection.

**Full FT on 1B OOMs:**
Drop it. Use zero-shot and few-shot as your only baselines. Less powerful but the reversal finding still stands on its own.

**Can't get human evaluators:**
Rely on model-agnostic metrics + LLM judge. The reversal finding doesn't depend on human eval. Acknowledge in limitations.

**Training too slow:**
Priority cuts in this order: (1) drop secondary ablation, (2) reduce seeds to 2, (3) cut rank 2, (4) second task becomes eval-only (Option B). The primary ablation with 3 seeds is the last thing you cut.

**Readability scores don't differ between age groups:**
Report honestly. It means LIMA-style age targeting on this dataset size isn't effective. The reversal finding is unaffected. Discuss what this implies about synthetic data quality for age-targeted fine-tuning.

**Human-ref and Claude-ref metrics diverge:**
This is a result, not a failure. It demonstrates that synthetic evaluation doesn't align with human judgment for your domain. Discuss as evidence that human references are necessary even when synthetic data is used throughout.
