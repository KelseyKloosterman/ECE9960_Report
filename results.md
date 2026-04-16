# Results

This section reports results for the eight training runs described in Section 3.5: two adapter configurations (Standard, $r = 8$, `num_layers = 16`; Fast, $r = 4$, `num_layers = 8`) trained across two model sizes (1B, 3B) and two age groups (5–11, 12–18). All adapter results correspond to the final checkpoint at step 600. Evaluation covers the three tiers described in Section 3.6.1: readability, latency and throughput, and inter-role style separation. Rule-based safety pattern-matching results are also reported (Section 4.4) as an exploratory supplementary check; this evaluation was implemented in `src/evaluate.py` but is not among the completed tiers described in Section 3.6.1. Base model (no-adapter) results are included throughout as a zero-shot reference, providing an empirical lower bound that isolates the contribution of fine-tuning. Perplexity results from validation loss trajectories are available for the Fast adapter configurations; Standard adapter training logs were not retained, and Standard adapter perplexity therefore requires a separate held-out evaluation pass to report. The planned evaluations identified in Section 3.6.2 — ROUGE-L, BERTScore, LLM-as-judge, adversarial probe testing, and multi-seed statistical analysis — have not been conducted and are deferred to future work.

Throughout this section, configurations are referred to by their short designators: **Standard-1B**, **Standard-3B**, **Fast-1B**, **Fast-3B**, **Base-1B**, and **Base-3B**, with age group appended where relevant (e.g., Standard-3B-5to11). Standard and Fast adapters differ in both rank and layer count simultaneously (Section 3.3); performance differences between them cannot be attributed to either factor in isolation.

## 4.1 Training Dynamics

Validation loss was computed every 60 steps on the full age-matched held-out set (`val_batches: -1`) for the four Fast adapter runs, yielding eleven evaluation points per run at steps 1, 60, 120, 180, 240, 300, 360, 420, 480, 540, and 600. Training logs for the Standard adapter runs were not preserved; training dynamics for those configurations are therefore not reported.

All four Fast adapter loss curves show a rapid decline during the warmup phase (steps 1–100), where the learning rate ramps from zero to peak, followed by continued decrease as the cosine decay schedule reduces the step size toward zero. Table 1 summarizes the validation loss at initialization and at final step.

**Table 1. Fast adapter validation loss at initialization (step 1) and final step (step 600).**

| Configuration | Age Group | Val Loss (step 1) | Val Loss (step 600) | Change |
|---|---|---|---|---|
| Fast-1B | 5–11 | 3.211 | 1.744 | −1.467 |
| Fast-1B | 12–18 | 3.245 | 1.908 | −1.337 |
| Fast-3B | 5–11 | 3.023 | 1.728 | −1.295 |
| Fast-3B | 12–18 | 2.998 | 1.840 | −1.158 |

All configurations use the step-600 checkpoint as the final artifact per the fixed protocol described in Section 3.5. For Fast-1B-5to11, this checkpoint (val loss 1.744) is within 0.006 of the per-run minimum observed at step 480 (1.742), indicating that the protocol-mandated final checkpoint is effectively at the convergence point for this run. Fast-1B-12to18 similarly plateaus: loss is approximately flat from step 420 onward (step 420: 1.917, step 480: 1.902, step 540: 1.910, step 600: 1.908), with a slight uptick at step 540 that reverses by step 600; the plateau suggests this configuration converges early relative to the 600-step budget. Fast-3B runs show a more gradual descent continuing through step 600 (5to11: 1.734 at step 540 → 1.728 at 600; 12to18: 1.844 at 540 → 1.840 at 600), indicating the 3B model may benefit from a modestly longer training horizon.

No validation loss divergence was observed in any run. The absence of late-training divergence is consistent with the dataset quality and training protocol but cannot substitute for multi-seed replication in establishing convergence reliability.

Peak memory usage during training, as recorded at the final step, was 2.661 GB for Fast-1B-5to11, 3.481 GB for Fast-1B-12to18, 3.901 GB for Fast-3B-5to11, and 4.761 GB for Fast-3B-12to18 — all well within the 16 GB unified memory ceiling on the Mac Mini M4. The Fast-1B-12to18 run showing higher peak memory than Fast-1B-5to11 (3.481 vs. 2.661 GB) likely reflects longer average sequence lengths in the 12–18 training corpus. Trainable parameters are 1.409M (0.114% of 1.235B total) for all 1B fast runs and 1.737M (0.054% of 3.213B total) for all 3B fast runs, confirming the parameter-efficiency of the LoRA approach.

## 4.2 Perplexity

Perplexity is computed as $e^{L_{\text{val}}}$ where $L_{\text{val}}$ is the final validation cross-entropy loss on the age-matched 50-example held-out set. Each adapter is evaluated only on its target age group's validation subset, as described in Section 3.4. Perplexity is available for the four Fast adapter configurations from the training logs; Standard adapter perplexity requires a separate evaluation run and is not reported here.

**Table 2. Fast adapter perplexity (derived from validation loss at step 600).**

| Configuration | Age Group | Val Loss | Perplexity |
|---|---|---|---|
| Fast-1B | 5–11 | 1.744 | 5.72 |
| Fast-1B | 12–18 | 1.908 | 6.74 |
| Fast-3B | 5–11 | 1.728 | 5.63 |
| Fast-3B | 12–18 | 1.840 | 6.30 |

Within the Fast configuration, Fast-3B achieves lower perplexity than Fast-1B for both age groups: a difference of 0.09 perplexity points for the 5–11 adapter (5.63 vs. 5.72) and 0.44 points for the 12–18 adapter (6.30 vs. 6.74). The 12–18 adapters show higher perplexity than the 5–11 adapters across both model sizes, consistent with the 12–18 validation set containing longer, more syntactically complex responses that are harder to predict at the token level.

The perplexity advantage of Fast-3B over Fast-1B is modest for the 5–11 group (1.6% relative reduction) and more pronounced for the 12–18 group (6.5% relative reduction). Whether this difference reflects a meaningful capacity advantage of the 3B model or is within seed-level noise cannot be determined from single-seed runs; multi-seed replication would be required to characterize the variance. Standard adapter perplexity comparison — which would require running held-out evaluation against both standard adapter checkpoints — is identified as a near-term data collection priority.

As noted in Section 3.4, both the training and validation sets are drawn from the same generative process. Perplexity therefore measures in-distribution coherence with the synthetic data distribution rather than generalization to naturalistic pediatric queries.

## 4.3 Readability

Readability is evaluated on outputs generated from the age-matched 50-example held-out prompts using the `textstat` library (version 0.7.13). Five metrics are reported: Flesch-Kincaid (FK) grade level, SMOG index, Gunning Fog index, Coleman-Liau index, and lexical diversity measured by type-token ratio (TTR). For the 5–11 adapter, a pass/fail threshold of FK $\leq 7.0$ is applied. SMOG values are reported as computed; the metric requires a minimum of 30 sentences for full reliability, and individual short responses may contribute a zero floor that compresses the average.

Table 3 reports readability metrics for all six configurations, including the base model.

**Table 3. Readability metrics (outputs generated from age-matched held-out prompts, 50 examples per group).**

| Configuration | Age Group | FK Grade | SMOG | Gunning Fog | Coleman-Liau | TTR | Avg Words | FK ≤ 7.0 |
|---|---|---|---|---|---|---|---|---|
| Standard-1B | 5–11 | 6.1 | 8.8 | 7.7 | 6.7 | 0.712 | 74 | 36/50 (72%) |
| Standard-1B | 12–18 | 8.8 | 11.7 | 11.3 | 10.3 | 0.701 | 90 | — |
| Standard-3B | 5–11 | 5.5 | 8.5 | 7.4 | 6.3 | 0.724 | 75 | 42/50 (84%) |
| Standard-3B | 12–18 | 8.7 | 11.4 | 10.9 | 10.0 | 0.714 | 92 | — |
| Fast-1B | 5–11 | 5.5 | 8.4 | 7.2 | 6.2 | 0.701 | 72 | 41/50 (82%) |
| Fast-1B | 12–18 | 8.4 | 11.3 | 10.6 | 9.4 | 0.650 | 92 | — |
| Fast-3B | 5–11 | 5.9 | 8.9 | 7.9 | 6.6 | 0.636 | 99 | 38/50 (76%) |
| Fast-3B | 12–18 | 8.3 | 11.2 | 10.7 | 9.7 | 0.613 | 122 | — |
| Base-1B | 5–11 | 9.5 | 12.1 | 12.2 | 9.0 | 0.508 | 201 | 7/50 (14%) |
| Base-1B | 12–18 | 10.8 | 13.1 | 13.4 | 11.2 | 0.527 | 210 | — |
| Base-3B | 5–11 | 9.4 | 12.1 | 12.2 | 9.6 | 0.609 | 152 | 6/50 (12%) |
| Base-3B | 12–18 | 10.6 | 12.9 | 13.3 | 11.4 | 0.565 | 206 | — |

All four fine-tuned adapter configurations substantially outperform the base models on the FK $\leq 7.0$ target for the 5–11 group: pass rates of 72–84% for fine-tuned adapters versus 12–14% for the base models. The best-performing configuration for the 5–11 target is Standard-3B (42/50, 84%). All fine-tuned 5–11 adapters meet the FK $\leq 7.0$ target on average (FK 5.5–6.1 vs. the threshold of 7.0), though individual responses occasionally exceed the threshold — the highest single-response offender across all configurations was FK 10.0 (Standard-1B-5to11, generated for the query 'Can I watch TV?'), suggesting that responses to low-context queries may adopt more complex syntactic structures.

The difference in FK between fine-tuned and base model outputs is largely mediated by response length. Fine-tuned adapters generate substantially shorter responses (72–99 words average for 5–11 adapters; 90–122 for 12–18) compared to the base model (152–210 words). Shorter responses contain fewer clauses and less complex sentence structures, mechanically reducing formula-based readability scores. This conflation of conciseness with readability is a known limitation of FK and similar formula-based metrics; the consistent pattern across all five metrics — including Coleman-Liau, which uses character and sentence counts rather than syllable counts, providing a partially independent computational pathway — provides corroborating evidence that the improvement reflects genuine register adaptation rather than only a syllable-count artifact, though all five metrics remain sensitive to response length through different mechanisms.

The 12–18 adapters produce consistently higher FK grade levels than the corresponding 5–11 adapters across all four fine-tuned configurations (gap: 2.4–3.2 FK grade levels), confirming that the two adapters have learned distinct communication registers. FK grade levels of 8.3–8.8 correspond approximately to grades 8–9, which is consistent with the reading level expected for adolescents in the 12–18 target age range in a health information context; no hard upper threshold is applied for this group (Section 3.6.1). Base model outputs show a smaller register gap (1.2–1.3 FK points), indicating that the base model has some capacity to modulate style when given age-targeted prompts, but that fine-tuning sharpens this separation substantially.

A configuration-ordering crossover in readability is visible in Table 3: for the Standard configuration, the 3B adapter achieves a higher FK $\leq 7.0$ pass rate than the 1B adapter (84% vs. 72%) and lower average FK grade (5.5 vs. 6.1); for the Fast configuration, this ordering reverses — the 1B adapter achieves a higher pass rate (82% vs. 76%) and lower or equal average FK (5.5 vs. 5.9). This crossover is present in FK grade, FK pass rate, SMOG, Gunning Fog, and Coleman-Liau for the 5–11 age group, and is consistent across the formula-based metrics. It provides preliminary single-seed evidence that adapter configuration effects on readability are not monotone across model sizes — the rank/layer combination that produces better grade-level calibration on the 3B model does not do so on the 1B model, and vice versa. Multi-seed replication is required before this observation can be interpreted as a reliable effect. TTR for the 5–11 group shows the same directional crossover (Standard-1B: 0.712 vs. Standard-3B: 0.724, with 3B higher; Fast-1B: 0.701 vs. Fast-3B: 0.636, with 1B higher), consistent with the formula-based metric results; however, TTR measures lexical diversity rather than grade-level difficulty and is not included in the formula-based enumeration above.

## 4.4 Rule-Based Safety (Supplementary Exploratory Check)

Although rule-based safety evaluation is not among the completed tiers in Section 3.6.1 — comprehensive safety evaluation, including adversarial probe testing and a learned guard model, is deferred to future work as described in Section 3.6.2 — the evaluation pipeline (`src/evaluate.py`) includes a pattern-matching check that was run on all six configurations as an exploratory supplement. Each generated response is checked against regular-expression patterns for three violation categories: (1) unsupported medical diagnosis, (2) failure to escalate emergency queries to a healthcare professional, and (3) age-inappropriate content (graphic or frightening language; applied to the 5–11 adapter only). Results are reported for completeness.

**Table 4. Rule-based safety pass rates (percentage of responses with no matched violation pattern, 50 examples per group).**

| Configuration | Age Group | Diagnosis (%) | Escalation (%) | Age-Inappropriate (%) | All-Category (%) |
|---|---|---|---|---|---|
| Standard-1B | 5–11 | 100 | 100 | 100 | 100 |
| Standard-1B | 12–18 | 100 | 100 | — | 100 |
| Standard-3B | 5–11 | 100 | 100 | 100 | 100 |
| Standard-3B | 12–18 | 100 | 100 | — | 100 |
| Fast-1B | 5–11 | 100 | 100 | 100 | 100 |
| Fast-1B | 12–18 | 100 | 100 | — | 100 |
| Fast-3B | 5–11 | 100 | 100 | 100 | 100 |
| Fast-3B | 12–18 | 100 | 100 | — | 100 |
| Base-1B | 5–11 | 100 | 100 | 100 | 100 |
| Base-1B | 12–18 | 100 | 100 | — | 100 |
| Base-3B | 5–11 | 100 | 100 | 100 | 100 |
| Base-3B | 12–18 | 100 | 100 | — | 100 |

All six configurations — including the unadapted base models — achieve a 100% all-category pass rate across all 50 responses per configuration. No violations were observed for any category in any configuration.

The 100% pass rate for the base models is notable: it indicates that the Llama 3.2 Instruct base models, even without fine-tuning, do not produce the specific violation patterns enumerated in the rule set when responding to these evaluation prompts. The instruct-tuning of the base models appears to provide a baseline level of safety-relevant behavior that fine-tuning does not degrade.

As noted in Section 3.6.2: a 100% pass rate indicates absence of the specific patterns enumerated in `src/evaluate.py`, not a comprehensive safety certification. The pattern set targets identifiable failure modes — explicit diagnostic language, absent emergency escalation, overtly graphic content — but cannot detect subtler violations such as ambiguous clinical framing, underspecified referrals, or indirect age-inappropriateness. Deployment-level safety requires additional layers beyond adapter-level pattern matching.

## 4.5 Latency and Throughput

Inference latency and throughput are measured sequentially (one prompt at a time, no batching) on the Mac Mini M4, reflecting the single-query deployment mode of the VR system. Per-response metrics recorded are total generation latency, total tokens generated, and tokens per second. All inference uses greedy decoding (temperature 0.0) with no system prompt, matching training conditions. The 1.0-second average latency target reflects real-time VR interaction requirements.

**Table 5. Inference latency and throughput (50 prompts per age group, sequential, Mac Mini M4).**

| Configuration | Age Group | Avg Latency (s) | Min (s) | Max (s) | Avg Tokens | Tok/s | Under 1.0s |
|---|---|---|---|---|---|---|---|
| Standard-1B | 5–11 | 1.09 | 0.66 | 1.69 | 89 | 81.7 | 21/50 (42%) |
| Standard-1B | 12–18 | 1.36 | 0.62 | 2.02 | 114 | 83.7 | 3/50 (6%) |
| Standard-3B | 5–11 | 2.37 | 1.31 | 3.45 | 92 | 38.8 | 0/50 (0%) |
| Standard-3B | 12–18 | 2.94 | 1.75 | 5.43 | 116 | 39.3 | 0/50 (0%) |
| Fast-1B | 5–11 | 0.93 | 0.50 | 1.40 | 88 | 94.3 | 35/50 (70%) |
| Fast-1B | 12–18 | 1.20 | 0.75 | 1.79 | 115 | 95.4 | 12/50 (24%) |
| Fast-3B | 5–11 | 2.83 | 1.31 | 6.80 | 120 | 41.9 | 0/50 (0%) |
| Fast-3B | 12–18 | 3.63 | 0.99 | 7.33 | 153 | 41.7 | 1/50 (2%) |
| Base-1B | 5–11 | 2.01 | 0.57 | 2.48 | 247 | 122.3 | 2/50 (4%) |
| Base-1B | 12–18 | 2.14 | 0.25 | 2.46 | 265 | 121.8 | 4/50 (8%) |
| Base-3B | 5–11 | 3.99 | 0.66 | 6.38 | 190 | 46.5 | 3/50 (6%) |
| Base-3B | 12–18 | 5.39 | 0.67 | 6.29 | 264 | 48.7 | 1/50 (2%) |

Fast-1B is the only configuration that achieves sub-1.0-second average latency for the 5–11 adapter (0.93s avg, 70% of responses under target). All other configurations exceed the 1.0-second average threshold, with all 3B configurations averaging 2.37–3.63 seconds for fine-tuned adapters and 3.99–5.39 seconds for the base model. For the Standard configuration, the latency gap between 1B and 3B is primarily attributable to the approximately 2.0–2.5× difference in throughput (81–95 tok/s for 1B vs. 38–42 tok/s for 3B fine-tuned adapters); response lengths are comparable across model sizes within each age group for the Standard adapter (5–11: 89 vs. 92 avg tokens for 1B vs. 3B; 12–18: 114 vs. 116 avg tokens). For the Fast-3B configuration, however, the dominant latency driver is response length rather than throughput, as discussed below. (Note: Table 3 reports average word count using whitespace tokenization, while Table 5 reports average token count using the model's subword tokenizer; the two metrics differ by a typical factor of 1.2–1.3 for these responses and are not directly comparable.)

A counterintuitive result appears in the Fast-3B configuration: Fast-3B is slower than Standard-3B despite having fewer trainable adapter parameters (rank 4/8 layers vs. rank 8/16 layers). The explanation lies in response length: Fast-3B generates substantially longer responses than Standard-3B (120/153 vs. 92/116 avg tokens for 5to11/12to18), while throughput in tokens per second is comparable (41.7–41.9 vs. 38.8–39.3). The output length increase — approximately 30% more tokens per response — overrides any latency benefit from the smaller adapter. This indicates that the rank-and-layer reduction in the Fast configuration altered the model's output length behavior in addition to its throughput characteristics; the mechanism is not fully understood and is noted as a finding for investigation in future work.

The base models generate responses substantially longer than fine-tuned adapters (190–265 avg tokens for base models vs. 88–153 avg tokens for fine-tuned adapters) while achieving higher throughput (122 tok/s for Base-1B vs. 82–95 for fine-tuned 1B), resulting in net latency that is substantially higher than fast fine-tuned adapters despite the throughput advantage. Fine-tuning reduces net latency primarily through substantially shorter responses. This benefit partially offsets the throughput reduction observed in fine-tuned vs. base models (82–95 tok/s vs. 122 tok/s for 1B; cause not confirmed), which works against deployment efficiency. The net latency improvement is therefore response-length-driven: shorter responses outweigh the throughput penalty in total generation time.

Maximum latency values for Fast-3B are notably high: 6.80s for a single 5-11 response and 7.33s for a single 12-18 response. The cause of these outliers is not confirmed from current data; one candidate explanation is responses that approached the generation length ceiling; they represent worst-case latency on specific prompts and would be perceptible as substantial lag in a VR interaction.

## 4.6 Inter-Role Style Separation

A TF-IDF + logistic regression classifier is trained on the combined outputs of both age-group adapters using 5-fold cross-validation (scikit-learn). Each adapter is run on its own age-matched 50-prompt held-out set with no system prompt, matching training conditions (`data/age_5_11/valid.jsonl` for the 5–11 adapter; `data/age_12_18/valid.jsonl` for the 12–18 adapter). Because each adapter is evaluated on its own age-matched prompts rather than a shared neutral prompt set, the classifier eliminates system-prompt vocabulary as a confound but does not fully eliminate age-matched instruction vocabulary: the 5–11 prompts may carry child-oriented vocabulary that propagates into response text, and similarly for 12–18 prompts, giving the classifier a potential signal source beyond the adapter-encoded register. A stronger design would run both adapters on an identical shared prompt set; the current design represents the practical constraint of age-matched validation data. Classification accuracy is the fraction of responses correctly assigned to their age group (5–11 vs. 12–18) across folds.

**Table 6. Inter-role TF-IDF + LR classification accuracy (5-fold CV; chance = 0.50).**

| Configuration | Classification Accuracy |
|---|---|
| Standard-1B | 0.890 ± 0.102 |
| Standard-3B | 0.920 ± 0.051 |
| Fast-1B | 0.940 ± 0.073 |
| Fast-3B | 0.900 ± 0.077 |
| Base-1B | 0.660 ± 0.097 |
| Base-3B | 0.700 ± 0.063 |

All four fine-tuned configurations achieve classification accuracy in the strong-separation range (≥ 0.89) defined in Section 3.6.1. The base models achieve substantially lower accuracy (0.660 for Base-1B; 0.700 for Base-3B), falling below or at the threshold for moderate separation and far below the fine-tuned configurations. This contrast confirms that fine-tuning is the primary driver of lexically and syntactically distinct age registers: the base Llama 3.2 Instruct models produce responses to their respective age-matched prompts that a surface-level classifier can separate at only chance-to-moderate accuracy, while fine-tuned adapters imprint register differences that are reliably detectable.

The ordering of fine-tuned configurations by accuracy shows a crossover consistent with the readability finding in Section 4.3: for the Standard configuration, the 3B adapter achieves higher classification accuracy than the 1B adapter (0.920 vs. 0.890); for the Fast configuration, this ordering reverses — the 1B adapter achieves higher accuracy than the 3B adapter (0.940 vs. 0.900). The Fast-1B configuration achieves the highest classification accuracy of any configuration (0.940), with the Standard-3B configuration second (0.920). The differences are small in absolute terms (0.020–0.030 accuracy points within each configuration type) and overlap substantially given the standard deviations, so this cannot be treated as a robust effect from single-seed runs; it is consistent with the readability crossover observation but not independently confirmatory.

The Standard-1B run shows the highest fold-to-fold variance (± 0.102). This CV variance reflects sampling variability across data partitions for the fixed trained model — it does not measure training instability across seeds, which would require multiple independent training runs. Whether the high fold variance reflects genuine instability in the adapter-encoded register or is an artifact of the small per-fold sample size (20 total responses per fold in a 5-fold split of 100 total, approximately 10 per class) cannot be determined from these results.

## 4.7 Discussion

### Standard vs. Fast Adapter

Across readability and style separation, the Standard and Fast configurations produce broadly similar outcomes: all achieve strong inter-role style separation (≥ 0.89), and all 5–11 adapters produce average FK grade levels well below the 7.0 threshold. The most salient systematic difference is in latency, where the Fast-1B configuration meets the 1.0-second deployment target for the 5–11 adapter (0.93s avg) while Standard-1B does not (1.09s avg), a 15% reduction in average latency with a 28-percentage-point improvement in per-response under-target rate (70% vs. 42%). This latency advantage makes Fast-1B the recommended configuration for the VR deployment use case.

The performance similarity between Standard and Fast on most metrics suggests that register adaptation for this dataset and task is achievable at rank 4 with 8 layers, consistent with the LIMA-hypothesis framing: high-quality, constrained training data reduces the adaptation capacity required to produce measurable style change. The counterintuitive latency behavior of Fast-3B (slower than Standard-3B) adds a caveat: reducing adapter capacity does not guarantee improved deployment efficiency when the adapter also influences output length.

It is important to note the structural limitation of the Standard vs. Fast comparison: rank and layer count differ simultaneously, making it impossible to attribute any observed difference to either variable. The Standard-1B configuration additionally saturates the entire 1B model depth (16/16 layers), making it qualitatively different in adaptation coverage from Standard-3B (16/28 layers, 57%). These confounds are discussed in Section 3.3 and limit the interpretive scope of any Standard-vs-Fast comparison.

### Model Scale Effects

The 3B configurations generally outperform the 1B configurations on readability and style separation, with the notable exception of the Fast adapter type discussed below; for the Fast configuration specifically, Fast-3B also achieves lower perplexity than Fast-1B (Standard adapter perplexity is not available for comparison), but at a latency cost that makes 3B unsuitable for the real-time VR deployment target. Standard-3B achieves the best 5–11 FK pass rate (84% vs. 72% for Standard-1B). For style separation, Standard-3B (0.920) outperforms Standard-1B (0.890).

However, the improvement of 3B over 1B is not uniform across adapter type: for the Fast configuration, 1B matches or outperforms 3B on FK pass rate (82% vs. 76%) and style separation accuracy (0.940 vs. 0.900). This reversal — 3B better than 1B on Standard, 1B better than 3B on Fast for readability and separation — is the central empirical observation of the current evaluation. It is discussed further in the following subsection.

### Readability and Style Crossover Observation

The most striking result in Tables 3 and 6 is a configuration-ordering crossover across model sizes. For the 5–11 adapter:

- **Standard configuration:** Standard-3B outperforms Standard-1B on FK pass rate (84% vs. 72%), average FK grade (5.5 vs. 6.1), and style separation accuracy (0.920 vs. 0.890).
- **Fast configuration:** Fast-1B outperforms Fast-3B on FK pass rate (82% vs. 76%), average FK grade (5.5 vs. 5.9), and style separation accuracy (0.940 vs. 0.900).

The same reversal is present across all five readability formula metrics (FK, SMOG, Gunning Fog, Coleman-Liau) for the 5–11 group, and in the style separation accuracy direction. A practitioner selecting the Standard adapter for a 1B deployment based on its 3B performance — or selecting the Fast adapter for a 3B deployment based on its 1B performance — would in both cases obtain the configuration that performs worse for their actual target model.

This finding is interpreted in the context of the paper's central hypothesis: the benefit of higher-rank, deeper-layer adaptation (Standard) may be model-capacity-dependent. The 3B model has sufficient representational capacity to exploit the additional degrees of freedom provided by rank 8 and 16-layer coverage; the 1B model may not, and may instead benefit from the implicit regularization of a smaller adapter. Alternatively, the 1B Standard configuration's full-layer coverage (100% of layers) may induce qualitatively different fine-tuning dynamics than partial-layer coverage, confounding the rank effect. These hypotheses cannot be distinguished from the current single-seed, two-point comparison; a controlled rank sweep ($r \in \{2, 4, 8, 16\}$) with layer count held constant and multiple seeds per configuration — as described in Section 3.6.2 — would be required to characterize the effect.

The crossover is not statistically confirmed. With a single seed per configuration and no variance estimates, the observed differences in FK pass rate (6–12 percentage points across the two crossover comparisons) and classification accuracy (0.020–0.030 points) may be within seed-level noise. This is flagged as the key open question motivating the planned multi-seed ablation.

### Limitations of Current Results

Four structural limitations constrain the conclusions that can be drawn from the results reported above.

First, **single-seed runs.** All eight training runs use seed 42. No variance estimates are available for any metric. The crossover observation in Sections 4.3 and 4.6 cannot be confirmed as a statistically reliable effect; it is a preliminary signal. The planned multi-seed rank ablation (Section 3.6.2) is the primary mechanism for establishing whether the effect is reproducible.

Second, **no Standard adapter perplexity.** Training logs were not retained for the Standard adapter runs. Perplexity — the primary metric for the rank-performance crossover hypothesis — is therefore only available for the Fast configurations, limiting the quantitative comparison between Standard and Fast. Running a held-out perplexity evaluation on the Standard adapter checkpoints would fill this gap at low additional cost.

Third, **in-distribution evaluation only.** All evaluation uses the same synthetic data distribution as training. Perplexity and readability scores measure in-distribution coherence, not naturalistic generalization. The base model comparison provides a useful lower bound within this distribution, but does not establish how fine-tuned adapters would perform on pediatric queries outside the training distribution. Human-written references, LLM-as-judge evaluation, and adversarial probing — all deferred to future work — would provide evidence on this question.

Fourth, **scope of safety evaluation.** The 100% rule-based safety pass rate covers only the specific violation patterns enumerated in the evaluation code. It does not certify absence of subtler safety-relevant failures. The base model's 100% pass rate on the same pattern set demonstrates that fine-tuning is not the primary mechanism ensuring these specific safety properties on these evaluation prompts.
