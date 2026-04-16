# Results

This section reports results for the eight training runs described in Section 3.5: two adapter configurations (Standard, $r = 8$, `num_layers = 16`; Fast, $r = 4$, `num_layers = 8`) across two model sizes (1B, 3B) and two age groups (5–11, 12–18). All results correspond to the final checkpoint at step 600. Base model (no-adapter) results are included as a zero-shot reference.

## 4.1 Training Dynamics and Perplexity

All four Fast adapter runs converge without divergence. Table 1 summarizes validation loss and derived perplexity ($e^{L_\text{val}}$) at the final step. Fast-3B achieves lower perplexity than Fast-1B for both age groups (5–11: 5.63 vs. 5.72; 12–18: 6.30 vs. 6.74). The 12–18 adapters show consistently higher perplexity than their 5–11 counterparts, consistent with more syntactically complex responses being harder to predict. Standard adapter perplexity requires a separate evaluation run and is not reported here.

**Table 1. Fast adapter validation loss and perplexity at step 600.**

| Configuration | Age Group | Val Loss (step 1) | Val Loss (step 600) | Perplexity |
|---|---|---|---|---|
| Fast-1B | 5–11 | 3.211 | 1.744 | 5.72 |
| Fast-1B | 12–18 | 3.245 | 1.908 | 6.74 |
| Fast-3B | 5–11 | 3.023 | 1.728 | 5.63 |
| Fast-3B | 12–18 | 2.998 | 1.840 | 6.30 |

## 4.2 Readability

Table 2 reports readability metrics for all six configurations. All four fine-tuned adapters substantially outperform the base models on the FK ≤ 7.0 target for the 5–11 group (72–84% pass rate vs. 12–14% for base models), driven largely by shorter response length (72–99 avg words vs. 152–210 for base models). The 12–18 adapters produce consistently higher FK grade levels than the 5–11 adapters (gap: 2.4–3.2 FK points), confirming that the two adapters have learned distinct communication registers.

A **configuration-ordering crossover** is visible in Table 2: for the Standard configuration, Standard-3B outperforms Standard-1B on FK ≤ 7.0 pass rate (84% vs. 72%); for the Fast configuration, this ordering reverses — Fast-1B outperforms Fast-3B (82% vs. 76%). This crossover is consistent across all five readability metrics for the 5–11 group and is the central empirical observation of this evaluation.

**Table 2. Readability metrics (outputs from age-matched held-out prompts, 50 examples per group).**

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

## 4.3 Rule-Based Safety

All six configurations — including the unadapted base models — achieved 100% pass rates across all three rule-based violation categories (unsupported diagnosis, failure to escalate, age-inappropriate content), indicating that the Llama 3.2 Instruct base models' safety-aligned behavior is preserved through fine-tuning. This probe detects only enumerated patterns and is not a comprehensive safety certification.

## 4.4 Latency and Throughput

Table 3 reports inference latency for all configurations (sequential, no batching, Mac Mini M4). **Fast-1B is the only configuration meeting the 1.0-second average latency target** for the 5–11 adapter (0.93s avg, 70% of responses under target). All 3B configurations exceed 2.3 seconds on average. Counterintuitively, Fast-3B is slower than Standard-3B despite fewer adapter parameters: Fast-3B generates ~30% more tokens per response (120/153 vs. 92/116 avg tokens), overriding any throughput benefit from the smaller adapter.

**Table 3. Inference latency and throughput (50 prompts per age group, sequential, Mac Mini M4).**

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

## 4.5 Inter-Role Style Separation

Table 4 reports TF-IDF + logistic regression classification accuracy (5-fold CV; chance = 0.50) for distinguishing 5–11 vs. 12–18 adapter outputs. All fine-tuned configurations achieve strong separation (0.89–0.94); base models reach only moderate separation (0.66–0.70), confirming that fine-tuning is the primary driver of distinct age registers. The same crossover observed in readability is present here: Standard-3B outperforms Standard-1B (0.920 vs. 0.890), while Fast-1B outperforms Fast-3B (0.940 vs. 0.900).

**Table 4. Inter-role TF-IDF + LR classification accuracy (5-fold CV; chance = 0.50).**

| Configuration | Classification Accuracy |
|---|---|
| Standard-1B | 0.890 ± 0.102 |
| Standard-3B | 0.920 ± 0.051 |
| Fast-1B | 0.940 ± 0.073 |
| Fast-3B | 0.900 ± 0.077 |
| Base-1B | 0.660 ± 0.097 |
| Base-3B | 0.700 ± 0.063 |

## 4.6 Discussion

**Fast-1B is the recommended configuration** for VR deployment: it is the only configuration meeting the 1.0-second latency target (0.93s avg) while maintaining strong readability (FK 5.5 avg) and style separation (0.940 accuracy). The similarity between Standard and Fast on most quality metrics suggests that register adaptation for this task is achievable at rank 4 with 8 adapted layers, consistent with the LIMA-hypothesis framing: high-quality constrained training data reduces the adaptation capacity required to produce measurable style change.

The **crossover observation** — Standard adapter better on 3B, Fast adapter better on 1B for both readability and style separation — is the central empirical finding. We tentatively interpret this as a capacity-regularization effect: the 1B model's limited representational capacity may benefit from the implicit regularization of a smaller adapter, while the 3B model can exploit the additional degrees of freedom of rank 8. However, since rank and layer count co-vary across configurations and all runs use a single seed, this crossover is not statistically confirmed and requires a controlled rank sweep with multiple seeds to establish.

Four structural limitations apply: (1) single-seed runs — no variance estimates for any metric; (2) no Standard adapter perplexity — training logs were not retained; (3) in-distribution evaluation only — both training and validation data share the same generative process; (4) rule-based safety evaluation covers only three enumerated patterns and is not a comprehensive certification.
