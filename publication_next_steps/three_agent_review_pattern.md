# Three-Agent Review Pattern: Reviewer + Writer + Validator

A reusable loop for hardening academic writing sections. Three agents run in rounds until all are satisfied. Use this whenever a section needs to be academically rigorous, factually grounded in project docs, and free of hallucinated claims.

---

## Agent Roles

### 1. Reviewer (antagonist)
Finds methodological weaknesses, logical gaps, unsupported claims, missing operationalizations, and poorly worded sentences. Does NOT check factual accuracy against source docs — that is the validator's job.

**Tell it:**
- What section type it is (methodology, results, related work, etc.)
- What framing choices are intentional and should not be challenged
- Which items are expected TODOs (do not flag these)
- To declare "essentially complete pending TODOs" when fewer than 3 substantive issues remain

### 2. Writer
Applies fixes from both the reviewer and validator in a single pass. Does NOT invent specific values — uses `[TODO: ...]` for anything the author must supply.

**Tell it:**
- The exact fixes to apply, numbered and precise
- The confirmed ground truth values available to fill in
- The full current text of the section
- To output the complete revised file

### 3. Validator
Checks every non-TODO factual claim against confirmed project documentation and known references. Catches hallucinated specifics, unresolved design decisions stated as facts, and citation misuse.

**Tell it:**
- The confirmed ground truth (actual configs, project docs, known reference details)
- Which items are already [TODO] — skip these
- To use two output categories: **A. Reference issues** and **B. Factual/hallucination issues**
- To say "No factual issues found — validation passed" if clean

---

## Loop Structure

```
Round N:
  ├── Reviewer (parallel)   → numbered critique list
  └── Validator (parallel)  → A/B issue list

  Writer (sequential, after both)  → revised file written to disk

Repeat until:
  - Reviewer declares "essentially complete pending TODOs"
  - Validator says "No factual issues found"
```

Reviewer and validator run **in parallel** (independent tasks, no shared context).
Writer runs **sequentially** after both, addressing all issues in one pass.

---

## Prompting Rules

### Reviewer prompt must include:
- Document context (what the paper is, what was actually done)
- List of confirmed ground truth relevant to the section
- Which framings are intentional (e.g., "declarative programming analogy — do not challenge")
- Instruction to skip [TODO] markers
- Prior round context: "X rounds of revision have already been applied — do not re-raise resolved issues"
- Stopping condition: "If fewer than 3 new substantive issues remain, declare essentially complete"

### Validator prompt must include:
- The full confirmed ground truth, organized by source (configs, project docs, known references)
- Explicit list of known-correct citations (author, year, venue, title)
- Instruction to skip [TODO] items
- Output format: A (reference issues) / B (factual issues), with quote + problem + correction for each
- Stopping condition: explicit "No factual issues found — validation passed" when clean

### Writer prompt must include:
- Numbered fixes — one per issue, with exact old text → new text where possible
- The confirmed values available to fill in (don't leave as TODO if the answer is known)
- Instruction not to invent values — use [TODO: specific description] for unknowns
- The complete current section text
- Instruction to output the complete revised file

---

## What Counts as Ground Truth for the Validator

The validator needs authoritative sources to check against. For this project:

| Source | What it confirms |
|--------|-----------------|
| `configs/*.yaml` | Optimizer, LR, schedule, batch size, iters, rank, scale, num_layers, dropout, mask_prompt, save_every, steps_per_eval, val_batches, seed, model IDs |
| `src/constants.py` | Model names, role names |
| `src/prepare_data.py` | Data format, role mapping, filtering logic |
| `scripts/train_*.sh` | Framework (mlx_lm.lora), config file names |
| `publication_next_steps/paper_outline_v4.md` | Study design, evaluation metrics, baselines, statistical methods |
| `docs_for_paper/prompts.md` | Data generation prompts, category names, safety rules, example counts |
| Known references in paper outline | Citation details for LoRA, LIMA, QLoRA, Biderman, etc. |

---

## Convergence Signals

The loop is done when:
- Reviewer: "fewer than 3 substantive issues remain" OR "essentially complete pending TODOs"
- Validator: "No factual issues found — validation passed"
- All resolvable TODOs have been filled in from ground truth

Remaining TODOs after convergence = genuine unknowns only the author can supply.

---

## Tips from Running This Pattern

- **Run reviewer and validator in parallel** in a single message — they are fully independent
- **Give the writer all fixes at once** — batching prevents re-reads and drift
- **Escalate ground truth rounds**: first rounds use project docs; later rounds use actual code/configs once located
- **The validator catches what the reviewer misses**: the reviewer thinks about logic; the validator thinks about facts. They complement, not duplicate, each other
- **Confirmed values eliminate whole classes of TODOs**: reading actual config files resolved ~7 TODOs in one pass that had persisted through 4 prior review rounds
- **Watch for cross-section consistency**: issues like the 500 vs. 475 example count discrepancy only appear when the validator reads the whole document, not just the section under review
- **Be explicit about tense**: methodology sections describe planned, in-progress, or completed work — the validator should flag when tense implies completion that hasn't happened (e.g., multi-seed ablation)
