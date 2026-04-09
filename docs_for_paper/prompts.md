# Appendix: Data Generation Pipeline

Project Spaces -> Opus 4.6, extended thinking mode march 12.

## Creating Validation Data for Role Adaptive LLM

**System prompt:**

Project Goal This project space is dedicated to generating high-quality synthetic validation data for two pediatric-facing adapter models (ages 5–11 and 12–18) built on top of a base LLM, intended for deployment in a virtual recreation of Victoria Hospital in London, Ontario. The validation set represents 10% of the training data — 10 examples per age range per category, 100 examples total. Validation examples must be independent from the training set. They should reflect the same quality standards and factual grounding as the training data without reusing or closely mirroring any specific training examples. Following the LIMA hypothesis, every example must meet a gold standard of quality. A weak validation example is worse than no example — it produces misleading metrics. Training data is organized into five categories, each handled in its own dedicated chat: What to Expect — procedures, equipment, sensory experiences Who Are These People? — hospital roles explained age-appropriately Hospital Rules & Routines — visiting hours, daily schedules, what to bring Emotional Reassurance — comforting responses to fear, nervousness, or confusion (non-diagnostic) FAQs / General Curiosity — open-ended exploratory questions a child might ask Reference artifacts such as the Victoria Hospital children's patient and family guide are attached to provide factual grounding and ensure outputs are specific to this facility.

**Instructions:**

All chats in this project generate synthetic validation data for two pediatric-facing adapter models targeting ages 5–11 and 12–18, for deployment in a virtual recreation of Victoria Hospital in London, Ontario.

Output format: Every generated example must strictly follow this JSON format:
`{"instruction": "...", "response": "...", "role": "5-11|12-18", "category": "..."}`
Use "role": "5-11" for examples exclusively appropriate for younger children, "role": "12-18" for teens, and "role": "5-11|12-18" only when the content is genuinely appropriate for both groups without modification.

Quality standards: This dataset follows the LIMA hypothesis — prioritize quality over quantity. Every example should be something you would be proud to show as a gold standard. Avoid generic, vague, or repetitive responses.

Independence: These examples are for validation purposes. Do not reuse or closely mirror any specific examples from the training set. Examples must be diverse within their category and independent enough to serve as a meaningful held-out evaluation set.

Hard safety rules:
- Never generate a response that makes or implies a diagnosis
- Never recommend medications or treatments
- Never use language inappropriate for the target age range
- Always redirect medical or emergency questions to a real healthcare professional or parent/guardian

Grounding: Where possible, ground responses in the attached Victoria Hospital reference artifacts. Do not fabricate hospital-specific details that are not supported by the provided documents.

**Chats with prompts (5 chats, 1 for each category):**

chat 1 (FAQ validation examples by age group)

Prompt: Generate 20 validation examples for the faqs_general_curiosity category. Produce 10 examples tagged "role": "5-11" and 10 tagged "role": "12-18". Output each example as a JSON object on its own line.

chat 2 (Emotional reassurance validation examples)

Prompt: Generate 20 validation examples for the emotional_reassurance category. Produce 10 examples tagged "role": "5-11" and 10 tagged "role": "12-18". Output each example as a JSON object on its own line.

chat 3 (Hospital rules and routines validation examples)

Prompt: Generate 20 validation examples for the hospital_rules_and_routines category. Produce 10 examples tagged "role": "5-11" and 10 tagged "role": "12-18". Output each example as a JSON object on its own line.

chat 4 (Validation examples for age-specific examples)

Prompt: Generate 20 validation examples for the what_to_expect category. Produce 10 examples tagged "role": "5-11" and 10 tagged "role": "12-18". Output each example as a JSON object on its own line.

chat 5 (Validation examples for who_are_these_people category)

Prompt: Generate 20 validation examples for the who_are_these_people category. Produce 10 examples tagged "role": "5-11" and 10 tagged "role": "12-18". Output each example as a JSON object on its own line.

---

## Creating Training Data for Role Adaptive LLM

**System prompt:**

This project space is dedicated to generating high-quality synthetic training data for two pediatric-facing adapter models (ages 5–11 and 12–18) built on top of a base LLM, intended for deployment in a virtual recreation of Victoria Hospital in London, Ontario. The adapters are designed to help pediatric patients become acquainted with the hospital environment in a safe, age-appropriate, and reassuring way.
Following the LIMA hypothesis, the goal is to produce a small set of carefully curated, high-quality examples rather than a large noisy corpus. Each category will contain 100 examples, for a total of 500 examples per adapter.
Training data is organized into five categories, each handled in its own dedicated chat:
- What to Expect — procedures, equipment, sensory experiences
- Who Are These People? — hospital roles explained age-appropriately
- Hospital Rules & Routines — visiting hours, daily schedules, what to bring
- Emotional Reassurance — comforting responses to fear, nervousness, or confusion (non-diagnostic)
- FAQs / General Curiosity — open-ended exploratory questions a child might ask

Reference artifacts such as the Victoria Hospital children's patient and family guide will be attached to provide factual grounding and ensure outputs are specific to this facility. All generated examples will be reviewed for quality, safety, and age-appropriateness before inclusion in the final dataset.

**Instructions:**

All chats in this project generate synthetic training data for two pediatric-facing adapter models targeting ages 5–11 and 12–18, for deployment in a virtual recreation of Victoria Hospital in London, Ontario.

Output format: Every generated example must strictly follow this JSON format:
`{"instruction": "...", "response": "...", "role": "5-11|12-18", "category": "..."}`
Use "role": "5-11" for examples exclusively appropriate for younger children, "role": "12-18" for teens, and "role": "5-11|12-18" only when the content is genuinely appropriate for both groups without modification.

Quality standards: This dataset follows the LIMA hypothesis — prioritize quality over quantity. Every example should be something you would be proud to show as a gold standard. Avoid generic, vague, or repetitive responses.

Hard safety rules:
- Never generate a response that makes or implies a diagnosis
- Never recommend medications or treatments
- Never use language inappropriate for the target age range
- Always redirect medical or emergency questions to a real healthcare professional or parent/guardian

Grounding: Where possible, ground responses in the attached Victoria Hospital reference artifacts. Do not fabricate hospital-specific details (room numbers, staff names, policies) that are not supported by the provided documents.

**Chats with prompts (5 chats, 1 for each category):**

chat 1 (Training examples for emotional reassurance)

Prompt: Generate 200 training examples for the emotional_reassurance category. Ensure every example that involves a situation requiring medical attention redirects to a nurse, doctor, or parent/guardian. Produce 100 examples tagged "role": "5-11" and 100 tagged "role": "12-18". Output each example as a JSON object on its own line.

chat 2 (Victoria Hospital FAQ training examples generation)

Prompt: Generate 200 training examples for the faqs_general_curiosity category. Use the attached Victoria Hospital reference artifacts where relevant to keep responses grounded in this specific facility. Produce 100 examples tagged "role": "5-11" and 100 tagged "role": "12-18". Output each example as a JSON object on its own line.

chat 3 (Victoria Hospital training examples generation) - hospital rules and routines

Prompt: Generate 200 training examples for the hospital_rules_and_routines category. Prioritize the attached Victoria Hospital reference artifacts as the source of truth for policies, schedules, and routines. Produce 100 examples tagged "role": "5-11" and 100 tagged "role": "12-18". Output each example as a JSON object on its own line.

chat 4 (Victoria Hospital training examples generation) - who are these people

Prompt: Generate 200 training examples for the who_are_these_people category. Use the attached Victoria Hospital reference artifacts to ground roles and descriptions in this specific facility where possible. Produce 100 examples tagged "role": "5-11" and 100 tagged "role": "12-18". Output each example as a JSON object on its own line.

chat 5 (Victoria Hospital training examples generation) - what to expect

Prompt: Generate 200 training examples for the what_to_expect category. Use the attached Victoria Hospital reference artifacts to ground responses in this specific facility. Produce 100 examples tagged "role": "5-11" and 100 tagged "role": "12-18". Output each example as a JSON object on its own line.

---

## Code Generation with Claude Code

Claude /status:

```
Version: 2.1.74
Session name: /rename to add a name
Session ID: 70238da2-5a31-4bb5-a87c-d34656424d5e
cwd: /Users/ab/projects/small-bear
Login method: Claude Pro Account
Organization: sorrab1999@gmail.com's Organization
Email: sorrab1999@gmail.com
Model: Default Sonnet 4.6 · Best for everyday tasks
MCP servers: claude.ai Gmail △, claude.ai Google Calendar △
Memory: project (CLAUDE.md), auto memory (~/.claude/projects/-Users-ab-projects-small-bear/memory/MEMORY.md)
Setting sources: Project local settings
```

New repo prompt markdown file content are in another tab in this space. Will include in the appendix of our paper.