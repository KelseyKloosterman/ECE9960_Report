# Markdown-to-LaTeX Conversion Agent Pattern

A two-agent pipeline for converting finalized paper section markdown files into correctly formatted `IEEEtai` LaTeX and integrating them into `paper/Paper.tex`. Run this whenever a section is ready to be committed to the paper.

---

## Why Two Agents (Not Three)

The Converter and its verification pass share the same context — the source markdown and the converted output are both already present. There is no benefit in handing off to a separate Auditor agent; it would re-derive exactly what the Converter can check for itself.

The Integrator stays separate because it writes to `Paper.tex`. Keeping the file write isolated means you can inspect the conversion output before anything is committed to disk. That separation is worth keeping.

---

## Pipeline Structure

```
Source: <section>.md  +  paper/Paper.tex (current state)
                      ↓
                  Converter
            (convert → self-verify)
                      ↓
              Validated LaTeX body
                      ↓
                  Integrator
                      ↓
              paper/Paper.tex (updated)
```

If the Converter's self-verification finds issues, it fixes them in the same pass before outputting. No separate loop agent required.

---

## Agent 1 — Converter

Translates the source markdown into valid `IEEEtai` LaTeX, then immediately runs a structured self-verification check on its own output. Outputs **only the section body** — no `\documentclass`, no preamble, no `\begin{document}`.

### Phase 1: Convert

Apply the following markdown → LaTeX mapping:

| Markdown | LaTeX |
|---|---|
| `# Section Title` | `\section{Section Title}` |
| `## Subsection Title` | `\subsection{Subsection Title}` |
| `### Subsubsection Title` | `\subsubsection{Subsubsection Title}` |
| `**bold text**` | `\textbf{bold text}` |
| `*italic text*` | `\textit{italic text}` |
| `` `inline code` `` | `\texttt{inline code}` |
| `$$...$$` display math | `\begin{equation}...\end{equation}` |
| `$...$` inline math | `$...$` (unchanged) |
| `- item` / `* item` bullet | `\begin{itemize}\item ...\end{itemize}` |
| `1. item` numbered list | `\begin{enumerate}\item ...\end{enumerate}` |
| Markdown table | `\begin{table}` block (see table rules below) |
| `[Author, Year]` citation | `\cite{citationKey}` using the key map below |
| `---` horizontal rule | remove (markdown-only separator) |

**Table format (IEEEtai):**

```latex
\begin{table}[t]
  \caption{Caption text.}
  \label{tab:tableN}
  \begin{tabular}{l c c c}
    \hline
    Col 1 & Col 2 & Col 3 & Col 4 \\
    \hline
    value & value & value & value \\
    \hline
  \end{tabular}
\end{table}
```

- Caption goes *above* the tabular (IEEE convention)
- Use `\hline` for top, header-separator, and bottom rules
- Label convention: `tab:table1`, `tab:table2`, etc.
- In-text reference: `Table~\ref{tab:table1}`

**Math format:**

- Labeled equations: `\begin{equation}\label{eq:name}...\end{equation}`
- Unlabeled displayed equations: `\[...\]`
- Multi-line aligned equations: `\begin{align}...\end{align}` with `&` markers
- Inline math: `$...$` unchanged

**IEEEtai section rules:**

- First paragraph of Introduction only: `\IEEEPARformat{F}{irst}` for the opening word drop cap. Do NOT use it anywhere else.
- No blank line between `\section{}` and the first paragraph.
- Abstract section: content goes inside `\begin{abstract}...\end{abstract}`, not a `\section{}`.

### Phase 2: Self-Verify

After producing the converted output, re-read the source markdown and verify the output against these five checks. Fix any issue found before outputting.

**Check 1 — Tables complete.** For every markdown table, count the rows (excluding header). Confirm the LaTeX `tabular` has the same row count. A dropped row is a silent error.

**Check 2 — Equations intact.** For every `$$...$$` block in the source, confirm all terms appear in the corresponding `\begin{equation}...\end{equation}`. Check superscripts, subscripts, and matrix dimension annotations (e.g., `\mathbb{R}^{d \times k}`).

**Check 3 — References mapped.** For every `[Author, Year]` citation in the source:
1. Confirm it has a corresponding `\cite{key}` in the output using the citation key map below.
2. Confirm that key appears in the "Known BibTeX keys" list below.
3. If a citation has no key in the map, or no BibTeX entry yet: output `\cite{key} % [TODO: add BibTeX entry]` — do not silently drop or leave as plain text.

**Check 4 — No content dropped.** Scan the source for any paragraph, sentence, or list item not present in the output. Flag any that are missing.

**Check 5 — Subsection hierarchy correct.** Confirm every `##` subsection in the source has a matching `\subsection{}` in the output, in the same order.

---

## Agent 2 — Integrator

Patches the validated LaTeX body into the correct location in `paper/Paper.tex`. Outputs the **complete updated `Paper.tex`** file.

**Instructions for the Integrator:**

- Replace only the content between the two bounding section markers for the target section (see boundary table below). Preserve the `\section{}` commands themselves.
- Do not touch any other section's content.
- Output the complete file (not a diff).

**Boundary table:**

| Section | Replace content between... |
|---|---|
| Abstract | `\begin{abstract}` and `\end{abstract}` |
| Introduction | `\section{Introduction}` and `\section{Related Work}` |
| Related Work | `\section{Related Work}` and `\section{Dataset}` |
| Dataset | `\section{Dataset}` and `\section{Methodology}` |
| Methodology | `\section{Methodlogy}` *(typo)* and `\section{Results}` — also fix the typo to `\section{Methodology}` |
| Results | `\section{Results}` and `\section{Discussion/Conclusion}` |
| Discussion/Conclusion | `\section{Discussion/Conclusion}` and `\section*{Appendix}` |
| Appendix | `\section*{Appendix}` and `\section*{References and Footnotes}` |

**Multi-section sessions:** When converting multiple sections in the same session, always pass the *current, already-updated* `Paper.tex` to each Integrator call — not the original. Using a stale version will overwrite previously integrated sections.

---

## Citation Key Map

| Markdown citation | `\cite{}` key | BibTeX entry exists? |
|---|---|---|
| `[Hu et al., 2022]` | `hu2022lora` | TODO |
| `[Dettmers et al., 2023]` | `dettmers2023qlora` | TODO |
| `[Biderman et al., 2024]` | `biderman2024lora` | TODO |
| `[Zhou et al., 2023]` | `zhou2023lima` | TODO |
| `[Dubey et al., 2024]` | `dubey2024llama3` | TODO |
| `[Mangrulkar et al., 2022]` | `mangrulkar2022peft` | TODO |
| `[Apple, 2023]` | `apple2023mlx` | TODO |
| `[Pedregosa et al., 2011]` | `pedregosa2011sklearn` | TODO |
| `[Lin, 2004]` | `lin2004rouge` | TODO |
| `[Zhang et al., 2020]` | `zhang2020bertscore` | TODO |
| `[Kincaid et al., 1975]` | `kincaid1975fk` | TODO |
| `[McLaughlin, 1969]` | `mclaughlin1969smog` | TODO |
| `[Gunning, 1952]` | `gunning1952fog` | TODO |
| `[Coleman and Liau, 1975]` | `coleman1975cli` | TODO |
| `[Bansal, 2023]` | `bansal2023textstat` | TODO |
| `[Inan et al., 2023]` | `inan2023llamaguard` | TODO |

> Update the "BibTeX entry exists?" column to `YES` as entries are added to `Paper.tex`.

---

## Converter Prompt Template

```
You are converting a finalized markdown section of an academic paper into valid IEEEtai LaTeX.

## Source markdown
<paste full content of section.md here>

## Current Paper.tex (for context on document structure and existing BibTeX keys)
<paste current paper/Paper.tex here>

## Your task

**Phase 1 — Convert** the markdown to LaTeX using these rules:
[paste the Phase 1 conversion rules from this doc]

**Phase 2 — Self-verify** your output against the source. Check all five items:
1. Table row counts match
2. All equation terms preserved
3. Every [Author, Year] citation mapped to \cite{key}; flag missing BibTeX entries
4. No paragraphs or sentences dropped
5. Subsection hierarchy matches

Fix any issue you find before outputting.

Output only the section body (no \documentclass or \begin{document}).
Use % [TODO: ...] comments for anything that requires author input.
```

---

## Integrator Prompt Template

```
You are integrating a validated LaTeX section body into paper/Paper.tex.

## Current paper/Paper.tex
<paste full current Paper.tex here>

## Validated LaTeX section body
<paste Converter output here>

## Target section
Section name: [e.g., Methodology]
Replace content between: \section{Methodlogy} and \section{Results}
Also fix the typo: \section{Methodlogy} → \section{Methodology}

Replace only the body between those two section markers.
Preserve the \section{} commands themselves (with the typo fix noted above).
Output the complete updated Paper.tex file.
```
