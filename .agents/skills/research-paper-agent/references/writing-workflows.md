# Academic Writing Workflows

## Literature review

1. Define scope and screening criteria.
2. Build the evidence matrix.
3. Group studies by problem, mechanism, and evaluation regime.
4. Compare agreements, conflicts, and methodological limitations.
5. End with a research gap that follows from the evidence and connects directly to this project's research question.

## Method section

Write enough detail to reconstruct the experiment: datasets and provenance, cohort/filtering, temporal split, preprocessing, sampling, model pool, ensemble rule, drift/weight mechanism, tuning boundary, baselines, metrics, statistical analysis, software, and seeds. Use equations only when they remove ambiguity.

## Results section

Report observations without causal interpretation. Name the evaluated split, sample or transition count, metric direction, uncertainty, missing runs, and exact result artifact. Present primary outcomes before exploratory analyses. Keep validation and test results visibly separate.

## Discussion section

Use this order:

1. answer the research question;
2. compare with relevant literature;
3. explain plausible mechanisms without overstating causality;
4. discuss practical meaning;
5. state limitations and threats to validity;
6. propose bounded future work.

## Peer review

Review in this priority order:

1. unsupported or incorrect claims;
2. leakage, invalid comparisons, or statistical-unit errors;
3. method/result inconsistency and reproducibility gaps;
4. missing limitations or overgeneralization;
5. organization, clarity, and style.

For each major issue, provide evidence, consequence, and a concrete revision. Do not rewrite the author's conclusions to be stronger than the data.

## Revision control

Preserve a traceability table mapping each reviewer comment to disposition, changed location, and rationale. Do not mark an issue resolved until the manuscript and its evidence agree.
