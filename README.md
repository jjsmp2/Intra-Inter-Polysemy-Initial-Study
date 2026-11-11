# Inter- and Intra-Domain Polysemy in Software Requirements: Sentence-Level Prototype

Code, benchmark and results for the REFSQ 2027 Research Preview
*Distinguishing Inter- and Intra-Domain Polysemy in Software Requirements:
Early Lessons from a Sentence-Level Prototype*.

Maintainer: Siba Mishra, C. V. Raman Global University, Bhubaneswar.
(For the double-blind review copy, run `./make_anonymous.sh`; it removes this line and the git history.)

## What the prototype does
For a target word in a requirement sentence it retrieves WordNet 3.0 senses, ranks them by TF-IDF
similarity to the sentence, and combines sense competition (Spread), polysemy (Poly) and domain
diversity (Div) into one score. The type (Inter-/Intra-Domain) is read from the domains of the two best senses.

## Layout
| Path | Content |
|---|---|
| `src/` | `ambiguity_framework.py` (pipeline and score), `domain_glossary.py` (six domains, 36-term glossary), `evaluate.py` (metrics, tuning) |
| `data/gold_standard_250.csv` | author-written 250-item benchmark (110 Inter, 64 Intra, 76 None; 120 target words) |
| `experiments/` | one script per result of the paper |
| `results/` | CSV files and logs written by the experiments |
| `docs/` | annotation guidelines and benchmark notes |
| `archive/` | earlier files that are **not** part of the paper |

## Reproduce
```bash
./reproduce.sh
```
| Paper element | Script | Output |
|---|---|---|
| Table 1 (grouped split, seed 42) and ten-split column | `experiments/grouped_split.py` | `results/grouped_split_results.csv`, `grouped_split_repeats.csv` |
| Tuned baselines on the earlier random split | `experiments/tune_baselines.py` | `results/tuned_baselines.csv` |
| Vocabulary sensitivity (F1 0.721) | `experiments/grouped_split_allvocab.py` | `results/*_allvocab.csv` |
| Why recall is low (24 / 13 / 29 items) | `experiments/recall_analysis.py` | `results/recall_analysis_log.txt` |
| Mapping check (10 of 29 senses) | `experiments/evaluate_domain_mapping.py` | `results/domain_mapping_*` |
| Worked examples | `experiments/worked_examples.py` | `results/worked_examples_log.txt` |

## Important limitations
* The benchmark was written by the authors and padded with mechanical rewordings. There is **no independent
  human annotation** and no agreement figure. The annotator columns in `gold_standard_250.csv` are not human
  annotations and are not used by any script.
* On the grouped split our score is below tuned baselines (F1 0.59 against 0.79 to 0.85); over ten other grouped
  splits it is level with them (about 0.85). Type accuracy is below the majority baseline.
* Results depend on the split; see `docs/benchmark_notes.md`.
