# Inter- and Intra-Domain Polysemy in Software Requirements: Sentence-Level Prototype

Code, benchmark and results for the REFSQ 2027 Research Preview
*Distinguishing Inter- and Intra-Domain Polysemy in Software Requirements:
Early Lessons from a Sentence-Level Prototype*.


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


