#!/usr/bin/env bash
# Reproduces the tables and numbers of the paper (grouped split, tuned baselines,
# domain-mapping check). Run from the repository root. Takes several minutes.
set -euo pipefail
cd "$(dirname "$0")"
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
python - <<'PY'
import nltk
for p in ("wordnet", "omw-1.4"):
    nltk.download(p, quiet=True)
PY
python experiments/grouped_split.py            | tee results/grouped_split_log.txt            # Table 1 and ten-split column
python experiments/tune_baselines.py           | tee results/tune_baselines_log.txt           # random-split baselines (earlier version)
python experiments/grouped_split_allvocab.py   | tee results/grouped_split_allvocab_log.txt   # vocabulary sensitivity
python experiments/recall_analysis.py          | tee results/recall_analysis_log.txt          # why recall is low
python experiments/evaluate_domain_mapping.py  | tee results/domain_mapping_log.txt           # mapping check
python experiments/worked_examples.py          | tee results/worked_examples_log.txt          # examples of the paper
