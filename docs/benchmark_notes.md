# Benchmark notes
* 250 items, 120 target words, six domains. Labels: Inter-Domain 110, Intra-Domain 64, None 76.
* Items were written by the authors; part of the set consists of mechanically reworded copies of base items.
* The columns `annotator_1_req_engineer`, `annotator_2_architect`, `annotator_3_domain_expert` and
  `is_ambiguous` are simulated and are **not** human annotations. Only `consensus_label` is used, and it is the
  authors' label.
* Grouped split: `StratifiedGroupKFold(10, shuffle=True, random_state=42)` on `target_word`; 3 folds are
  development (77 items) and 7 are test (173 items). Seeds 1 to 10 give the repeated splits.
* The TF-IDF vocabulary is fitted on development sentences only; test words absent from them get zero similarity.
