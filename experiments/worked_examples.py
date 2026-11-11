"""Prints the worked examples of the paper (grouped split, seed 42)."""
import sys, os, io, contextlib, warnings
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src')); os.chdir(os.path.join(ROOT, 'data'))
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from ambiguity_framework import AmbiguityPipeline
from evaluate import extract_features_and_score, tune_dev_split

df = pd.read_csv('gold_standard_250.csv', keep_default_na=False)
df['group'] = df.target_word.str.lower()
idx = list(StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42).split(df, df['consensus_label'], df['group']))
dev = df.iloc[np.concatenate([idx[i][1] for i in range(3)])]
test = df.iloc[np.concatenate([idx[i][1] for i in range(3, 10)])]
with contextlib.redirect_stdout(io.StringIO()):
    pipe = AmbiguityPipeline(ambiguity_threshold=0.35)
    pipe.fit_vocabulary(dev['requirement_sentence'].tolist())
    dfe = extract_features_and_score(pipe, dev); tfe = extract_features_and_score(pipe, test)
    scorer, tau = tune_dev_split(dfe)
tfe['S'] = (scorer.w_spread*tfe['spread'] + scorer.w_polysemy*tfe['baseline_polysemy'] + scorer.w_diversity*tfe['domain_diversity']).values
print('tau* = %.2f' % tau)
# the four items quoted in the paper (found by item id, or by sentence for 'pipeline')
picks = [('review', tfe[tfe.item_id == 'GS-161']), ('pipeline', tfe[tfe.sentence.str.contains('trans-continental')]),
         ('charge', tfe[tfe.item_id == 'GS-113']), ('firewall', tfe[tfe.item_id == 'GS-129'])]
for word, sel in picks:
    r = sel.iloc[0]
    print('\n%s | %s | gold %s | S=%.3f spread=%.3f poly=%.2f div=%.1f' % (r['item_id'], word, r['gold_label'], r['S'], r['spread'], r['baseline_polysemy'], r['domain_diversity']))
    print('  ', r['sentence'])
    for s in r['ranked_senses'][:3]:
        print('   %-18s %.3f  %s' % (s.domain, s.context_similarity, s.gloss[:70]))
