import sys, os, warnings, io, contextlib
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src')); os.chdir(os.path.join(ROOT, 'data'))
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from ambiguity_framework import AmbiguityPipeline, classify_ambiguity
from evaluate import extract_features_and_score, tune_dev_split
df = pd.read_csv('gold_standard_250.csv', keep_default_na=False); df['group']=df.target_word.str.lower()
sgkf=StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
idx=list(sgkf.split(df, df['consensus_label'], df['group']))
dev=df.iloc[np.concatenate([idx[i][1] for i in range(3)])]; test=df.iloc[np.concatenate([idx[i][1] for i in range(3,10)])]
with contextlib.redirect_stdout(io.StringIO()):
    pipe=AmbiguityPipeline(ambiguity_threshold=0.35); pipe.fit_vocabulary(dev['requirement_sentence'].tolist())
    dfe=extract_features_and_score(pipe,dev); tfe=extract_features_and_score(pipe,test); sc,tau=tune_dev_split(dfe)
tfe['S']=(sc.w_spread*tfe['spread']+sc.w_polysemy*tfe['baseline_polysemy']+sc.w_diversity*tfe['domain_diversity']).values
g=df.set_index('item_id')
def show(r):
    print(r['item_id'],r['word'],'gold',r['gold_label'],'S %.3f sp %.3f po %.2f dv %.1f'%(r['S'],r['spread'],r['baseline_polysemy'],r['domain_diversity']),'|',r['sentence'])
    print('   rationale:',g.loc[r['item_id'],'annotation_rationale'])
    for s in r['ranked_senses'][:3]: print('     ',s.domain,round(s.context_similarity,3),s.gloss[:85])

import numpy as np
amb=tfe[tfe.gold_label!='None']; miss=amb[amb.S<tau]
def cat(r):
    rs=r['ranked_senses']
    if len(rs)==0: return 'not candidate'
    if max(x.context_similarity for x in rs)<0.05: return 'all sim ~0 (OOV)'
    return 'has context'
c=miss.apply(cat,axis=1); print(len(miss), c.value_counts().to_dict())
hit=amb[amb.S>=tau].apply(cat,axis=1); print('hits',len(hit),hit.value_counts().to_dict())
print('amb by cat',amb.apply(cat,axis=1).value_counts().to_dict())
print('tau',tau)
for k,g in miss.groupby(c): print(k, 'S mean',g.S.mean().round(3), 'poly mean',g.baseline_polysemy.mean().round(2))
