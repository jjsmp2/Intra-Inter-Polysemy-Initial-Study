import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src')); os.chdir(os.path.join(ROOT, 'data'))
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import wordnet as wn
from ambiguity_framework import AmbiguityPipeline, classify_ambiguity
from evaluate import prf1, extract_features_and_score, tune_dev_split, bootstrap_ci

df = pd.read_csv('gold_standard_250.csv', keep_default_na=False)
dev, test = train_test_split(df, test_size=0.70, random_state=42, stratify=df['consensus_label'])
print(len(dev), len(test))

def best_threshold(scores, y, grid):
    best, bf = None, -1
    for t in grid:
        f = prf1(y, scores >= t)['f1']
        if f > bf: bf, best = f, t
    return best, bf

def type_acc(gold, pred):
    amb = [(g,p) for g,p in zip(gold,pred) if g!='None']
    return sum(g==p for g,p in amb)/len(amb)

def report(name, y, pred, gold, ptype, extra=''):
    m = prf1(y, pred)
    ta = type_acc(gold, ptype)
    print(f"{name:28s} acc {m['accuracy']:.3f} P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f} type {ta:.3f} {extra}")
    return dict(method=name, acc=m['accuracy'], prec=m['precision'], rec=m['recall'], f1=m['f1'], type=ta, extra=extra)

rows = []
ydev = dev['consensus_label'].ne('None').values
ytest = test['consensus_label'].ne('None').values
gold_test = test['consensus_label'].values

# ---- full pipeline (reproduce) ----
pipe = AmbiguityPipeline(ambiguity_threshold=0.35)
pipe.fit_vocabulary(dev['requirement_sentence'].tolist())
dfe = extract_features_and_score(pipe, dev)
tfe = extract_features_and_score(pipe, test)
scorer, tau = tune_dev_split(dfe)
def score_of(f): return (scorer.w_spread*f['spread']+scorer.w_polysemy*f['baseline_polysemy']+scorer.w_diversity*f['domain_diversity']).values
def types_from(f, pred):
    out=[]
    for i,(p,(_,r)) in enumerate(zip(pred, f.iterrows())):
        if not p: out.append('None')
        else:
            s=r.get('sigma_star',[])
            out.append(classify_ambiguity(s) if s else 'Intra-Domain')
    return out
pred = score_of(tfe) >= tau
rows.append(report('Proposed (reproduced)', ytest, pred, gold_test, types_from(tfe,pred), f'tau={tau:.2f}'))

# ---- Polysemy count: tuned k on dev ----
cnt = lambda d: np.array([len(wn.synsets(w.lower())) for w in d['target_word']])
cd, ct = cnt(dev), cnt(test)
k, kf = best_threshold(cd.astype(float), ydev, range(1, 30))
pred = ct >= k
rows.append(report('Polysemy count (tuned)', ytest, pred, gold_test, ['Intra-Domain' if p else 'None' for p in pred], f'k={k} devF1={kf:.3f}'))
pred4 = ct >= 4
rows.append(report('Polysemy count (k=4 fixed)', ytest, pred4, gold_test, ['Intra-Domain' if p else 'None' for p in pred4], 'k=4'))

# ---- Spread only: tuned tau on dev ----
grid = np.arange(0.15, 0.85, 0.02)
ts, tsf = best_threshold(dfe['spread'].values, ydev, grid)
pred = tfe['spread'].values >= ts
rows.append(report('Spread only (tuned)', ytest, pred, gold_test, types_from(tfe,pred), f'tau={ts:.2f} devF1={tsf:.3f}'))
pred = tfe['spread'].values >= 0.55
rows.append(report('Spread only (0.55 fixed)', ytest, pred, gold_test, types_from(tfe,pred), 'tau=0.55'))

# ---- Lesk: tune (gap, min overlap) on dev ----
def lesk_feats(d):
    out=[]
    for _,r in d.iterrows():
        sw=set(r['requirement_sentence'].lower().split())
        senses=wn.synsets(r['target_word'].lower())
        if len(senses)<2: out.append((0,99)); continue
        ov=sorted([len(sw & set((s.definition()+' '+' '.join(s.examples())).lower().split())) for s in senses], reverse=True)
        out.append((ov[0], ov[0]-ov[1]))
    return np.array(out)
ld, lt = lesk_feats(dev), lesk_feats(test)
best=(None,-1)
for g in range(0,8):
    for m in range(1,6):
        f=prf1(ydev,(ld[:,0]>=m)&(ld[:,1]<=g))['f1']
        if f>best[1]: best=((g,m),f)
(g,m),lf=best
pred=(lt[:,0]>=m)&(lt[:,1]<=g)
rows.append(report('Classic Lesk (tuned)', ytest, pred, gold_test, ['Intra-Domain' if p else 'None' for p in pred], f'gap<={g},ov>={m} devF1={lf:.3f}'))
pred=(lt[:,0]>=1)&(lt[:,1]<=1)
rows.append(report('Classic Lesk (gap<=1 fixed)', ytest, pred, gold_test, ['Intra-Domain' if p else 'None' for p in pred], 'gap<=1'))

# ---- no glossary (retuned) ----
p2 = AmbiguityPipeline(use_glossary=False); p2.fit_vocabulary(dev['requirement_sentence'].tolist())
d2 = extract_features_and_score(p2, dev); t2 = extract_features_and_score(p2, test)
sc2, tau2 = tune_dev_split(d2)
s2 = (sc2.w_spread*t2['spread']+sc2.w_polysemy*t2['baseline_polysemy']+sc2.w_diversity*t2['domain_diversity']).values
pred = s2>=tau2
rows.append(report('Score w/o glossary (tuned)', ytest, pred, gold_test, types_from(t2,pred), f'tau={tau2:.2f}'))

# ---- trivial ----
pred = np.ones(len(test), bool)
rows.append(report('All ambiguous', ytest, pred, gold_test, ['Inter-Domain']*len(test)))
print('dev all-ambiguous F1', prf1(ydev, np.ones(len(dev),bool))['f1'])
# dev F1 of each tuned method
print('dev F1 proposed', prf1(ydev, score_of(dfe)>=tau)['f1'])
pd.DataFrame(rows).to_csv(os.path.join(ROOT,'results','tuned_baselines.csv'), index=False)
