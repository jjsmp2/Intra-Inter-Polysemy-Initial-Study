import sys, os, json, warnings
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src')); os.chdir(os.path.join(ROOT, 'data'))
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score
from nltk.corpus import wordnet as wn
from ambiguity_framework import AmbiguityPipeline, classify_ambiguity
from evaluate import prf1, extract_features_and_score, tune_dev_split, bootstrap_ci
import io, contextlib

df = pd.read_csv('gold_standard_250.csv', keep_default_na=False)
df['group'] = df['target_word'].str.lower()
print('groups', df.group.nunique(), 'items', len(df))

def best_threshold(scores, y, grid):
    best, bf = None, -1
    for t in grid:
        f = prf1(y, scores >= t)['f1']
        if f > bf: bf, best = f, t
    return best, bf
def type_acc(gold, pred):
    amb = [(g,p) for g,p in zip(gold,pred) if g!='None']
    return sum(g==p for g,p in amb)/len(amb)
def types_from(f, pred):
    out=[]
    for p,(_,r) in zip(pred, f.iterrows()):
        if not p: out.append('None')
        else:
            s=r.get('sigma_star',[])
            out.append(classify_ambiguity(s) if s else 'Intra-Domain')
    return out
def lesk_feats(d):
    out=[]
    for _,r in d.iterrows():
        sw=set(r['requirement_sentence'].lower().split())
        senses=wn.synsets(r['target_word'].lower())
        if len(senses)<2: out.append((0,99)); continue
        ov=sorted([len(sw & set((s.definition()+' '+' '.join(s.examples())).lower().split())) for s in senses], reverse=True)
        out.append((ov[0], ov[0]-ov[1]))
    return np.array(out)

def run(dev, test, detail=False):
    res={}
    ydev=dev['consensus_label'].ne('None').values; ytest=test['consensus_label'].ne('None').values
    gold=test['consensus_label'].values
    with contextlib.redirect_stdout(io.StringIO()):
        pipe=AmbiguityPipeline(ambiguity_threshold=0.35); pipe.fit_vocabulary(df["requirement_sentence"].tolist())
        dfe=extract_features_and_score(pipe,dev); tfe=extract_features_and_score(pipe,test)
        scorer,tau=tune_dev_split(dfe)
    S=lambda f:(scorer.w_spread*f['spread']+scorer.w_polysemy*f['baseline_polysemy']+scorer.w_diversity*f['domain_diversity']).values
    pred=S(tfe)>=tau
    m=prf1(ytest,pred); res['Proposed']=dict(dev_f1=prf1(ydev,S(dfe)>=tau)['f1'],**m,type=type_acc(gold,types_from(tfe,pred)),param=f'tau={tau:.2f}; w=({scorer.w_spread:.3f},{scorer.w_polysemy:.3f},{scorer.w_diversity:.3f})')
    if detail:
        res['_proposed_detail']=dict(scores=S(tfe), ytest=ytest, pred=pred, types=types_from(tfe,pred), gold=gold, pr_auc=average_precision_score(ytest,S(tfe)))
    cnt=lambda d: np.array([len(wn.synsets(w.lower())) for w in d['target_word']]).astype(float)
    cd,ct=cnt(dev),cnt(test); k,kf=best_threshold(cd,ydev,range(1,30)); pr=ct>=k
    res['Polysemy count']=dict(dev_f1=kf,**prf1(ytest,pr),type=type_acc(gold,['Intra-Domain' if x else 'None' for x in pr]),param=f'k={k}')
    grid=np.arange(0.15,0.85,0.02); ts,tsf=best_threshold(dfe['spread'].values,ydev,grid); ps=tfe['spread'].values>=ts
    res['Spread only']=dict(dev_f1=tsf,**prf1(ytest,ps),type=type_acc(gold,types_from(tfe,ps)),param=f'tau={ts:.2f}')
    ld,lt=lesk_feats(dev),lesk_feats(test); best=(None,-1)
    for g in range(0,8):
        for mm in range(1,6):
            f=prf1(ydev,(ld[:,0]>=mm)&(ld[:,1]<=g))['f1']
            if f>best[1]: best=((g,mm),f)
    (g,mm),lf=best; pl=(lt[:,0]>=mm)&(lt[:,1]<=g)
    res['Classic Lesk']=dict(dev_f1=lf,**prf1(ytest,pl),type=type_acc(gold,['Intra-Domain' if x else 'None' for x in pl]),param=f'g<={g},m>={mm}')
    with contextlib.redirect_stdout(io.StringIO()):
        p2=AmbiguityPipeline(use_glossary=False); p2.fit_vocabulary(df["requirement_sentence"].tolist())
        d2=extract_features_and_score(p2,dev); t2=extract_features_and_score(p2,test); sc2,tau2=tune_dev_split(d2)
    s2=(sc2.w_spread*t2['spread']+sc2.w_polysemy*t2['baseline_polysemy']+sc2.w_diversity*t2['domain_diversity']).values
    p2p=s2>=tau2
    res['No glossary']=dict(dev_f1=prf1(ydev,(sc2.w_spread*d2['spread']+sc2.w_polysemy*d2['baseline_polysemy']+sc2.w_diversity*d2['domain_diversity']).values>=tau2)['f1'],**prf1(ytest,p2p),type=type_acc(gold,types_from(t2,p2p)),param=f'tau={tau2:.2f}')
    allp=np.ones(len(test),bool)
    res['All ambiguous']=dict(dev_f1=prf1(ydev,np.ones(len(dev),bool))['f1'],**prf1(ytest,allp),type=type_acc(gold,['Inter-Domain']*len(test)),param='')
    maj=pd.Series([g for g in gold if g!='None']).value_counts().iloc[0]/sum(g!='None' for g in gold)
    res['_majority']=maj
    return res

def split(seed, folds=10, dev_folds=3):
    sgkf=StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    idx=list(sgkf.split(df, df['consensus_label'], df['group']))
    dev_idx=np.concatenate([idx[i][1] for i in range(dev_folds)]); test_idx=np.concatenate([idx[i][1] for i in range(dev_folds,folds)])
    return df.iloc[dev_idx], df.iloc[test_idx]

# primary split, seed 42
dev,test=split(42)
print('PRIMARY: dev',len(dev),'test',len(test),'overlap words',len(set(dev.group)&set(test.group)))
print(dev.consensus_label.value_counts().to_dict(), test.consensus_label.value_counts().to_dict())
print('dev groups',dev.group.nunique(),'test groups',test.group.nunique())
res=run(dev,test,detail=True)
d=res.pop('_proposed_detail'); maj=res.pop('_majority')
rows=[]
for k,v in res.items():
    print(f"{k:16s} devF1 {v['dev_f1']:.3f} acc {v['accuracy']:.3f} P {v['precision']:.3f} R {v['recall']:.3f} F1 {v['f1']:.3f} type {v['type']:.3f} {v['param']}")
    rows.append(dict(method=k,**v))
print('majority type acc',maj,'PR-AUC',d['pr_auc'])
ci=bootstrap_ci(list(d['ytest']),list(d['pred']),list(d['gold']),list(d['types']),1000,seed=42)
print({k:(round(a*100,1),round(b*100,1)) for k,(a,b) in ci.items()})
nt=sum(g!='None' for g in d['gold']); nc=sum(1 for g,p in zip(d['gold'],d['types']) if g!='None' and g==p)
mm=prf1(d['ytest'],d['pred']); print('proposed tp fp fn tn',mm['tp'],mm['fp'],mm['fn'],mm['tn'],'type',nc,nt, 'detected',mm['tp'])
pd.DataFrame(rows).to_csv(os.path.join(ROOT,'results','grouped_split_results_allvocab.csv'),index=False)

# robustness: 10 other grouped splits
allr=[]
for seed in range(1,11):
    dv,ts_=split(seed); r=run(dv,ts_); r.pop('_majority')
    for k,v in r.items(): allr.append(dict(seed=seed,method=k,f1=v['f1'],type=v['type'],dev_f1=v['dev_f1'],prec=v['precision'],rec=v['recall']))
    print('seed',seed,'done',{k:round(v['f1'],3) for k,v in r.items()})
a=pd.DataFrame(allr); a.to_csv(os.path.join(ROOT,'results','grouped_split_repeats_allvocab.csv'),index=False)
print(a.groupby('method')[['f1','type']].agg(['mean','std']).round(3))
