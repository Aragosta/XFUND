# FEATURE IMPORTANCE

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import KFOLD
import MP
from sklearn.metrics import f1_score

def featImpMDA(clf, X, y, cv, sample_weight, t1, pctEmbargo, scoring='neg_log_loss'):
    # feat importance based on OOS score reduction
    if scoring not in ['neg_log_loss', 'accuracy']:
        raise Exception('wrong scoring method.')

    from sklearn.metrics import log_loss, accuracy_score
    cvGen = KFOLD.PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)  # purged cv
    scr0, scr1 = pd.Series(), pd.DataFrame(columns=X.columns)

    for i, (train, test) in enumerate(cvGen.split(X=X)):
        X0, y0, w0 = X.iloc[train, :], y.iloc[train], sample_weight.iloc[train]
        X1, y1, w1 = X.iloc[test, :], y.iloc[test], sample_weight.iloc[test]
        fit = clf.fit(X=X0, y=y0, sample_weight=w0.values)

        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X1)
            scr0.loc[i] = -log_loss(y1, prob, sample_weight=w1.values, labels=clf.classes_)
        else:
            pred = fit.predict(X1)
            scr0.loc[i] = accuracy_score(y1, pred, sample_weight=w1.values)

        for j in X.columns:
            X1_ = X1.copy(deep=True)
            np.random.shuffle(X1_[j].values)  # permutation of a single column

            if scoring == 'neg_log_loss':
                prob = fit.predict_proba(X1_)
                scr1.loc[i, j] = -log_loss(y1, prob, sample_weight=w1.values, labels=clf.classes_)
            else:
                pred = fit.predict(X1_)
                scr1.loc[i, j] = accuracy_score(y1, pred, sample_weight=w1.values)

    imp = (-scr1).add(scr0, axis=0)

    if scoring == 'neg_log_loss':
        imp = imp / -scr1
    else:
        imp = imp / (1. - scr1)

    imp = pd.concat({'mean': imp.mean(), 'std': imp.std() * imp.shape[0]**-.5}, axis=1)
    return imp, scr0.mean()


def auxFeatImpSFI(featNames, clf, trnsX, cont, scoring, cvGen):
    imp = pd.DataFrame(columns=['mean', 'std'])
    for featName in featNames:
        df0 = KFOLD.cvScore(clf, X=trnsX[[featName]], y=cont['bin'], sample_weight=cont['w'], scoring=scoring, cvGen=cvGen)
        imp.loc[featName, 'mean'] = -df0.mean()
        imp.loc[featName, 'std'] = df0.std() * df0.shape[0]**-.5
    return imp



def featImportance(trnsX,cont,n_estimators=1000,cv=10,max_samples=1.,numThreads=24, pctEmbargo=0,scoring='accuracy',method='SFI',minWLeaf=0.,**kargs):
    # feature importance from a random forest
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import BaggingClassifier
    
    n_jobs=(-1 if numThreads>1 else 1) 
    # run 1 thread with ht_helper in dirac1 #1) prepare classifier,cv. max_features=1, to prevent masking
    clf=DecisionTreeClassifier(criterion='entropy',max_features=1,
    class_weight='balanced',min_weight_fraction_leaf=minWLeaf) 
    clf=BaggingClassifier(base_estimator=clf,n_estimators=n_estimators,
    max_features=1.,max_samples=max_samples,oob_score=True,n_jobs=n_jobs)
    fit=clf.fit(X=trnsX,y=cont['bin'],sample_weight=cont['w'].values) 
    oob=fit.oob_score_
    if method=='MDA':
        imp,oos=featImpMDA(clf,X=trnsX,y=cont['bin'],cv=cv,sample_weight=cont['w'], t1=cont['t1'],pctEmbargo=pctEmbargo,scoring=scoring)
        cvGen=KFOLD.PurgedKFold(n_splits=cv,t1=cont['t1'],pctEmbargo=pctEmbargo) 
    elif method=='SFI': 
        cvGen=KFOLD.PurgedKFold(n_splits=cv,t1=cont['t1'],pctEmbargo=pctEmbargo) 
        oos=KFOLD.cvScore(clf,X=trnsX,y=cont['bin'],sample_weight=cont['w'],scoring=scoring,
    cvGen=cvGen).mean()
    clf.n_jobs=1 # paralellize auxFeatImpSFI rather than clf 
    imp=MP.MultiProcessingFunctions.mp_pandas_obj(auxFeatImpSFI,('featNames',trnsX.columns),numThreads,
    clf=clf,trnsX=trnsX,cont=cont,scoring=scoring,cvGen=cvGen) 
    return imp,oob,oos


def plotFeatImportance(imp, oob, oos, method, tag=0, simNum=0, **kargs):
    # plot mean imp bars with std
    plt.figure(figsize=(10, imp.shape[0] / 5.))
    imp = imp.sort_values('mean', ascending=True)
    ax = imp['mean'].plot(kind='barh', color='b', alpha=.25, xerr=imp['std'], error_kw={'ecolor': 'r'})

    if method == 'MDI':
        plt.xlim([0, imp.sum(axis=1).max()])
        plt.axvline(1. / imp.shape[0], linewidth=1, color='r', linestyle='dotted')
    else:
        plt.xlim([imp['mean'].min() - imp['std'].max(), imp['mean'].max() + imp['std'].max()])

    ax.get_yaxis().set_visible(False)
    for i, j in zip(ax.patches, imp.index):
        ax.text(i.get_width() / 2, i.get_y() + i.get_height() / 2, j, ha='center', va='center', color='black')

    plt.title(f'tag={tag} | simNum={simNum} | oob={round(oob, 4)} | oos={round(oos, 4)}')
    plt.show()
    plt.clf()
    return


## PCA

def get_eVec(dot,varThres):
    # Compute eVec from dot prod matrix, reduce dimension
    eVal,eVec=np.linalg.eigh(dot)
    idx=eVal.argsort()[::-1] # arguments for sorting eVal desc
    eVal,eVec=eVal[idx],eVec[:,idx]
    #2) only positive eVals
    eVal=pd.Series(eVal,index=['PC_'+str(i+1) for i in range(eVal.shape[0])])
    eVec=pd.DataFrame(eVec,index=dot.index,columns=eVal.index)
    eVec=eVec.loc[:,eVal.index]
    #3) reduce dimension, form PCs
    cumVar=eVal.cumsum()/eVal.sum()
    dim=cumVar.values.searchsorted(varThres)
    eVal,eVec=eVal.iloc[:dim+1],eVec.iloc[:,:dim+1]
    return eVal,eVec

def orthoFeats(dfX,varThres=.95):
    # Given a DataFrame dfX of features, compute orthofeatures dfP
    dfZ=dfX.sub(dfX.mean(),axis=1).div(dfX.std(),axis=1) # standardize
    dot=pd.DataFrame(np.dot(dfZ.T,dfZ)/dfZ.shape[0],index=dfX.columns,columns=dfX.columns)
    eVal,eVec=get_eVec(dot,varThres)
    dfP=np.dot(dfZ,eVec)
    return dfP