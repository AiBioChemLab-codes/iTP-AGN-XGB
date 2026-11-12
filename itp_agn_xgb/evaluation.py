"""Evaluation metrics, summaries and figures generated at runtime."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                            f1_score, matthews_corrcoef, precision_score, roc_auc_score)

METRICS = ['ACC', 'MCC', 'SEN', 'SPE', 'PRE', 'F1', 'ROC_AUC', 'AP']


def evaluate_metrics(y, probability, threshold=.5):
    prediction = np.asarray(probability) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return dict(zip(METRICS, [accuracy_score(y, prediction), matthews_corrcoef(y, prediction),
        tp / (tp + fn), tn / (tn + fp), precision_score(y, prediction, zero_division=0),
        f1_score(y, prediction, zero_division=0), roc_auc_score(y, probability),
        average_precision_score(y, probability)]))

def make_report(run,config):
    single = pd.read_csv(run/'test_metrics.csv')
    voting = pd.read_csv(run/'voting_metrics.csv')
    single.groupby(['model','mode'],as_index=False)[METRICS].mean().to_csv(run/'classifier_equal_task_means.csv',index=False)
    contrasts=[]
    for kind in config['models']:
        s=single[single.model==kind]
        initial=s[s['mode']=='INIT'].set_index('task')[METRICS]
        augmented=s[s['mode']=='AGN'].set_index('task')[METRICS]
        contrasts.append((augmented-initial).reset_index().assign(model=kind,contrast='AGN-INIT'))
    if len(voting):
        voting.groupby(['mode','n_models'],as_index=False)[METRICS].mean().to_csv(run/'voting_equal_task_means.csv',index=False)
        for mode in ['INIT','AGN']:
            reference=single[(single.model=='XGBoost')&(single['mode']==mode)].set_index('task')[METRICS]
            for size in config['voting_sizes']:
                s=voting[(voting['mode']==mode)&(voting.n_models==size)].set_index('task')[METRICS]
                contrasts.append((s-reference).reset_index().assign(model='SoftVoting',mode=mode,n_models=size,contrast='Voting-Single'))
    pd.concat(contrasts,ignore_index=True).to_csv(run/'paired_differences.csv',index=False)
    cv=pd.read_csv(run/'cv_metrics.csv')
    gap=cv.copy()
    for name in METRICS:
        gap[name+'_train_minus_validation']=gap['train_'+name]-gap[name]
    gap.to_csv(run/'train_validation_gaps.csv',index=False)
    figures=run/'figures'
    figures.mkdir(exist_ok=True)
    for metric in ['MCC','SEN','F1','ROC_AUC']:
        fig,axes=plt.subplots(1,len(config['models']),figsize=(5*len(config['models']),4),sharey=True,squeeze=False)
        for ax,kind in zip(axes[0],config['models']):
            values=single[single.model==kind].pivot(index='task',columns='mode',values=metric).reindex(config['tasks'])
            for i,task in enumerate(config['tasks']):
                ax.plot([values.loc[task,'INIT'],values.loc[task,'AGN']],[i,i],color='gray')
            ax.scatter(values.INIT,range(len(values)),label='INIT')
            ax.scatter(values.AGN,range(len(values)),label='AGN')
            ax.set_yticks(range(len(values)),values.index)
            ax.set_title(kind)
            ax.set_xlabel(metric)
        axes[0,0].legend()
        fig.tight_layout()
        fig.savefig(figures/f'classifier_{metric}.png',dpi=180)
        plt.close(fig)
    means=single.groupby(['model','mode'])[METRICS].mean()
    lines=['# iTP-AGN-XGB Run Report','',f'Protocol: `{config["protocol"]}`.',
        f'Tasks: {", ".join(config["tasks"])}; folds: {config["folds"]}; threshold: {config["threshold"]}.','',
        '## Equal-task Mean Test Metrics','', '```text', means.to_string(), '```','',
        '## Evaluation Details','',
        '- The primary model is AGN-enhanced XGBoost; soft voting is a comparator.',
        '- Features and evaluation outputs are computed from the supplied FASTA files.',
        '- Cross-validation selects parameters using training data only.',
        '- Test data are evaluated after parameters and ensemble settings are fixed.',
        '- Training/validation metric differences are saved in train_validation_gaps.csv.','',
        '## Figures','']
    for p in sorted(figures.glob('*.png')):
        lines.extend([f'![{p.stem}](figures/{p.name})',''])
    (run/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
