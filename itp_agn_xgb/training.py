"""Cross-validation selection, final fitting and soft voting."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from .models import fit_model, predict, perturbed_params
from .evaluation import METRICS, evaluate_metrics
from .utils import write_json


def select(workflow):
    """Select parameters using training folds only."""
    selected, all_cv, scores, oof = [], [], [], []
    for task in workflow.tasks:
        f = workflow.frame(task,'Train')
        X,y = f.iloc[:,2:].to_numpy(float), f.Target.to_numpy(int)
        folds = workflow.folds(task)
        for kind in workflow.config['models']:
            for mode in ['INIT','AGN']:
                print(f'  select {task}/{kind}/{mode}', flush=True)
                best_score, best_params, best_outputs = -np.inf, None, None
                for candidate, params in enumerate(ParameterGrid(workflow.config['search'][kind])):
                    outputs, candidate_scores = [], []
                    for fold,(tr,va) in enumerate(folds,1):
                        scaler, model = fit_model(X[tr],y[tr],kind,params,mode,workflow.config)
                        p = predict(scaler,model,X[va])
                        fit = predict(scaler,model,X[tr])
                        measured = evaluate_metrics(y[va],p,workflow.config['threshold'])
                        candidate_scores.append(measured['ROC_AUC'])
                        scores.append(dict(task=task,model=kind,mode=mode,candidate=candidate,fold=fold,
                            ROC_AUC=measured['ROC_AUC'],params=json.dumps(params,sort_keys=True)))
                        outputs.append((fold,va,p,measured,evaluate_metrics(y[tr],fit,workflow.config['threshold'])))
                    mean_score = float(np.mean(candidate_scores))
                    if mean_score > best_score:
                        best_score,best_params,best_outputs = mean_score,params,outputs
                selected.append(dict(task=task,model=kind,mode=mode,params=best_params,selection_metric='mean_validation_ROC_AUC',cv_score=best_score))
                for fold,va,p,measured,train_metric in best_outputs:
                    all_cv.append(dict(task=task,model=kind,mode=mode,fold=fold,**measured,
                        **{'train_'+k:v for k,v in train_metric.items()}))
                    oof.extend(dict(task=task,model=kind,mode=mode,fold=fold,PID=f.PID.iloc[i],Target=int(y[i]),prob=float(v)) for i,v in zip(va,p))
    write_json(workflow.run/'locked_configurations.json',selected)
    # Lock all soft-voting member configurations before test evaluation.
    comparisons = []
    for row in selected:
        if row['model'] != 'XGBoost':
            continue
        comparisons.extend(voting_configurations(row,workflow.config))
    write_json(workflow.run/'locked_comparisons.json',comparisons)
    workflow.csv(scores,'search_fold_scores.csv')
    workflow.csv(all_cv,'cv_metrics.csv')
    workflow.csv(oof,'oof_predictions.csv')
    cv = pd.DataFrame(all_cv)
    workflow.csv(cv.groupby(['task','model','mode'])[METRICS].agg(['mean','std']).reset_index(), 'cv_summary.csv')

def evaluate(workflow):
    rows, fits = [], []
    folder = workflow.run/'models'
    folder.mkdir(exist_ok=True)
    for row in json.loads((workflow.run/'locked_configurations.json').read_text()):
        task,kind,mode = row['task'],row['model'],row['mode']
        f,t = workflow.frame(task,'Train'),workflow.frame(task,'Test')
        X,y = f.iloc[:,2:].to_numpy(float),f.Target.to_numpy(int)
        scaler,model = fit_model(X,y,kind,row['params'],mode,workflow.config)
        p = predict(scaler,model,t.iloc[:,2:].to_numpy(float))
        rows.append(workflow.store_prediction(task,kind,mode,t,p,f'{task}_{kind}_{mode}.csv'))
        fits.append(dict(task=task,model=kind,mode=mode,**evaluate_metrics(y,predict(scaler,model,X),workflow.config['threshold'])))
        joblib.dump(dict(scaler=scaler,model=model,columns=f.columns[2:].tolist(),params=row['params']), folder/f'{task}_{kind}_{mode}.joblib')
        if kind=='XGBoost':
            model.save_model(str(folder/f'{task}_{kind}_{mode}.json'))
    workflow.csv(rows,'test_metrics.csv')
    workflow.csv(fits,'training_fit_metrics.csv')

def voting_configurations(row, config):
    """Fix member counts, parameters, seeds and equal weights before testing."""
    return [dict(task=row['task'],mode=row['mode'],model='SoftVoting',n_models=size,
        members=[dict(seed=config['seed']+i*100,params=perturbed_params(row['params'],i,config['seed'])) for i in range(size)],
        weights=[1/size]*size) for size in config['voting_sizes']]

def run_voting(workflow):
    """Fit the configured voting members and save models and pooled predictions."""
    voting, cache = [], {}
    for row in json.loads((workflow.run/'locked_comparisons.json').read_text()):
        if row['model'] != 'SoftVoting':
            continue
        task,mode = row['task'],row['mode']
        f,t = workflow.frame(task,'Train'),workflow.frame(task,'Test')
        X,y = f.iloc[:,2:].to_numpy(float),f.Target.to_numpy(int)
        probs = []
        for i, member in enumerate(row['members']):
            key = task,mode,i
            if key not in cache:
                scaler,model = fit_model(X,y,'XGBoost',member['params'],mode,workflow.config,member['seed'])
                cache[key] = predict(scaler,model,t.iloc[:,2:].to_numpy(float))
                joblib.dump(dict(scaler=scaler,model=model,params=member['params'],columns=f.columns[2:].tolist()),workflow.run/'models'/f'{task}_{mode}_voting_member{i}.joblib')
            probs.append(cache[key])
        p = np.average(probs,axis=0,weights=row['weights'])
        voting.append(workflow.store_prediction(task,'SoftVoting',mode,t,p,f'{task}_{mode}_voting{row["n_models"]}.csv',n_models=row['n_models']))
    workflow.csv(pd.DataFrame(voting,columns=['task','model','mode','n_models',*METRICS]),'voting_metrics.csv')
