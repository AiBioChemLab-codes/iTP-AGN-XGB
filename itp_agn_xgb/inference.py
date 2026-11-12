"""Predict from labeled FASTA using pretrained XGBoost models and saved scaling parameters."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .features.feature_extraction import extract_all_features
from .features.feature_merge import merge_feature_tables
from .utils import TASKS
from .evaluation import evaluate_metrics
from .data import read_fasta


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', choices=['paper'], default='paper')
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=TASKS)
    parser.add_argument('--modes', nargs='+', choices=['init','agn'], default=['init','agn'])
    parser.add_argument('--fasta', type=Path, help='Custom labeled FASTA; requires exactly one task')
    parser.add_argument('--output', type=Path, default=root/'runs/prediction',
                        help='New output directory (default: runs/prediction under the project root)')
    args = parser.parse_args()
    if tuple(int(v) for v in xgb.__version__.split('.')[:2]) < (3, 0):
        parser.error('Pretrained models require XGBoost 3.0.5; use Python >=3.10 and install requirements.txt')
    if args.fasta and len(args.tasks) != 1:
        parser.error('--fasta requires exactly one task')
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for task in args.tasks:
        fasta = args.fasta or root/'data/fasta'/task/'Test.fasta'
        seq = read_fasta(fasta)
        frame = merge_feature_tables(extract_all_features(seq.rename(columns={'PID':'id','Target':'label'})))
        for mode in args.modes:
            model_path = root/'pretrained'/task/f'{mode}_model.json'
            scaler_path = root/'pretrained'/task/f'{mode}_scaler.npz'
            with np.load(scaler_path,allow_pickle=False) as scaler:
                if frame.columns[2:].tolist() != scaler['columns'].tolist():
                    raise ValueError('Extracted feature order differs from saved model order')
                X = frame.iloc[:,2:].to_numpy(np.float32)
                X -= scaler['mean']
                X /= scaler['scale']
            model = xgb.Booster()
            model.load_model(model_path)
            model.set_param({'device':'cpu','nthread':4})
            probability = model.predict(xgb.DMatrix(X))
            pred = (probability>=0.5).astype(int)
            pd.DataFrame(dict(PID=seq.PID,Target=seq.Target,prob=probability,pred=pred)).to_csv(
                args.output/f'{task}_{mode}_predictions.csv',index=False)
            measured = dict(version=args.version,task=task,mode=mode,samples=len(seq),
                **evaluate_metrics(seq.Target.to_numpy(),probability,0.5))
            results.append(measured)
            print(task,mode,measured['ROC_AUC'],flush=True)
    pd.DataFrame(results).to_csv(args.output/'metrics.csv',index=False)
    (args.output/'run.json').write_text(json.dumps(dict(version=args.version,xgboost=xgb.__version__,
        tasks=args.tasks,modes=args.modes,input='custom' if args.fasta else 'bundled'),indent=2)+'\n')
    print('Results:',args.output)


if __name__=='__main__':
    main()
