"""Training entry point, run directories and workflow stages."""
import argparse
from pathlib import Path
from datetime import datetime, timezone
import importlib.metadata
import json
import platform
import sys
import pandas as pd
from . import data, training
from .utils import TASKS, STAGES, digest, write_json
from .evaluation import evaluate_metrics


class Workflow:
    def __init__(self, root, config_path, run_dir):
        self.root = Path(root).resolve()
        self.run = Path(run_dir).resolve()
        self.config = json.loads(Path(config_path).read_text())
        c = self.config
        if (not c['tasks'] or len(set(c['tasks'])) != len(c['tasks'])
                or not set(c['tasks']) <= set(TASKS) or c['folds'] < 2
                or c['agn']['copies'] < 1 or c['agn']['sigma'] < 0
                or not 0 < c['threshold'] < 1
                or not c['models'] or 'XGBoost' not in c['models']
                or not set(c['models']) <= {'XGBoost', 'LR', 'KNN'}
                or len(set(c['models'])) != len(c['models'])
                or any(size < 1 for size in c['voting_sizes'])):
            raise ValueError('Invalid experiment configuration')
        self.run.mkdir(parents=True, exist_ok=True)
        inputs = [{'path': f'data/fasta/{task}/{split}.fasta',
                   'sha256': digest(self.root/'data/fasta'/task/f'{split}.fasta')}
                  for task in c['tasks'] for split in ['Train', 'Test']]
        source = [{'path': p.relative_to(self.root).as_posix(), 'sha256': digest(p)}
                  for p in sorted((self.root/'itp_agn_xgb').rglob('*.py'))]
        versions = {p: importlib.metadata.version(p) for p in
                    ['numpy','pandas','scipy','scikit-learn','xgboost','matplotlib','joblib']}
        identity = dict(config=c, inputs=inputs, source=source, versions=versions,
                        python=platform.python_version())
        meta = self.run/'run_metadata.json'
        if meta.exists():
            old = json.loads(meta.read_text())
            if old['identity'] != identity:
                raise ValueError('Config/input/code/environment changed; use a NEW --run-dir')
        else:
            write_json(meta, dict(identity=identity, created_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv))
            write_json(self.run/'config.json', c)
        self.tasks = c['tasks']

    def csv(self, data, filename):
        frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        frame.to_csv(self.run/filename, index=False)
        return frame

    def run_stage(self, stage):
        idx = STAGES.index(stage)
        for previous in STAGES[:idx]:
            if not (self.run/(previous+'.done.json')).exists():
                raise RuntimeError(f'Run prerequisite stage first: {previous}')
        marker = self.run/(stage+'.done.json')
        if marker.exists():
            recorded = json.loads(marker.read_text())
            for row in recorded['artifacts']:
                p = self.run/row['path']
                if not p.is_file() or digest(p) != row['sha256']:
                    raise RuntimeError(f'Checkpoint artifact changed/missing: {p}')
            print(f'[cached] {stage}', flush=True)
            return
        # Verify completed upstream artifacts before executing later stages.
        for previous in STAGES[:idx]:
            recorded = json.loads((self.run/(previous+'.done.json')).read_text())
            for row in recorded['artifacts']:
                p = self.run/row['path']
                if not p.is_file() or digest(p) != row['sha256']:
                    raise RuntimeError(f'Upstream artifact changed/missing: {p}')
        print(f'[start] {stage}', flush=True)
        getattr(self, stage)()
        artifacts = [{'path': p.relative_to(self.run).as_posix(), 'sha256': digest(p)}
                     for p in sorted(self.run.rglob('*')) if p.is_file()
                     and not p.name.endswith('.done.json')]
        write_json(marker, dict(completed_utc=datetime.now(timezone.utc).isoformat(), artifacts=artifacts))
        print(f'[done] {stage}', flush=True)

    def store_prediction(self, task, kind, mode, f, probability, filename, **extra):
        threshold = self.config['threshold']
        folder = self.run/'predictions'
        folder.mkdir(exist_ok=True)
        pd.DataFrame(dict(PID=f.PID,Target=f.Target,prob=probability,pred=(probability>=threshold).astype(int))).to_csv(folder/filename,index=False)
        return dict(task=task,model=kind,mode=mode,**extra,**evaluate_metrics(f.Target.to_numpy(),probability,threshold))

    def audit(self):
        return data.audit(self)

    def extract(self):
        return data.extract(self)

    def frame(self, task, split):
        return data.frame(self, task, split)

    def split(self):
        return data.split(self)

    def folds(self, task):
        return data.folds(self, task)

    def select(self):
        return training.select(self)

    def evaluate(self):
        return training.evaluate(self)

    def compare(self):
        training.run_voting(self)

    def report(self):
        from .evaluation import make_report
        make_report(self.run,self.config)

def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description='iTP-AGN-XGB: FASTA to results')
    parser.add_argument('stage',nargs='?',default='run',choices=['run',*STAGES])
    parser.add_argument('--config',type=Path,default=root/'configs/default.json')
    parser.add_argument('--run-dir',type=Path,default=root/'runs/full',
                        help='Output directory (default: runs/full under the project root); resume matching runs')
    args=parser.parse_args()
    workflow=Workflow(root,args.config,args.run_dir)
    for stage in STAGES if args.stage=='run' else [args.stage]:
        workflow.run_stage(stage)
    print('Results:',workflow.run)
