"""FASTA input, data checks, feature tables and parent-sample fold assignment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from .utils import write_json
from .features.feature_extraction import extract_all_features
from .features.feature_merge import merge_feature_tables


def read_fasta(path):
    """Read >PID|0/1 records and retain sequences before and after normalization."""
    records, header, sequence = [], None, []
    for line in Path(path).read_text(encoding='utf-8').splitlines() + ['>__END__']:
        if line.startswith('>'):
            if header is not None:
                fields = header.rsplit('|', 1)
                if len(fields) != 2 or not fields[0] or fields[1] not in ['0', '1']:
                    raise ValueError(f'Invalid FASTA header: {header}')
                raw = ''.join(sequence)
                clean = ''.join(raw.split()).upper()
                if len(clean) <= 2 or not set(clean) <= set('ACDEFGHIKLMNPQRSTVWY'):
                    raise ValueError(f'Invalid/too short sequence: {header}')
                records.append(dict(PID=fields[0], Target=int(fields[1]), sequence=clean, raw_sequence=raw))
            header, sequence = line[1:].strip(), []
        elif line.strip():
            if header is None:
                raise ValueError('FASTA sequence precedes first header')
            sequence.append(line)
    if not records:
        raise ValueError(f'Empty FASTA: {path}')
    return pd.DataFrame(records)

def audit(workflow):
    counts, issues, cross_task = [], [], []
    all_records = []
    for task in workflow.tasks:
        pair = {}
        for split in ['Train', 'Test']:
            frame = read_fasta(workflow.root/'data/fasta'/task/f'{split}.fasta')
            pair[split] = frame
            all_records.append(frame.assign(task=task, split=split))
            counts.append(dict(task=task, split=split, samples=len(frame), positive=int(frame.Target.sum()),
                negative=int(frame.Target.eq(0).sum()), extra_duplicates=int(frame.sequence.duplicated().sum()),
                duplicate_IDs=int(frame.PID.duplicated().sum()), normalized_rows=int(frame.raw_sequence.ne(frame.sequence).sum())))
            if frame.Target.nunique() != 2:
                issues.append(dict(task=task, split=split, issue='not_binary_classes', sequence=''))
            bad = frame.sequence.duplicated(keep=False) | frame.PID.duplicated(keep=False)
            issues.extend(dict(task=task, split=split, issue='within_split_duplicate', sequence=s) for s in frame.loc[bad, 'sequence'])
            if split == 'Train' and frame.Target.value_counts().min() < workflow.config['folds']:
                raise ValueError(f'{task}: too few class members for requested folds')
        shared = set(pair['Train'].sequence) & set(pair['Test'].sequence)
        issues.extend(dict(task=task, split='Train/Test', issue='identical_sequence', sequence=s) for s in sorted(shared))
        both = pd.concat(pair.values())
        bad = both.groupby('sequence').Target.nunique().loc[lambda x: x > 1].index
        issues.extend(dict(task=task, split='Train/Test', issue='label_conflict', sequence=s) for s in bad)
    workflow.csv(counts, 'dataset_counts.csv')
    workflow.csv(pd.DataFrame(issues, columns=['task','split','issue','sequence']), 'audit_issues.csv')
    all_records = pd.concat(all_records, ignore_index=True)
    shared = all_records.groupby('sequence').task.nunique().loc[lambda x: x > 1].index
    workflow.csv(all_records[all_records.sequence.isin(shared)], 'cross_task_shared_sequences.csv')
    fatal = [
        row for row in issues if row['issue'] in ['not_binary_classes','label_conflict']]
    if fatal:
        raise ValueError('Data audit failed; inspect audit_issues.csv. Source files are NOT modified.')

def frame(workflow, task, split):
    frame = pd.read_csv(workflow.run/'features'/f'{task}_{split}.csv',dtype={'PID':str})
    groups = json.loads((workflow.run/'feature_columns.json').read_text())
    expected = [c for cols in groups.values() for c in cols]
    if frame.columns[2:].tolist() != expected or not np.isfinite(frame.iloc[:,2:].to_numpy(float)).all():
        raise ValueError('Invalid feature matrix or column order')
    return frame

def extract(workflow):
    feature_dir = workflow.run/'features'
    feature_dir.mkdir(exist_ok=True)
    groups = None
    quality = []
    for task in workflow.tasks:
        for split in ['Train','Test']:
            records = read_fasta(workflow.root/'data/fasta'/task/f'{split}.fasta')
            tables = extract_all_features(records.rename(columns={'PID':'id','Target':'label'}))
            frame = merge_feature_tables(tables).rename(columns={'id':'PID','label':'Target'})
            assert frame.PID.tolist() == records.PID.tolist()
            np.testing.assert_array_equal(frame.Target, records.Target)
            columns = {name: table.columns[2:].tolist() for name, table in tables.items()}
            if groups is None:
                groups = columns
            assert columns == groups and frame.shape[1] - 2 == 2125
            values = frame.iloc[:,2:].to_numpy(float)
            if not np.isfinite(values).all():
                raise ValueError(f'Non-finite features: {task}/{split}')
            frame.to_csv(feature_dir/f'{task}_{split}.csv',index=False)
            records.to_csv(feature_dir/f'{task}_{split}_sequences.csv',index=False)
            quality.append(dict(task=task, split=split, samples=len(frame), dimensions=values.shape[1],
                missing=int(np.isnan(values).sum()), infinite=int(np.isinf(values).sum())))
            print(f'  {task}/{split}: {values.shape}', flush=True)
    write_json(workflow.run/'feature_columns.json', groups)
    workflow.csv(quality, 'feature_quality.csv')
    rows = []
    start = 0
    for name, cols in groups.items():
        rows.append(dict(group=name, dimension=len(cols), first=cols[0], last=cols[-1], start=start, stop=start+len(cols)))
        start += len(cols)
    workflow.csv(rows, 'feature_groups.csv')

def split(workflow):
    rows, counts = [], []
    for task in workflow.tasks:
        train = workflow.frame(task,'Train')
        seq = pd.read_csv(workflow.run/'features'/f'{task}_Train_sequences.csv',dtype={'PID':str})
        assert train.PID.tolist() == seq.PID.tolist()
        y = train.Target.to_numpy(int)
        if seq.sequence.is_unique:
            splitter = StratifiedKFold(workflow.config['folds'],shuffle=True,random_state=workflow.config['seed'])
            assignments = splitter.split(train,y)
        else:
            splitter = StratifiedGroupKFold(workflow.config['folds'],shuffle=True,random_state=workflow.config['seed'])
            assignments = splitter.split(train,y,groups=seq.sequence.to_numpy())
        for fold,(tr,va) in enumerate(assignments,1):
            assert set(seq.sequence.iloc[tr]).isdisjoint(seq.sequence.iloc[va])
            rows.extend(dict(task=task,row=int(i),PID=train.PID.iloc[i],Target=int(y[i]),fold=fold) for i in va)
            counts.append(dict(task=task,fold=fold,train=len(tr),train_positive=int(y[tr].sum()),
                train_negative=int((y[tr]==0).sum()),validation=len(va),val_positive=int(y[va].sum()),
                val_negative=int((y[va]==0).sum()),AGN_train=len(tr)*(workflow.config['agn']['copies']+int(workflow.config['agn']['keep_original']))))
    workflow.csv(rows,'fold_assignments.csv')
    workflow.csv(counts,'fold_counts.csv')

def folds(workflow, task):
    frame = workflow.frame(task,'Train')
    assignments = pd.read_csv(workflow.run/'fold_assignments.csv',dtype={'PID':str})
    a = assignments[assignments.task==task].sort_values('row')
    if a.row.tolist() != list(range(len(frame))) or a.PID.tolist() != frame.PID.tolist():
        raise ValueError('Fold assignment does not match training sample order')
    np.testing.assert_array_equal(a.Target,frame.Target)
    return [(a.loc[a.fold!=fold,'row'].to_numpy(),a.loc[a.fold==fold,'row'].to_numpy())
            for fold in range(1,workflow.config['folds']+1)]
