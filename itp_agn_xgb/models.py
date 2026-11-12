"""AGN augmentation, classifier construction and soft-voting parameters."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def augment(X, y, config):
    """Add noise in feature units; parent indices refer to the supplied training subset."""
    copies, sigma = config['agn']['copies'], config['agn']['sigma']
    parent = np.repeat(np.arange(len(y)), copies)
    data = np.asarray(X, dtype=float)[parent].copy()
    data += np.random.RandomState(config['seed']).normal(0, sigma, data.shape)
    if config['agn']['keep_original']:
        data = np.vstack([X, data])
        parent = np.concatenate([np.arange(len(y)), parent])
    return data, np.asarray(y)[parent], parent

def make_model(kind, params, config, seed=None):
    seed = config['seed'] if seed is None else seed
    if kind == 'XGBoost':
        return XGBClassifier(**params, objective='binary:logistic', tree_method='hist',
            eval_metric='logloss', n_jobs=config['n_jobs'], random_state=seed,
            use_label_encoder=False)
    if kind == 'LR':
        return LogisticRegression(**params, solver='lbfgs', max_iter=5000, random_state=seed)
    if kind == 'KNN':
        return KNeighborsClassifier(**params, n_jobs=config['n_jobs'])
    raise ValueError(kind)

def fit_model(X, y, kind, params, mode, config, seed=None):
    scaler = StandardScaler().fit(X)
    data, labels, _ = augment(X, y, config) if mode == 'AGN' else (X, y, None)
    model = make_model(kind, params, config, seed)
    model.fit(scaler.transform(data).astype('float32'), labels)
    return scaler, model

def predict(scaler, model, X):
    return model.predict_proba(scaler.transform(X).astype('float32'))[:, 1]

def perturbed_params(base, member, seed):
    """Apply a fixed +/-10% parameter perturbation rule for voting members."""
    rng = np.random.RandomState(seed + member * 100)
    params = base.copy()
    for name, low, high in [('n_estimators', 10, 2000), ('max_depth', 3, 20)]:
        params[name] = int(np.clip(int(params[name] * rng.uniform(.9, 1.1)), low, high))
    for name in ['learning_rate', 'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda']:
        limits = (.001, 1.) if name == 'learning_rate' else ((.1, 1.) if name in ['subsample', 'colsample_bytree'] else (.001, 10.))
        params[name] = float(np.clip(params[name] * rng.uniform(.9, 1.1), *limits))
    return params
