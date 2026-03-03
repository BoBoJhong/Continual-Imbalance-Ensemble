"""
Experiment 22: Stock - Threshold Tuning + Cost-Sensitive Learning
"""
import sys, pandas as pd, numpy as np
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from src.utils import set_seed, get_logger
from src.data import ImbalanceSampler
from src.models import LightGBMWrapper, ModelPool
from src.evaluation import compute_metrics
from experiments._shared.common_dataset import get_splits
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
SCALE_POS_WEIGHTS = [5, 10, 20, 50]

def eval_thresholds(name, y_true, y_proba):
    rows = []
    for thr in THRESHOLDS:
        m = compute_metrics(y_true, y_proba, threshold=thr)
        m.update({'threshold': thr, 'method': name})
        rows.append(m)
    return rows

def main():
    logger = get_logger('Stock_Threshold', console=True, file=True)
    set_seed(42)
    logger.info('='*70)
    logger.info('Experiment 22: Stock Threshold + Cost-Sensitive')
    logger.info('='*70)
    X_h, y_h, X_n, y_n, X_t, y_t = get_splits('stock', logger)
    y_ta = np.asarray(y_t)
    sampler = ImbalanceSampler()
    rows = []

    # Fine-tune hybrid
    Xhr, yhr = sampler.apply_sampling(X_h, y_h.values, strategy='hybrid')
    m = LightGBMWrapper(name='ft_hybrid'); m.fit(Xhr, yhr)
    Xnr, ynr = sampler.apply_sampling(X_n, y_n.values, strategy='hybrid')
    m.fit(Xnr, ynr)
    rows += eval_thresholds('finetune_hybrid', y_ta, m.predict_proba(X_t))

    # Ensemble Old-3
    op = ModelPool(pool_name='old'); op.create_pool(X_h, y_h.values, prefix='old')
    np_ = ModelPool(pool_name='new'); np_.create_pool(X_n, y_n.values, prefix='new')
    ap = {**op.predict_proba(X_t), **np_.predict_proba(X_t)}
    p3 = np.mean([ap[k] for k in ['old_under','old_over','old_hybrid']], axis=0)
    rows += eval_thresholds('ensemble_old_3', y_ta, p3)
    pn3 = np.mean([ap[k] for k in ['new_under','new_over','new_hybrid']], axis=0)
    rows += eval_thresholds('ensemble_new_3', y_ta, pn3)
    rows += eval_thresholds('ensemble_all_6', y_ta, np.mean(list(ap.values()), axis=0))

    # Cost-sensitive (scale_pos_weight)
    X_c = pd.concat([X_h, X_n]); y_c = pd.concat([y_h, y_n])
    for spw in SCALE_POS_WEIGHTS:
        Xr, yr = sampler.apply_sampling(X_c, y_c.values, strategy='hybrid')
        mc = LightGBMWrapper(name=f'cost_spw{spw}', is_unbalance=False, scale_pos_weight=spw)
        mc.fit(Xr, yr)
        r = eval_thresholds(f'cost_spw{spw}', y_ta, mc.predict_proba(X_t))
        best = max(r, key=lambda x: x['F1'])
        logger.info(f'  scale_pos_weight={spw}: best F1={best["F1"]:.4f} @ thr={best["threshold"]}')
        rows += r

    df = pd.DataFrame(rows)
    out = project_root / 'results/phase5_analysis'
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / 'stock_threshold_cost_study.csv', index=False)
    best_per = df.loc[df.groupby('method')['F1'].idxmax()]
    logger.info('\n=== Best threshold per method ===')
    logger.info(best_per[['method','threshold','AUC','F1','Recall','Type1_Error']].to_string())
    return df

if __name__ == '__main__':
    main()
