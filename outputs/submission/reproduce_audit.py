import os, json, hashlib, contextlib, io
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
root = Path.cwd()
protected = ['notebooks/Avangard_ticket_demand.ipynb', 'outputs/predictions.csv', 'docs/PROTOCOL.md']
protected += [str(p.relative_to(root)) for p in (root/'data').glob('*') if p.is_file()]
hashes = lambda: {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in protected}
before = hashes()
nb = json.loads((root/protected[0]).read_text(encoding='utf-8'))
ns = {}
with contextlib.redirect_stdout(io.StringIO()):
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and i != 25:
            exec(compile(''.join(cell['source']), f'cell_{i}', 'exec'), ns)
import numpy as np
import pandas as pd
test = ns['template'][ns['keys']].merge(ns['calendar'], on='match_id', how='left', sort=False).merge(ns['zones'], on=ns['keys'], how='left', sort=False)
test['cutoff'] = ns['forecast_date']
test['has_sales'] = test['sales_open'] < ns['forecast_date']
test['sold'] = [ns['sales_before'](r.match_id, r.zone, ns['forecast_date']) for r in test.itertuples()]
history = ns['train'].loc[ns['train']['date'] < ns['forecast_date']]
pred = ns['apply_bounds'](ns['forecast_tickets'](history, test), test['sold'], test['seats']-test['season_tickets'])
existing = pd.read_csv(root/'outputs/predictions.csv')
checks = {'rows_136':len(existing)==136, 'columns':existing.columns.tolist()==ns['template'].columns.tolist(), 'order':existing[ns['keys']].equals(ns['template'][ns['keys']]), 'unique':not existing.duplicated(ns['keys']).any(), 'finite':bool(np.isfinite(existing['tickets_total']).all()), 'integer':bool((existing['tickets_total']%1==0).all()), 'nonnegative':bool((existing['tickets_total']>=0).all()), 'lower':bool((existing['tickets_total']>=test['sold']).all()), 'upper':bool((existing['tickets_total']<=test['seats']-test['season_tickets']).all()), 'reproduced':bool(np.array_equal(pred,existing['tickets_total']))}
assert all(checks.values()), checks
assert before == hashes(), 'Protected artifact changed'
ns['validation'].to_csv(root/'outputs/submission/audit_validation.csv', index=False)
ns['window_results'].to_csv(root/'outputs/submission/audit_window_results.csv')
ns['scores'].to_csv(root/'outputs/submission/audit_scores.csv')
report = {'checks':checks,'hashes_before':before,'hashes_after':hashes(),'history_matches':history['match_id'].nunique(),'test_matches_with_sales':test.loc[test['has_sales'],'match_id'].nunique(),'scores':ns['scores'].reset_index().to_dict('records'),'validation_rows_per_method':ns['validation'].groupby('method').size().to_dict()}
(root/'outputs/submission/audit_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if not k.startswith('hashes')},ensure_ascii=False,indent=2))
