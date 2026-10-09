import json, io, contextlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
root=Path.cwd(); ns={}
nb=json.loads((root/'notebooks/Avangard_ticket_demand.ipynb').read_text(encoding='utf-8'))
with contextlib.redirect_stdout(io.StringIO()):
    for i,c in enumerate(nb['cells'][:14]):
        if c['cell_type']=='code': exec(''.join(c['source']),ns)
h=ns['train']; cats=ns['categorical_features']; nums=ns['numeric_features']
pre=ColumnTransformer([('categories',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats),('number',StandardScaler(),nums)])
model=make_pipeline(pre,Ridge(alpha=10,random_state=42))
centers=h.groupby(['zone','season']).tickets_total.transform('mean')
model.fit(ns['model_features'](h),h.tickets_total-centers)
X=pre.transform(ns['model_features'](h)); coef=model[-1].coef_; offset=0; rows=[]
base=ns['zone_mean'](h,h); level=ns['season_level'](h,h)
effects={'zone_base':base-base.mean(),'season_adjustment':level-base}
for name,values in zip(cats,pre.named_transformers_['categories'].categories_):
    part=X[:,offset:offset+len(values)]@coef[offset:offset+len(values)]
    effects[name]=part-part.mean(); offset+=len(values)
effects['season_match_number']=X[:,offset]*coef[offset]
for name,v in effects.items():
    rows.append({'group':name,'mean_abs_effect':float(np.mean(np.abs(v))),'rms_effect':float(np.sqrt(np.mean(v*v))),'min_effect':float(v.min()),'max_effect':float(v.max())})
summary={'method':'Descriptive decomposition of fixed final model on85 historical matches; contribution magnitude in tickets, not causal importance or validation gain. Zone baseline centered on overall mean; seasonal adjustment relative to zone baseline; Ridge categorical contributions centered on training mean. All8zones per match equal weight.','effects':rows,'zone_means':h.groupby('zone').tickets_total.agg(['mean','count']).reset_index().to_dict('records'),'season_zone_means':h.groupby(['season','zone']).tickets_total.agg(['mean','count']).reset_index().to_dict('records')}
(root/'outputs/submission/DEMAND_FACTORS.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(pd.DataFrame(rows).sort_values('mean_abs_effect',ascending=False).round(2).to_string(index=False))
print(h.groupby('zone').tickets_total.mean().round(1).to_dict())
print(h.groupby('season').tickets_total.mean().round(1).to_dict())
