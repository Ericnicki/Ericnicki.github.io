from review import A
from audit_results import *
import json
rows=[]
for i in range(4):
 p=CONF/f'worker{i}/summary.json'
 if p.exists():rows+=json.loads(p.read_text())
first=[(r['spec']['task'],arm) for r in rows for arm,a in r['arms'].items() if a['returns'][0] is None]
print(dict(completed_tasks=len(rows),judged=sum(a['judged'] for r in rows for a in r['arms'].values()),slots=66*len(rows),first_trial_censored=first))
