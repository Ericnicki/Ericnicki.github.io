import sys
from pathlib import Path
R=Path('/public/home/2023141520023/return_selector_20261006');sys.path.insert(1,str(R/'ttt_rescue_v117'))
from rescue_common import *
import json
setup();old=R/'ttt_rescue_fit_v119_output';new=R/'ttt_rescue_fit_v122_output';rows=[]
for seed in [2033000,2033001]:
 for arm in ['mc','hindsight','grounded']:
  key=f'{arm}_{seed}';p=new/key/'step1500.pt'
  if not p.exists():continue
  a=torch.load(p,map_location='cpu',weights_only=False);b=torch.load(old/key/'step1500.pt',map_location='cpu',weights_only=False);row=dict(arm=arm,seed=seed)
  for group in ['model','actor']:
   row[group+'_exact']=all(torch.equal(v,b[group][k]) for k,v in a[group].items());row[group+'_max_abs_difference']=max(float((v-b[group][k]).abs().max()) for k,v in a[group].items())
  rows.append(row)
write(new/'prior_endpoint_comparison.json',dict(source_sha256=sha(Path(__file__)),rows=rows));print(json.dumps(rows,indent=2))
