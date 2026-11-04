import sys,subprocess,shutil
from pathlib import Path
SUP=Path(__file__).resolve().parent;sys.path.insert(0,str(SUP.parent/'gmp_memory_ood_v187'))
from common import *
assert json.loads((O/'audit.json').read_text())['status']==json.loads((O/'independent_review.json').read_text())['status']=='passed';assert not (R/'gmp_memory_ood_v191_launch.json').exists() and not (O/'result.json').exists();snapshot=O/'failed_reporting_v187';snapshot.mkdir()
for name in ['plots','native_outcomes.csv','memory_decisions.csv']:shutil.move(str(O/name),str(snapshot/name))
with (R/'gmp_memory_ood_v191_pipeline.log').open('xb') as f:p=subprocess.Popen(['bash',str(SUP/'run.sh')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
write(R/'gmp_memory_ood_v191_launch.json',dict(pid=p.pid,sources=sources(),supplement_sources={q.name:sha(q) for q in SUP.iterdir() if q.suffix in ('.py','.md','.sh')},new_policy_slots=0,new_physics_starts=0,node=socket.gethostname()));print(p.pid,flush=True)
