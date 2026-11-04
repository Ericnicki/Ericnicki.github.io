import sys,subprocess
from pathlib import Path
SUP=Path(__file__).resolve().parent;sys.path.insert(0,str(SUP.parent/'gmp_memory_ood_v187'))
from common import *
assert (O/'collection_complete.json').exists() and not (R/'gmp_memory_ood_v190_launch.json').exists();manifest=json.loads((O/'manifest.json').read_text());assert manifest['sources']==sources()
with (R/'gmp_memory_ood_v190_pipeline.log').open('xb') as f:p=subprocess.Popen(['bash',str(SUP/'run.sh')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
write(R/'gmp_memory_ood_v190_launch.json',dict(pid=p.pid,sources=sources(),supplement_sources={q.name:sha(q) for q in SUP.iterdir() if q.suffix in ('.py','.md','.sh')},node=socket.gethostname(),new_policy_slots=0));print(p.pid,flush=True)
