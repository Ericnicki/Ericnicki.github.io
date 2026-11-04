import subprocess
from common import *
assert not (R/'gmp_memory_ood_v187_launch.json').exists()
with (R/'gmp_memory_ood_v187_pipeline.log').open('xb') as f:p=subprocess.Popen(['bash',str(S/'run.sh')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
write(R/'gmp_memory_ood_v187_launch.json',dict(pid=p.pid,sources=sources(),node=socket.gethostname(),status='launched'));print(p.pid,flush=True)
