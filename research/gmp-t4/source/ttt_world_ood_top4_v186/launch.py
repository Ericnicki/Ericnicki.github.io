import subprocess,time
from common import *
assert not O.exists();assert json.loads((R/'ttt_world_ood_top4_v185_boundary_stop.json').read_text())['interrupted_policy_attempts']==0
with (R/'ttt_world_ood_top4_v186_pipeline.log').open('x') as stream:p=subprocess.Popen(['bash',str(S/'run.sh')],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pipeline_pid=p.pid,started_unix=time.time(),sources=sources());write(R/'ttt_world_ood_top4_v186_launch.json',receipt);print(json.dumps(receipt))
