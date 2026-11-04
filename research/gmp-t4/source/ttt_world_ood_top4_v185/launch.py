import subprocess,time
from common import *
assert not O.exists()
with (R/'ttt_world_ood_top4_v185_pipeline.log').open('x') as stream:p=subprocess.Popen(['bash',str(S/'run.sh')],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pipeline_pid=p.pid,started_unix=time.time(),sources=sources());write(R/'ttt_world_ood_top4_v185_launch.json',receipt);print(json.dumps(receipt))
