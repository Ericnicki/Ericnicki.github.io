import json,subprocess,time
from common import R,O
assert not O.exists()
with (R/'ttt_joint_v100.log').open('xb') as log:
    p=subprocess.Popen(['bash',str(R/'endpoint_newnode_runtime_v1/runtime.sh'),'ttt_joint_v100/pipeline.py'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=p.pid,started_unix=time.time(),training_tasks=500,trials=3000,actor_updates=3150,Q_updates=6300,slow_writer_updates=630)
(R/'ttt_joint_v100_launch.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
