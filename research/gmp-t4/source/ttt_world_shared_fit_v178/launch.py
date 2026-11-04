import subprocess,time
from common import *
assert not O.exists();pf=R/'ttt_world_shared_fit_v178_preflight.json';f=json.loads(pf.read_text());assert f['status']=='passed' and f['sources']==sources()
with (R/'ttt_world_shared_fit_v178_pipeline.log').open('x') as stream:p=subprocess.Popen(['bash',str(S/'run.sh')],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pipeline_pid=p.pid,started_unix=time.time(),sources=sources(),preflight_sha256=sha(pf),waiting_native_parent=40063);write(R/'ttt_world_shared_fit_v178_launch.json',receipt);print(json.dumps(receipt))
