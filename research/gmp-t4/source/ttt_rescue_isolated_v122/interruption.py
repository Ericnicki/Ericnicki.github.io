from review import A
from audit_results import *
import json
setup();old=R/'ttt_rescue_confirm_v120_output';rows=[]
for i in range(4):
 p=old/f'worker{i}';b=json.loads((p/'budget.json').read_text());rows.append(dict(worker=i,evaluation_starts=b['evaluation_starts'],calibration_starts=b['near_starts'],complete_tasks=len(json.loads((p/'summary.json').read_text())) if (p/'summary.json').exists() else 0))
write(old/'superseded.json',dict(reason='Unequal inherited optimizer counters; corrected hindsight tensors differ. Workers suspended then terminated before corrected confirmation.',fit_source='ttt_rescue_isolated_v122/fit.py',replacement='ttt_rescue_confirm_v123_output',workers=rows,old_worker_pids=[1718,1719,1720,1721],parent_pid=1484,reusable='Only saved baseline arms and bitwise-equal corrected learned endpoints, with same initial snapshot and speed. Original partial output preserved.',source_sha256=sha(Path(__file__))));print(json.dumps(rows,indent=2))
