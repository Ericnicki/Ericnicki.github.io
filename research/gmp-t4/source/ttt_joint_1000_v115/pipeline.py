"""Owned continuation: training, independent audit, matched eval and three plots."""
import json,subprocess,sys,time
from common import R,O,write
def main():
    script='train.py'
    try:
        for script in ['recover.py','train.py','audit.py','evaluate.py','report.py']:
            print('PIPELINE_STAGE',script,flush=True)
            command=[sys.executable,str(R/'ttt_joint_1000_v115'/script)]
            if script=='train.py':command.append('--resume')
            subprocess.run(command,check=True)
            if script=='train.py' and not (O/'training_complete.json').exists():
                assert (O/'stopped.json').exists();write(O/'pipeline_stopped.json',dict(status='stopped'));return
        write(O/'pipeline_complete.json',dict(status='completed',finished_unix=time.time()))
    except BaseException as e:
        if O.exists():write(O/'pipeline_failed.json',dict(stage=script,error=repr(e),finished_unix=time.time()))
        raise
if __name__=='__main__':main()
