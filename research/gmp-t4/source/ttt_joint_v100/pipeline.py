"""Single owner: fit, independently audit, evaluate, render/report. No polling loop."""
import subprocess,sys,json,time
from common import R,O,write
def main():
    for script in ['train.py','audit.py','evaluate.py','report.py']:
        print('PIPELINE_STAGE',script,flush=True)
        subprocess.run([sys.executable,str(R/'ttt_joint_v100'/script)],check=True)
    write(O/'pipeline_complete.json',dict(status='completed',finished_unix=time.time()))
if __name__=='__main__':main()
