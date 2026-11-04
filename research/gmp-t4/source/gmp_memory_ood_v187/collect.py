import time,shutil,sys
from common import *
from policy import Official
from trial import environment,run_trial

def main():
 worker=int(sys.argv[1]);setup();pf=json.loads((R/'gmp_memory_ood_v187_preflight.json').read_text());assert pf['status']=='passed' and pf['sources']==sources() and pf['upstream_sources']==upstream_files();manifest=json.loads((O/'manifest.json').read_text());env=environment();p=Official();mh=state_hash(p.model);nh=state_hash(p.norm);rows=[];begin=time.time();starts=0
 try:
  for parent in json.loads((B/'summary.json').read_text()):
   tid=parent['spec']['task']
   if tid%4!=worker:continue
   spec=parent['spec'];assert spec['friction']<.005 or spec['friction']>.015;folder=O/f'task{tid:04d}';folder.mkdir();shutil.copy2(B/f'task{tid:04d}/initial.pt',folder/'initial.pt');start=torch.load(folder/'initial.pt',map_location='cpu',weights_only=False);p.new_task(tid);trials=[]
   for ti in range(12):
    starts+=1;assert starts<=96;write(O/f'worker{worker}_budget.json',dict(starts=starts,task=tid,trial=ti+1));tr=run_trial(env,start,p,dict(task=tid,trial=ti+1,method='official_GMP_memory'),folder);trials.append(tr);save(folder/f'gmp_t{ti+1:02d}.pt',tr)
   assert state_hash(p.model)==mh and state_hash(p.norm)==nh;save(folder/'gmp.pt',dict(spec=spec,trials=trials,final_history=p.history()));row=dict(spec=spec,initial_sha256=sha(folder/'initial.pt'),raw_sha256=sha(folder/'gmp.pt'),returns=[t.get('terminal_return') for t in trials],statuses=[t['status'] for t in trials],history_end=p.history());rows.append(row);write(O/f'worker{worker}_summary.json',rows);write(O/f'worker{worker}_status.json',dict(status='running',tasks=len(rows),starts=starts,seconds=time.time()-begin));print('OFFICIAL_GMP_OOD_TASK',worker,tid,spec['region'],starts,row['returns'],flush=True)
  assert starts==96;write(O/f'worker{worker}_complete.json',dict(status='completed',trials=96,model_sha256=mh,normalizer_sha256=nh,sources=sources(),upstream_sources=upstream_files(),seconds=time.time()-begin))
 except BaseException as e:write(O/f'worker{worker}_failed.json',dict(error=repr(e),starts=starts));raise
 finally:env.physics.free()
if __name__=='__main__':main()
