from fit import *
setup(2178000)
if json.loads((O/'manifest.json').read_text())['conditional_gate_passed']:
 tasks=tasks_load();ids=sorted(tid for tid,t in tasks.items() if t['spec']['partition']=='train')
 for arm in ARMS:
  setup(2178000);w,v,opt=networks();rng=np.random.default_rng(2178100);rows=torch.load(O/f'{arm}_world_samples.pt',map_location='cpu',weights_only=False);assert len(rows)==WORLD_STEPS
  for row in rows:
   qs=sample(tasks,ids,rng,12);assert qs==row['queries'];loss,parts=objective(w,v,batch(tasks,qs),arm=='joint');assert float(loss.detach())==row['loss'] and parts==row['parts'];opt.zero_grad();loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(w.parameters(),1));assert norm==row['norm'];opt.step();assert all(p.grad is None for p in v.parameters())
  ck=torch.load(O/f'{arm}_world.pt',map_location='cpu',weights_only=False);equal(w.state_dict(),ck['world']);equal(v.state_dict(),ck['value']);equal(opt.state_dict(),ck['optimizer']);equal(rng.bit_generator.state,ck['sampler_rng']);w.zero_grad(set_to_none=True);freeze(w);cache,scale=private(w,tasks,ids);stored=torch.load(O/f'{arm}_private.pt',map_location='cpu',weights_only=False);equal(cache,stored['cache']);equal(scale,stored['scale']);setup(2178200);actor=Correction().cuda();equal(actor.state_dict(),torch.load(O/f'{arm}_actor_initial.pt',map_location='cpu',weights_only=False));opt=torch.optim.Adam(actor.parameters(),lr=1e-4);rng=np.random.default_rng(2178300);rows=torch.load(O/f'{arm}_actor_samples.pt',map_location='cpu',weights_only=False);assert len(rows)==ACTOR_STEPS
  for row in rows:
   qs=sample(tasks,ids,rng,24);assert qs==row['queries'];loss=actor_loss(actor,w,v,tasks,qs,cache,scale);assert float(loss.detach())==row['loss'];opt.zero_grad();loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(actor.parameters(),1));assert norm==row['norm'];opt.step();assert all(p.grad is None for p in list(w.parameters())+list(v.parameters()))
  ck=torch.load(O/f'{arm}_actor.pt',map_location='cpu',weights_only=False);equal(actor.state_dict(),ck['correction']);equal(opt.state_dict(),ck['optimizer']);equal(rng.bit_generator.state,ck['sampler_rng']);print('SHARED_OUTER_ARM_EXACT_AUDIT',arm,flush=True)
 write(O/'audit.json',dict(status='passed',world_updates_each=1000,actor_updates_each=1000,all_optimizer_samples_losses_tensors_moments_sampler_states_exact=True,private_W_and_belief_scales_reconstructed=True,frozen_value_world_and_actor_candidate_isolation=True,sources=sources()))
else:write(O/'audit.json',dict(status='skipped',sources=sources()))
