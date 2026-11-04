from fit import *
setup();rows=json.loads((D/'summary.json').read_text());rows=rows[:16];replay={};raw={}
for row in rows:
 tid=row['spec']['task'];d=torch.load(D/f'task{tid:04d}/task.pt',weights_only=False);replay[tid]=replay_task(d);raw[tid]=d
m,a=load();teacher,_=load()
for p in list(m.parameters())+list(a.parameters()):p.requires_grad_(True)
rng=np.random.default_rng(2039000);qs=sample(replay,rng,actor=True);h,mask,s,ac,y=hb(replay,[q[:3] for q in qs],True,'cuda')
ctx=m.context(h,mask,s);earlyq=m.head(torch.cat([ctx,s,ac],-1)).flatten();mc=F.smooth_l1_loss(earlyq,y)
teacherctx=[];stream_errors=[]
for tid,ti,ix,_ in qs:
 ts=raw[tid]['trials'];mem=rebuild(teacher,ts[:ti]+ts[ti+1:]);rr=ts[ti]['records']
 for rec in rr[:ix]:mem.observe(rec['event'])
 tc=mem.context(rr[ix]['state']);stream_errors.append(float((tc-mem.reference(rr[ix]['state'])).abs().max()));teacherctx.append(tc)
tc=torch.cat(teacherctx);old,prev,delta=controls(replay,qs)
def candidates(context,critic):
 # Same physically feasible action endpoints, actor-independent constant requests.
 vs=[((old+(sign*.05)).clamp(-.1,.1)+old)/2 for sign in [-1.,1.]]
 dv=[((prev+delta+.1*v).clamp(-.85,.25)-prev-delta)/.1 for v in vs]
 return [critic.head(torch.cat([context,s,(v/.5).unsqueeze(-1)],-1)).flatten() for v in dv]
pq=candidates(ctx,m)
with torch.no_grad():tq=candidates(tc,teacher)
aux=F.smooth_l1_loss(((pq[1]-pq[0])-(tq[1]-tq[0]))/.1,torch.zeros_like(y));loss=mc+.1*aux;loss.backward()
writer_norm=float(torch.nn.utils.clip_grad_norm_(m.writer_parameters(),float('inf')));head_norm=float(torch.nn.utils.clip_grad_norm_(m.head.parameters(),float('inf')))
assert writer_norm>0 and head_norm>0 and all(p.grad is None for p in teacher.parameters()) and all(p.grad is None for p in a.parameters());assert max(stream_errors)<3e-5
m.zero_grad(set_to_none=True)
for p in m.parameters():p.requires_grad_(False)
dv,acc,_=actor_action(a,ctx.detach(),s,old,prev,delta);actorloss=-m.head(torch.cat([ctx.detach(),s,(dv/.5).unsqueeze(-1)],-1)).mean();actorloss.backward();assert all(p.grad is None for p in m.parameters());an=float(torch.nn.utils.clip_grad_norm_(a.parameters(),float('inf')));assert an>0
receipt=dict(status='passed',student_writer_gradient_norm=writer_norm,student_head_gradient_norm=head_norm,actor_gradient_norm=an,teacher_gradients_absent=True,actor_to_writer_gradients_absent=True,teacher_stream_reference_max_error=max(stream_errors),optimizer_operations=0,source_sha256=sha(Path(__file__)),fit_source_sha256=sha(Path(__file__).with_name('fit.py')))
write(D/'gradient_fixture.json',receipt);print(json.dumps(receipt,indent=2))
