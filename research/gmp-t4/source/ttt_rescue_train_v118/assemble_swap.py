from audit_results import *
setup();m,a=load('new','old')
p=FITS/'actor_v100_writer_v115_inference.pt'
assert not p.exists()
save(p,dict(model={k:v.cpu() for k,v in m.state_dict().items()},actor={k:v.cpu() for k,v in a.state_dict().items()},inference_only=True,writer_and_head_parent=HASHES['new'],actor_parent=HASHES['old'],writer_steps=13880,Q_steps=88800,actor_steps=9450,description='Frozen component swap evaluated in v117 and v120; no optimizer resume state; no new training updates.'))
write(FITS/'component_swap_receipt.json',dict(sha256=sha(p),path=str(p),source_sha256=sha(Path(__file__)),new_optimizer_updates=0));print(sha(p))
