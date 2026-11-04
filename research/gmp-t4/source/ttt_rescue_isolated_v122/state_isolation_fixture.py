from compare import *
import copy
parent=torch.load(PATHS['new'],map_location='cpu',weights_only=False);m,a=load();rows=[]
for name,params,cls in [('head_optimizer',list(m.head.parameters()),torch.optim.AdamW),('writer_optimizer',list(m.writer_parameters()),torch.optim.AdamW),('actor_optimizer',list(a.parameters()),torch.optim.Adam)]:
 original=cls(params);original.load_state_dict(parent[name]);first_id=next(iter(parent[name]['state']));s=parent[name]['state'][first_id]['step'];old_count=int(s);alias=original.state[params[0]]['step'].data_ptr()==s.data_ptr();assert alias
 isolated=cls(params);isolated.load_state_dict(copy.deepcopy(parent[name]));v=isolated.state[params[0]]['step'];assert v.data_ptr()!=s.data_ptr();v.add_(1);assert int(s)==old_count
 rows.append(dict(group=name,old_scalar_alias=alias,deepcopy_scalar_independent=True,parent_count=old_count))
write(new/'state_isolation_fixture.json',dict(status='passed',groups=rows,optimizer_operations=0,source_sha256=sha(Path(__file__))));print(json.dumps(rows,indent=2))
