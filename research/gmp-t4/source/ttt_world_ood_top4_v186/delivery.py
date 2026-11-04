import tarfile
from common import *
assert json.loads((O/'plot_audit.json').read_text())['status']=='passed'
files={str(p.relative_to(R)):sha(p) for root in [O,S,OLD,R/'ttt_world_ood_top4_v185',R/'ttt_world_ood_top4_v185_fixture_output'] for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and not p.name.startswith('._')}
parents=json.loads((O/'manifest.json').read_text())['parents']
for rel,digest in parents.items():assert sha(R/rel)==digest;files[rel]=digest
for p in [R/'ttt_world_ood_top4_v186_launch.json',R/'ttt_world_ood_top4_v185_launch.json',R/'ttt_world_ood_top4_v185_preflight.json',R/'ttt_world_ood_top4_v185_boundary_stop.json',R/'ttt_world_ood_top4_v185_preflight.log',R/'ttt_world_ood_top4_v185_collect.log']+list(R.glob('ttt_world_ood_top4_v186_*.log')):
 if not p.name.endswith(('pipeline.log','delivery.log')):files[str(p.relative_to(R))]=sha(p)
manifest=R/'ttt_world_ood_top4_v186_manifest.json';write(manifest,dict(status='bound',files=files,sources=sources(),count=len(files),ID_tasks=0,optimizer_updates=0))
archive=R/'ttt_world_ood_top4_v186.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as tar:
 for rel in files:tar.add(R/rel,arcname=rel,recursive=False)
 tar.add(manifest,arcname=manifest.name,recursive=False)
receipt=dict(status='completed',archive=str(archive),sha256=sha(archive),files=len(files)+1,manifest_sha256=sha(manifest),sources=sources());write(R/'ttt_world_ood_top4_v186_delivery.json',receipt);print(json.dumps(receipt),flush=True)
