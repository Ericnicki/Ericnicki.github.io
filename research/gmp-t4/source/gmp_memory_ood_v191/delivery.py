import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"gmp_memory_ood_v187"))
import tarfile
from common import *
assert json.loads((O/'status.json').read_text())['status']=='completed';assert sha(CK)==CK_SHA
files={str(p.relative_to(R)):sha(p) for root in [O,S,R/'gmp_memory_ood_v188',R/'gmp_memory_ood_v189',R/'gmp_memory_ood_v190',R/'gmp_memory_ood_v191',R/'gmp_memory_ood_v187_fixture_output'] for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and not p.name.startswith('._')}
for p in [R/'gmp_memory_ood_v187_launch.json',R/'gmp_memory_ood_v187_preflight.json']+list(R.glob('gmp_memory_ood_v187_*.log'))+list(R.glob('gmp_memory_ood_v188_*.log'))+list(R.glob('gmp_memory_ood_v189_*.log'))+list(R.glob('gmp_memory_ood_v190_*.log'))+list(R.glob('gmp_memory_ood_v191_*.log'))+[R/'gmp_memory_ood_v188_launch.json',R/'gmp_memory_ood_v189_launch.json',R/'gmp_memory_ood_v190_launch.json',R/'gmp_memory_ood_v191_launch.json']:
 if not p.name.endswith(('pipeline.log','delivery.log')):files[str(p.relative_to(R))]=sha(p)
for name in ['manifest.json','summary.json','result.json','audit.json']:
 p=B/name;files[str(p.relative_to(R))]=sha(p)
manifest=R/'gmp_memory_ood_v191_manifest.json';write(manifest,dict(status='bound',files=files,count=len(files),sources=sources(),supplement_sources={q.name:sha(q) for q in Path(__file__).resolve().parent.iterdir() if q.suffix in ('.py','.md','.sh')},checkpoint_external_asset=dict(path=str(CK),sha256=CK_SHA,repo_commit='c23bb40566a9eec2f90a570deb67ec71868e0820',HF_revision='29be4b9b364d293ba399394fe72745ac5af78c6a',binary_in_archive=False),ID_tasks=0,optimizer_updates=0))
archive=R/'gmp_memory_ood_v191.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz',compresslevel=3) as tar:
 for rel in files:tar.add(R/rel,arcname=rel,recursive=False)
 tar.add(manifest,arcname=manifest.name,recursive=False)
receipt=dict(status='completed',archive=str(archive),sha256=sha(archive),files=len(files)+1,manifest_sha256=sha(manifest),sources=sources());write(R/'gmp_memory_ood_v191_delivery.json',receipt);print(json.dumps(receipt),flush=True)
