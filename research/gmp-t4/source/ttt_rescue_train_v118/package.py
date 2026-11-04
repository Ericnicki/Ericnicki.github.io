from audit_results import *
import tarfile,hashlib
setup();assert json.loads((OUT/'audit.json').read_text())['status']=='passed';assert (OUT/'rescue_summary.png').is_file()
roots=[O,D,FITS,CONF,OUT,R/'ttt_rescue_v117',R/'ttt_rescue_train_v118']
files=sorted(p for root in roots for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.startswith('._') and p.name not in ['archive_manifest.json','archive_receipt.json'])
manifest={str(p.relative_to(R)):sha(p) for p in files};inherited={str(p.relative_to(R)):sha(p) for p in (R/'ttt_joint_1000_v115').glob('*.py')};inherited['endpoint_currentnode_runtime_v2/runtime.sh']=sha(R/'endpoint_currentnode_runtime_v2/runtime.sh')
write(OUT/'archive_manifest.json',dict(files=manifest,inherited_sources=inherited,parent_checkpoints=HASHES,source_sha256=sha(Path(__file__))))
archive=R/'ttt_rescue_v117_v121.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=3) as tf:
 for p in files+[OUT/'archive_manifest.json']:tf.add(p,arcname=str(p.relative_to(R)),recursive=False)
write(OUT/'archive_receipt.json',dict(path=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,members=len(files)+1));print((OUT/'archive_receipt.json').read_text())
