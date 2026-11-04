from common import *
from policy import Official
setup();m=Official();m.new_task(0);print('OFFICIAL_GMP_LOADED',json.dumps(dict(policy_name=m.cfg.policy_name,checkpoint_sha256=CK_SHA,memory_enabled=not m.model.skip_memory,max_history=m.model.denoising_network.max_history_len,steps=m.model.noise_scheduler.inference_step_num,parameters=sum(p.numel() for p in m.model.parameters()),gpu_MB=torch.cuda.memory_allocated()/1e6)),flush=True)
