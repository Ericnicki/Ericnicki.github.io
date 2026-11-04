"""Original gated HistoryDenoisingPolicy and original ManipulationPolicyAgent decoder."""
import dill,hydra,robotmq
from omegaconf import OmegaConf
from common import *
from imitation_learning.common.dataclasses import construct_data_meta_dict
from imitation_learning.datasets.normalizer import FixedNormalizer
from imitation_learning.policies.history_denoising_policy import HistoryDenoisingPolicy
from env.modules.agents.manipulation_policy_agent import ManipulationPolicyAgent
from env.modules.agents.base_agent import BaseAgent
from robot_utils.image_utils import resize_with_padding
from imitation_learning.envs.policy_server import PolicyServer
class Official:
 def __init__(self):
  assert sha(CK)==CK_SHA;ck=torch.load(CK,map_location='cpu',pickle_module=dill,weights_only=False);self.cfg=OmegaConf.create(ck['cfg_str_unresolved']);mc=self.cfg.workspace.model
  # Exactly the official PolicyServer compatibility settings; no model arithmetic changes.
  if 'memory_gate' in mc and 'binary_gating' not in mc.denoising_network_partial:mc.denoising_network_partial.binary_gating=True
  if mc.memory_gate is not None and 'ckpt_path' in mc.memory_gate:mc.memory_gate.ckpt_path=''
  self.model=hydra.utils.instantiate(mc);self.model.load_state_dict(ck['model_state_dict'],strict=True);self.model.to(torch.device('cuda')).eval().requires_grad_(False);self.model.reset();assert isinstance(self.model,HistoryDenoisingPolicy) and not self.model.skip_memory and self.model.memory_gate is not None
  assert self.model.denoising_network.max_history_len==30 and self.model.noise_scheduler.inference_step_num==8 and self.model.action_decoder.token_num==16 and self.model.history_action_num_per_chunk==8
  self.norm=FixedNormalizer(construct_data_meta_dict(self.cfg.workspace.train_dataset.output_data_meta)).to('cuda');self.norm.load_state_dict(ck['normalizer_state_dict'],strict=False);self.norm.eval().requires_grad_(False);self.versions=tuple(x._version for x in self.model.parameters());self.last=None
  self.agent=object.__new__(ManipulationPolicyAgent)
  BaseAgent.__init__(self.agent,robot_num=1,agent_update_freq_hz=10,action_prediction_horizon=16,action_execution_horizon=8,obs_history_len=1,action_history_len=1,image_obs_frames_ids=[-1],proprio_obs_frames_ids=[-1],movement_constraints=['y'],default_actions=[[.1,-.73,.365,.707,-.707,0.,0.]],move_gripper=False,default_gripper_widths=[.08])
  self.agent.policy_obs_keys=['robot0_tcp_xyz_wxyz','robot0_gripper_width','third_person_camera'];self.agent.use_relative_pose=False;self.agent.smoothen_action_weight=.2;self.agent.current_episode_idx=0;self.agent.skip_policy_reset=True;self.agent.policy_client=self
 def new_task(self,task):
  self.model.reset();self.model.torch_rng.manual_seed(2187200+task);self.agent.current_episode_idx=task;self.task=task;assert not self.model.history_noisy_actions_dict and not self.model.history_img_features_dict
 def history(self):
  actions=self.model.history_noisy_actions_dict.get(self.task,[]);images=self.model.history_img_features_dict.get(self.task,[]);h=hashlib.sha256()
  for slot in actions:
   for x in slot:h.update(x.detach().cpu().numpy().tobytes())
  for x in images:h.update(x.detach().cpu().numpy().tobytes())
  return dict(action_chunks=len(actions),image_chunks=len(images),sha256=h.hexdigest())
 def prepare(self,raw):
  data={k:v.copy() for k,v in raw.items() if k!='episode_idx'}
  for k,v in data.items():
   if v.dtype==np.uint8:assert 'camera' in k or 'image' in k;v=v.astype(np.float32)/255.
   if 'camera' in k or 'image' in k:
    if v.shape[-1]==3:v=v.transpose(0,1,4,2,3)
    shape=self.model.global_cond_encoder.cond_meta[k].shape[-2:]
    if v.shape[-2:]!=shape:v=resize_with_padding(v,shape)
   data[k]=v
  return data
 @torch.no_grad()
 def predict_raw(self,raw):
  before=self.history();rng=self.model.torch_rng.get_state().cpu().clone();global_rng=dict(cpu=torch.get_rng_state().clone(),cuda=[x.clone() for x in torch.cuda.get_rng_state_all()]);data=self.prepare(raw);batch={k:torch.from_numpy(v).float().cuda() for k,v in data.items()};batch['episode_idx']=torch.tensor(raw['episode_idx'],device='cuda');batch=self.norm.normalize(batch);gate=float(self.model.memory_gate.get_gate_value(batch).item());normalized=self.model.predict_action(batch);physical=self.norm.unnormalize(normalized);result={k:v.detach().cpu().numpy() for k,v in physical.items()};after=self.history();assert after['action_chunks']==min(30,before['action_chunks']+1) and after['image_chunks']==min(30,before['image_chunks']+1);assert self.versions==tuple(x._version for x in self.model.parameters())
  self.last=dict(raw_input=copy.deepcopy(raw),rng_before=rng,global_rng_before=global_rng,history_before=before,history_after=after,gate=gate,history_attention_skipped=bool(self.model.denoising_network.skip_history_attn),normalized_action={k:v.cpu() for k,v in normalized.items()},physical_action=result)
  return result
 def request_with_data(self,topic,data,**kwargs):
  assert topic=='policy_inference';raw=robotmq.deserialize(data);return robotmq.serialize(self.predict_raw(raw))
 def choose(self,obs,previous):
  actions=self.agent.predict_actions([obs['robots_obs']],[obs['env_objs_obs']],[previous]);self.last['decoded_actions']=copy.deepcopy(actions);return actions,self.last
 def snapshot(self):return dict(actions=copy.deepcopy(self.model.history_noisy_actions_dict),images=copy.deepcopy(self.model.history_img_features_dict),recorded=copy.deepcopy(self.model.recorded_data_dicts),rng=self.model.torch_rng.get_state().clone(),global_cpu=torch.get_rng_state().clone(),global_cuda=[x.clone() for x in torch.cuda.get_rng_state_all()])
 def restore(self,s):
  self.model.history_noisy_actions_dict=copy.deepcopy(s['actions']);self.model.history_img_features_dict=copy.deepcopy(s['images']);self.model.recorded_data_dicts=copy.deepcopy(s['recorded']);self.model.torch_rng.set_state(s['rng']);torch.set_rng_state(s['global_cpu']);torch.cuda.set_rng_state_all(s['global_cuda'])
 @torch.no_grad()
 def server_reference(self,raw):
  server=object.__new__(PolicyServer);server.policy=self.model;server.normalizer=self.norm;server.robot_num=1;server.device=torch.device('cuda');server.debug_cnt=0
  return server._predict_action(self.prepare(raw),np.asarray(raw['episode_idx']))
