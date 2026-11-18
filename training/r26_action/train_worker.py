import os,sys,random,math,time,json,hashlib,argparse,traceback
import numpy as np,torch
from runtime import *
from boundary_model import BoundaryModel
from data import Data,action_forward,boundary_loss
from state_utils import digest,rng_state,restore_rng,save_checkpoint
from evaluation import evaluate

def schedule(opt,step,cfg):
 lr=cfg['min_lr']+(cfg['lr']-cfg['min_lr'])*(1+math.cos(math.pi*step/cfg['steps']))/2
 assert len(opt.param_groups)==1;opt.param_groups[0]['lr']=lr
 return dict(step=step,next_lr=lr,schedule='same_cosine_reset_AdamW_for_both_R26_arms')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',required=True);ap.add_argument('--phase',required=True);a=ap.parse_args();cfg,meta=ensure();gpu=cfg['arms'][a.arm]['physical_gpu'];guard(gpu)
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 torch.cuda.set_per_process_memory_fraction(cfg['memory_cap_bytes']/torch.cuda.get_device_properties(0).total_memory,0)
 random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
 data=Data();model=BoundaryModel().cuda();model.load_state_dict(torch.load(DATA/'initial_head.pth',map_location='cpu',weights_only=True));model.eval()
 opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);run=RUN/a.arm;run.mkdir(exist_ok=True);plan=read(DATA/'sample_plan.json');step=0;lrstate=schedule(opt,step,cfg);restore=None
 first=plan['batches'][0];probe=data.batch(first['action'][:4]);initial_digest=digest(model.state_dict())
 if a.phase=='short':
  if (run/'last.pth').exists():raise ValueError('R26_ALREADY_SHORT_TRAINED')
  with np.load(DATA/'cpu_probe.npz',allow_pickle=False) as archive:cpu={k:archive[k] for k in archive.files}
  with torch.inference_mode():pred=model(**{k:torch.from_numpy(v).cuda() for k,v in cpu.items() if k not in ['action_logits','boundary_logits']})
  diffs={k:float((pred[k].cpu()-torch.from_numpy(cpu[k])).abs().max()) for k in ['logits','boundary_logits'] if k in cpu}
  action_diff=float((pred['logits'].cpu()-torch.from_numpy(cpu['action_logits'])).abs().max());boundary_diff=float((pred['boundary_logits'].cpu()-torch.from_numpy(cpu['boundary_logits'])).abs().max())
  if max(action_diff,boundary_diff)>2e-4:raise ValueError('R26_CPU_GPU_INPUT_OUTPUT_MISMATCH')
  atomic(run/'preflight.json',dict(CPU_GPU_action_error=action_diff,CPU_GPU_boundary_error=boundary_diff,initial_model_digest=initial_digest,parameters=sum(p.numel() for p in model.parameters())))
  atomic(run/'initial_evaluation.json',evaluate(model,data))
 elif a.phase=='resume':
  state=torch.load(run/'last.pth',map_location='cpu',weights_only=False)
  if state['step']!=cfg['short_steps'] or state['arm']!=a.arm or state['config_sha256']!=sha(CONFIG) or state['sample_plan_sha256']!=sha(DATA/'sample_plan.json'):raise ValueError('R26_WRONG_RESUME')
  model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);step=state['step'];lrstate=schedule(opt,step,cfg)
  if digest(model.state_dict())!=state['model_digest'] or digest(opt.state_dict())!=state['optimizer_digest'] or lrstate!=state['schedule']:raise ValueError('R26_STATE_RESTORE_MISMATCH')
  with torch.inference_mode():out=action_forward(model,probe).cpu()
  if not torch.equal(out,state['output_probe'].cpu()):raise ValueError('R26_RESTORE_PREDICTION_MISMATCH')
  restore_rng(state['rng'])
  if digest(rng_state())!=state['rng_digest']:raise ValueError('R26_RNG_RESTORE_MISMATCH')
  restore=dict(full_model_AdamW_schedule_RNG_output_exact=True,step=step,checkpoint_sha256=sha(run/'last.pth'));atomic(run/'restore_verification.json',restore)
 else:raise ValueError('R26_UNKNOWN_PHASE')
 start=step;end=cfg['short_steps'] if a.phase=='short' else cfg['steps'];t0=time.monotonic();qd=hashlib.sha256();lam=cfg['arms'][a.arm]['boundary_loss_weight'];boundary_initial={k:v.detach().cpu().clone() for k,v in model.boundary.state_dict().items()}
 while step<end:
  if step==start or step%50==0:guard(gpu)
  row=plan['batches'][step];qd.update(json.dumps(row,separators=(',',':')).encode());b=data.batch(row['action']);model.train();opt.zero_grad(set_to_none=True)
  act=torch.nn.functional.cross_entropy(action_forward(model,b),b['target'],label_smoothing=.05);aux,auxmeta=boundary_loss(model,data,row['boundaries']);loss=act+lam*aux if lam else act
  if not torch.isfinite(loss):raise ValueError('R26_NONFINITE_LOSS')
  loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
  if not torch.isfinite(gn) or float(gn)<=0 or float(gn)>10000:raise ValueError('R26_BAD_GRADIENT')
  if lam and (model.boundary.weight.grad is None or not torch.isfinite(model.boundary.weight.grad).all()):raise ValueError('R26_MISSING_BOUNDARY_GRADIENT')
  if not lam and model.boundary.weight.grad is not None:raise ValueError('R26_CONTROL_BOUNDARY_MUST_REMAIN_UNTRAINED')
  opt.step();step+=1;lrstate=schedule(opt,step,cfg)
  if step%10==0 or step==end:atomic(run/'progress.json',dict(step=step,phase=a.phase,action_loss=float(act),boundary_loss=float(aux),total_loss=float(loss),gradient_norm=float(gn),next_lr=lrstate['next_lr'],time_UTC=now()))
 if not lam and any(not torch.equal(v.cpu(),boundary_initial[k]) for k,v in model.boundary.state_dict().items()):raise ValueError('R26_CONTROL_BOUNDARY_WEIGHTS_CHANGED')
 model.eval()
 with torch.inference_mode():out=action_forward(model,probe).cpu()
 rng=rng_state();state=dict(kind='operation_boundary_full_state_v1',step=step,arm=a.arm,model=model.state_dict(),optimizer=opt.state_dict(),schedule=lrstate,rng=rng,rng_digest=digest(rng),model_digest=digest(model.state_dict()),optimizer_digest=digest(opt.state_dict()),output_probe=out,config_sha256=sha(CONFIG),sample_plan_sha256=sha(DATA/'sample_plan.json'),initial_model_digest=initial_digest)
 disk(save=True);save_checkpoint(run/'last.pth',state)
 if a.phase=='short':shutil.copy2(run/'last.pth',run/'short16.pth');metrics=None
 else:
  metrics=evaluate(model,data,run/'evaluation');save_checkpoint(run/'inference.pth',dict(kind='hand_tool_operation_boundary_v1',channels=96,optimizer_steps=step,semantic_scope='visible_operation_and_operation_boundaries_not_actual_connection',model={k:v.cpu() for k,v in model.state_dict().items()},config_sha256=sha(CONFIG),parent_sha256=cfg['parent_inference_sha256'],arm=a.arm))
 atomic(run/(a.phase+'_completion.json'),dict(status='COMPLETED',step=step,phase=a.phase,new_optimizer_steps_this_process=step-start,short_steps_included_once=True,query_plan_phase_digest=qd.hexdigest(),initial_model_digest=initial_digest,last_sha256=sha(run/'last.pth'),restore=restore,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),wall_seconds=time.monotonic()-t0,parameters=sum(p.numel() for p in model.parameters()),metrics=metrics,time_UTC=now()))

if __name__=='__main__':
 try:main()
 except Exception as e:
  arm=sys.argv[sys.argv.index('--arm')+1];atomic(ROOT/'receipts'/(INTENT+'_'+arm+'_worker_failure.json'),dict(type=type(e).__name__,reason=str(e)[:180],frames=[dict(file=Path(f.filename).name,line=f.lineno) for f in traceback.extract_tb(e.__traceback__)]));raise
