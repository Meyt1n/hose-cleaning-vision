import os,sys,time,math,json,traceback,collections
import numpy as np,torch
from runtime import *
from data import Data
from evaluation import evaluate
from state_utils import digest,rng_state,restore_rng,seed_all,save_checkpoint
from detection_model import build,load_mutable,mutable_state,training_mode
def setup(cfg):
 torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 torch.cuda.set_per_process_memory_fraction(cfg['memory_cap_bytes']/torch.cuda.get_device_properties(0).total_memory,0)
def probe(model,data):
 model.eval();x,_=data.load(data.hard['2026-09-09'][0])
 with torch.inference_mode():p=model([x.cuda()])[0]
 return digest({k:v.cpu() for k,v in p.items()})
def main():
 phase=sys.argv[1];cfg=ensure();guard();setup(cfg);data=Data(cfg)
 basecfg=read(cfg['detector_config']);model=build(basecfg).cuda();parent=torch.load(cfg['parent'],map_location='cpu');load_mutable(model,parent['model'])
 params=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.SGD(params,lr=cfg['lr'],momentum=.9,weight_decay=.0001)
 def lr_scale(s):return (.1+.9*min(1.,s/32.))*(.01+.99*.5*(1+math.cos(math.pi*min(s,cfg['steps'])/cfg['steps'])))
 scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_scale);seed_all(cfg['seed']);initial=digest(mutable_state(model));start=0;cost=collections.Counter()
 if phase=='short':
  if (RUN/'last.pth').exists():raise ValueError('R29_SHORT_ALREADY_EXISTS')
  atomic(RUN/'initialization.json',dict(initial_mutable_digest=initial,parent_sha256=sha(cfg['parent']),trainable_parameters=sum(p.numel() for p in params),frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),optimizer_state_policy='new SGD moments; initialize model only from completed R19',time_UTC=now()))
  atomic(RUN/'baseline_detection.json',evaluate(model,data))
 elif phase=='resume':
  state=torch.load(RUN/'last.pth',map_location='cpu')
  assert state['step']==cfg['short_steps'] and state['initial_digest']==initial
  load_mutable(model,state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
  for val in optimizer.state.values():
   for k,v in val.items():
    if torch.is_tensor(v):val[k]=v.cuda()
  restore_rng(state['rng']);cost.update(state['source_counts']);start=state['step']
  restored={'model':digest(mutable_state(model)),'optimizer':digest(optimizer.state_dict()),'scheduler':digest(scheduler.state_dict()),'rng':digest(rng_state())}
  assert restored==state['restore_digests'];assert data.batch(start)[2]==state['next_batch_ids'];assert probe(model,data)==state['output_probe_digest']
  restore_rng(state['rng'])
  atomic(RUN/'restore_verification.json',dict(step=start,new_process=True,complete_mutable_model_optimizer_scheduler_rng=True,output_probe_exact=True,next_batch_exact=True,digests=restored,time_UTC=now()))
 else:raise ValueError('R29_PHASE')
 end=cfg['short_steps'] if phase=='short' else cfg['steps'];beg=time.monotonic();grads={}
 for step in range(start,end):
  if step==start or step%40==0:guard()
  training_mode(model);images,targets,ids,sources=data.batch(step);images=[x.cuda() for x in images];targets=[{k:v.cuda() for k,v in t.items()} for t in targets]
  torch.manual_seed(cfg['seed']+step*29);torch.cuda.manual_seed_all(cfg['seed']+step*29);optimizer.zero_grad(set_to_none=True)
  losses=model(images,targets);loss=sum(losses.values());assert torch.isfinite(loss)
  loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,5.);assert torch.isfinite(norm)
  if step==start:
   for n,p in model.named_parameters():
    if p.requires_grad and p.grad is not None and any(t in n for t in ['fpn.inner_blocks.0.0.weight','rpn.head.cls_logits.weight','box_predictor.cls_score.weight']):grads[n]=float(p.grad.abs().mean())
   assert grads and all(math.isfinite(v) and v>0 for v in grads.values());assert all(p.grad is None for p in model.backbone.body.parameters())
  optimizer.step();scheduler.step();cost.update(sources)
  if (step+1)%20==0 or step==start or step+1==end:
   row=dict(step=step+1,phase=phase,loss=float(loss.detach()),losses={k:float(v.detach()) for k,v in losses.items()},grad_norm=float(norm),next_lr=optimizer.param_groups[0]['lr'],batch_ids=ids,source_counts=dict(cost),time_UTC=now())
   with (RUN/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  if step+1==end:
   guard();disk(save=True);modelstate=mutable_state(model);optstate=optimizer.state_dict();schstate=scheduler.state_dict();rng=rng_state();outprobe=probe(model,data);restore_rng(rng)
   state=dict(format='R29_COCO_frozen_body_complete_mutable_SGD_scheduler_RNG',model=modelstate,optimizer=optstate,scheduler=schstate,rng=rng,step=step+1,initial_digest=initial,source_counts=dict(cost),next_batch_ids=data.batch(step+1)[2],output_probe_digest=outprobe,parent_sha256=cfg['parent_sha256'],
    restore_digests={'model':digest(modelstate),'optimizer':digest(optstate),'scheduler':digest(schstate),'rng':digest(rng)})
   save_checkpoint(RUN/'last.pth',state)
   if phase=='short':os.link(RUN/'last.pth',RUN/'short16.pth')
 final=None
 if phase=='resume':
  final=evaluate(model,data);atomic(RUN/'final_detection.json',final)
  save_checkpoint(RUN/'inference.pth',dict(model=mutable_state(model),step=end,format='R29_mutable_detector_compatible_with_R19_loader',parent_sha256=cfg['coco_sha256'],semantic_scope='hand_detection_not_cleaning_or_connection'))
 atomic(RUN/(phase+'_completion.json'),dict(phase=phase,step=end,phase_optimizer_steps=end-start,unique_new_optimizer_steps=end,wall_seconds=time.monotonic()-beg,source_counts=dict(cost),gradient_evidence=grads,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),last_sha256=sha(RUN/'last.pth'),initial_mutable_digest=initial,inference_sha256=sha(RUN/'inference.pth') if phase=='resume' else None,metrics=None if final is None else final['metrics'],time_UTC=now()))
if __name__=='__main__':
 try:main()
 except Exception as e:
  atomic(ROOT/'receipts'/(INTENT+'_worker_failure.json'),dict(reason=str(e)[:180],type=type(e).__name__,frames=[dict(file=Path(f.filename).name,line=f.lineno) for f in traceback.extract_tb(e.__traceback__)],time_UTC=now()));raise
