# -*- coding: utf-8 -*-
from pathlib import Path
import os,random,json,time,math,collections
import numpy as np
import torch
from detection_model import build,training_mode,mutable_state,load_mutable
from detection_data import Data
from evaluate_detector import evaluate
from state_utils import digest,rng_state,set_rng,save
from runtime import atomic,sha,now


def train(cfg,out,arm,phase,guard,installation):
    out=Path(out)
    if phase=='short_test':out.mkdir(exist_ok=False)
    elif not out.exists():raise ValueError('R19_RESUME_DIRECTORY_REQUIRED')
    model=build(cfg).cuda();data=Data(cfg)
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.SGD(params,lr=cfg['lr'],momentum=.9,weight_decay=.0001)
    def schedule(step):
        warm=.1+.9*min(1.,step/50.)
        return warm*(.01+.99*.5*(1+math.cos(math.pi*min(step,cfg['steps'])/cfg['steps'])))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    initial=digest(mutable_state(model));start=0;cost=collections.Counter();best=-1.;beststep=None
    if initial!=installation['initial_mutable_digest']:raise ValueError('R19_COMMON_INITIALIZATION_DIGEST_MISMATCH')
    atomic(out/'initialization.json',{'mutable_initial_digest':initial,'parent_SHA':cfg['coco_parent_sha256'],
        'trainable_parameters':sum(p.numel() for p in params),'frozen_parameters':sum(p.numel() for p in model.parameters() if not p.requires_grad),
        'optimizer':'SGD;one_group_FPN_RPN_boxhead','frozen_ResNet50_eval':True,'source_sha256':installation['source_sha256']}) if phase=='short_test' else None
    if phase=='resume':
        state=torch.load(out/'last.pth',map_location='cpu')
        if state['step']!=cfg['short_steps'] or state['initial_digest']!=initial or state['arm']!=arm:raise ValueError('R19_SHORT_PARENT_STATE_CHANGED')
        load_mutable(model,state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        for value in optimizer.state.values():
            for key,tensor in value.items():
                if torch.is_tensor(tensor):value[key]=tensor.cuda()
        set_rng(state['rng']);cost=collections.Counter(state['source_counts']);best=state['best_public_F1'];beststep=state['best_step'];start=state['step']
        restored={'model':digest(mutable_state(model)),'optimizer':digest(optimizer.state_dict()),'scheduler':digest(scheduler.state_dict()),'rng':digest(rng_state())}
        if restored!=state['restore_digests']:raise ValueError('R19_COMPLETE_RESTORE_DIGEST_MISMATCH')
        atomic(out/'restore_verification.json',{'step':start,'complete_mutable_model_optimizer_scheduler_RNG_restored':True,
            'digest_checks':restored,'next_batch_ids':data.batch(start,arm)[2],'expected_next_batch_ids':state['next_batch_ids'],'time_utc':now()})
        if data.batch(start,arm)[2]!=state['next_batch_ids']:raise ValueError('R19_SAMPLER_RESUME_MISMATCH')
    end=cfg['short_steps'] if phase=='short_test' else cfg['steps']
    started=time.monotonic();grad_evidence={};last_losses=None
    for step in range(start,end):
        if step==start or step%40==0:guard()
        training_mode(model);images,targets,ids,sources=data.batch(step,arm)
        images=[x.cuda() for x in images];targets=[{k:v.cuda() for k,v in t.items()} for t in targets]
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(cfg['seed']+step*29)
        torch.cuda.manual_seed_all(cfg['seed']+step*29)
        losses=model(images,targets);loss=sum(losses.values())
        if not torch.isfinite(loss):raise ValueError('R19_NONFINITE_LOSS')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,5.)
        if not torch.isfinite(norm):raise ValueError('R19_NONFINITE_GRADIENT')
        if step==start:
            for name,p in model.named_parameters():
                if p.requires_grad and p.grad is not None and any(t in name for t in ['fpn.inner_blocks.0.0.weight','rpn.head.cls_logits.weight','box_predictor.cls_score.weight']):grad_evidence[name]=float(p.grad.abs().mean())
            if not grad_evidence or not all(v>0 and math.isfinite(v) for v in grad_evidence.values()):raise ValueError('R19_GRADIENT_PROBE_FAILED')
            if any(p.grad is not None for p in model.backbone.body.parameters()):raise ValueError('R19_FROZEN_BACKBONE_GRADIENT')
        optimizer.step();scheduler.step();cost.update(sources)
        last_losses={k:float(v.detach()) for k,v in losses.items()}
        if (step+1)%20==0 or step==start or step+1==end:
            row={'step':step+1,'phase':phase,'losses':last_losses,'loss':float(loss.detach()),'grad_norm':float(norm),
                 'next_lr':optimizer.param_groups[0]['lr'],'batch_ids':ids,'source_counts':dict(cost),'time_UTC':now()}
            with(out/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        validation=None
        if phase=='resume' and ((step+1)%cfg['validate_every']==0 or step+1==end):
            validation=evaluate(model,data,cfg);validation['step']=step+1
            atomic(out/f'evaluation_step_{step+1:04d}.json',validation)
            with(out/'validation_curve.jsonl').open('a') as f:f.write(json.dumps({'step':step+1,'metrics':validation['metrics'],'time_UTC':now()})+'\n')
            current=validation['metrics']['development_hagrid_box']['F1']
            if current>best+1e-12:
                best=current;beststep=step+1;save(out/'best_delta.pth',{'model':mutable_state(model),'step':beststep,'parent_SHA':cfg['coco_parent_sha256'],'arm':arm})
                atomic(out/'best_ref.json',{'step':beststep,'public_author_F1':best,'delta_sha256':sha(out/'best_delta.pth'),'selection_scope':'public_author_boxes_only;primary_arm_comparison_fixed_final_step'})
        if step+1==end or (phase=='resume' and (step+1)%cfg['save_every']==0):
            guard();mstate=mutable_state(model);ostate=optimizer.state_dict();sstate=scheduler.state_dict();rng=rng_state()
            state={'format':'immutableCOCO_body_plus_complete_mutable_detector_optimizer_scheduler_RNG_v1',
                'model':mstate,'optimizer':ostate,'scheduler':sstate,'rng':rng,'step':step+1,'arm':arm,
                'parent_SHA':cfg['coco_parent_sha256'],'initial_digest':initial,'source_counts':dict(cost),
                'next_batch_ids':data.batch(step+1,arm)[2],'best_public_F1':best,'best_step':beststep,
                'restore_digests':{'model':digest(mstate),'optimizer':digest(ostate),'scheduler':digest(sstate),'rng':digest(rng)}}
            save(out/'last.pth',state)
    result={'arm':arm,'phase':phase,'step':end,'phase_optimizer_steps':end-start,
        'unique_new_optimizer_steps_including_short16':end,'phase_seconds':time.monotonic()-started,
        'source_counts_total':dict(cost),'gradient_evidence':grad_evidence,'last_losses':last_losses,
        'last_sha256':sha(out/'last.pth'),'best_public_F1':best,'best_step':beststep,
        'training_component':'direct_hand_ROI_detector;not21pointposefinetuning_or_action_count',
        'final_evaluation':str(out/f'evaluation_step_{end:04d}.json') if phase=='resume' else None,
        'independent_field_accuracy':False,'time_UTC':now()}
    atomic(out/('completion.json' if phase=='resume' else 'short_completion.json'),result)
    return result
