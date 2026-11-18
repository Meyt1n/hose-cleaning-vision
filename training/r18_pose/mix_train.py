from pathlib import Path
import json,time,random,math
import numpy as np
import torch
from PIL import Image
from native_hand5_model import Hand5Pose,decode_simcc
from mix_data import MixedData,train_loader,normalize,targets
from mix_state import configure,make_optimizer,schedule,delta,frozen_state,load_delta,digest,rng_state,set_rng,save
from pose_loss import KLDiscretLoss
from runtime import ROOT,RUN,atomic,sha,now
def training_mode(net):
    net.train()
    for m in net.modules():
        if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):m.eval()
def evaluate(net,data,device):
    net.eval();rows=[]
    with torch.no_grad():
        for b in torch.utils.data.DataLoader(data,batch_size=16,num_workers=0):
            xy,score=decode_simcc(*net(normalize(b['image'],device)));prediction=xy.cpu().numpy();truth=b['xy'].numpy();weights=b['weights'].numpy()
            for i,name in enumerate(b['id']):
                err=np.linalg.norm(prediction[i]-truth[i],axis=1)/float(b['normalizer'][i]);known=weights[i]>0;rows.append({'id':name,'split':b['split'][i],'known_joints':int(known.sum()),'hits10':int(((err<=.1)&known).sum()),'hits05':int(((err<=.05)&known).sum()),'errors_normalized':err.tolist(),'weights':weights[i].tolist(),'prediction_xy256':prediction[i].tolist(),'target_xy256':truth[i].tolist(),'normalizer_pixels':float(b['normalizer'][i]),'condition':b['condition'][i],'synthetic_cover_mask':b['synthetic_cover_mask'][i].tolist()})
    slices={}
    for name in sorted({r['split'] for r in rows}):
        part=[r for r in rows if r['split']==name];n=sum(r['known_joints'] for r in part);errors=[e for r in part for e,w in zip(r['errors_normalized'],r['weights']) if w>0]
        slices[name]={'images':len(part),'joints':n,'PCK0p10_hits':sum(r['hits10'] for r in part),'PCK0p10':sum(r['hits10'] for r in part)/n,'PCK0p05':sum(r['hits05'] for r in part)/n,'NME':float(np.mean(errors))}
    reg=json.loads((Path(data.cfg['data_root'])/'unlabeled_regression_manifest.json').read_text());pics=torch.stack([torch.from_numpy(np.array(Image.open(Path(data.cfg['data_root'])/r['image']).convert('RGB'))).permute(2,0,1) for r in reg])
    with torch.no_grad():pts,_=decode_simcc(*net(normalize(pics,device)))
    p=pts.cpu().numpy();edge=np.any((p<=.5)|(p>=255.),axis=2);regdiag={'images':len(reg),'joints':int(edge.size),'edge_points':int(edge.sum()),'per_case':[{'id':r['id'],'edge_points':int(edge[i].sum()),'prediction_xy256':p[i].tolist(),'joint_GT':None} for i,r in enumerate(reg)],'scope':'reusedunlabeleddevelopmentqualitydiagnostic;notaccuracy'}
    stress=[r for r in rows if r['split']=='development_synthetic_stress'];stress_groups={}
    for group in ['synthetically_covered','not_synthetically_covered']:
        errors=[e for r in stress for e,w,c in zip(r['errors_normalized'],r['weights'],r['synthetic_cover_mask']) if w>0 and (c>0)==(group=='synthetically_covered')]
        stress_groups[group]={'projected_joints':len(errors),'PCK0p10':float(np.mean(np.array(errors)<=.1)) if errors else None,'NME':float(np.mean(errors)) if errors else None}
    hard=[r for r in rows if r['split']=='development_new_field_AI' and r['condition']=='rotated_or_tool_grip'];he=[e for r in hard for e,w in zip(r['errors_normalized'],r['weights']) if w>0]
    hard_metrics={'images':len(hard),'joints':len(he),'PCK0p10':float(np.mean(np.array(he)<=.1)),'NME':float(np.mean(he))}
    primary=float(np.mean([slices[k]['NME'] for k in ['development_field_AI','development_new_field_AI']]))
    return {'primary_macro_field_NME':primary,'new_interaction_subset':hard_metrics,'synthetic_stress_breakdown':stress_groups,'by_slice':slices,'MC_macro_PCK0p10':float(np.mean([slices[k]['PCK0p10'] for k in ['development_mc_old113','development_mc_new115']])),'unlabeled_regression':regdiag,'AI_field_not_human_verified':True},rows
def eligibility(m,base):
    return {'old_AI_field_retention':m['by_slice']['development_field_AI']['PCK0p10']>=base['by_slice']['development_field_AI']['PCK0p10']-1/15-1e-9,'MC_retention':m['MC_macro_PCK0p10']>=base['MC_macro_PCK0p10']-.01,'FreiHAND_retention':m['by_slice']['development_freihand_retention']['PCK0p10']>=base['by_slice']['development_freihand_retention']['PCK0p10']-.01,'edge_regression_guard':m['unlabeled_regression']['edge_points']<=base['unlabeled_regression']['edge_points']+2}
def train(cfg,run,arm,phase,guard,install):
    began=time.monotonic();run=Path(run);run.mkdir(exist_ok=True)
    if (phase=='short_test' and (run/'last.pth').exists()) or (phase=='resume' and (run/'completion.json').exists()):raise ValueError('R18_EXISTING_RUN_RECONCILE_ONLY')
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    model=Hand5Pose().cuda();model.load_state_dict(torch.load(cfg['parent_checkpoint'],map_location='cpu',weights_only=True),strict=True);configure(model,arm);load_delta(model,torch.load(cfg['initial_delta_checkpoint'],map_location='cpu',weights_only=True)['delta']);frozen_digest=digest(frozen_state(model));optimizer,groups=make_optimizer(model,cfg,arm);step=0;lrstate=schedule(optimizer,step,cfg,groups)
    training=MixedData(cfg,True,arm);dev=MixedData(cfg,False,arm)
    if len(training)!=cfg['train_images'] or len(dev)!=cfg['development_images']:raise ValueError('R18_DATA_COUNTS_CHANGED')
    probe=normalize(torch.stack([dev[i]['image'] for i in [0,3,100,len(dev)-1]]),'cuda');model.eval();restore=None
    if phase=='short_test':
        baseline,bpred=evaluate(model,dev,'cuda');atomic(run/'baseline_metrics.json',baseline);atomic(run/'baseline_predictions.json',bpred)
        best={'step':0,'path':cfg['initial_delta_checkpoint'],'sha256':cfg['initial_delta_sha256'],'is_original_pretrained':False,'is_initial_parent':True,'metric_NME':baseline['primary_macro_field_NME'],'metrics':baseline};atomic(run/'best_ref.json',best)
    else:
        full=torch.load(run/'last.pth',map_location='cpu',weights_only=False)
        if full['kind']!='R18_full_delta_state_v1' or full['arm']!=arm or full['step']!=16:raise ValueError('R18_SHORT_STATE_MISMATCH')
        for k,v in [('source_sha256',install['source_sha256']),('config_sha256',install['config_sha256']),('data_manifest_sha256',cfg['data_manifest_sha256']),('parent_sha256',cfg['parent_checkpoint_sha256']),('initial_delta_sha256',cfg['initial_delta_sha256'])]:
            if full[k]!=v:raise ValueError('R18_PROVENANCE_CHANGED')
        if digest(full['delta'])!=full['delta_digest'] or digest(full['optimizer'])!=full['optimizer_digest'] or groups!=full['optimizer_groups']:raise ValueError('R18_FULL_MUTABLE_STATE_CHANGED')
        load_delta(model,full['delta']);optimizer.load_state_dict(full['optimizer']);step=full['step'];lrstate=schedule(optimizer,step,cfg,groups);best=full['best'];baseline=full['baseline']
        if digest(model.state_dict())!=full['full_model_digest'] or digest(optimizer.state_dict())!=full['optimizer_digest'] or lrstate!=full['scheduler']:raise ValueError('R18_MODEL_OPTIMIZER_LR_RESTORE_FAILED')
        with torch.no_grad():out=tuple(v.cpu() for v in model(probe))
        if any(not torch.equal(a,b) for a,b in zip(out,full['output_probe'])):raise ValueError('R18_OUTPUT_PROBE_RESTORE_NOT_EXACT')
        restore={'new_process_restore_exact':True,'restored_step':step,'full_model_digest':full['full_model_digest'],'optimizer_digest':full['optimizer_digest'],'actual_groups':len(groups),'scheduler':lrstate,'checkpoint_sha256':sha(run/'last.pth'),'RNG_digest':full['rng_digest']}
    start=step;end=16 if phase=='short_test' else cfg['steps'];loader=train_loader(training,start,end);iterator=iter(loader)
    if phase=='resume':
        set_rng(full['rng'])
        if digest(rng_state())!=full['rng_digest']:raise ValueError('R18_RNG_RESTORE_FAILED')
        atomic(run/'restore_verification.json',restore);del full
    lossfn=KLDiscretLoss(beta=10.,label_softmax=True,use_target_weight=True);gradient_evidence=None;latest=None;last_loss=None
    while step<end:
        if step==start or step%100==0:guard()
        b=next(iterator);training_mode(model);x=normalize(b['image'],'cuda');xy=b['xy'].cuda(non_blocking=True);w=b['weights'].cuda(non_blocking=True);sigma=b['sigma'].cuda(non_blocking=True);optimizer.zero_grad(set_to_none=True);pred=model(x);target,w=targets(xy,w,sigma);loss=lossfn(pred,target,w)
        if not torch.isfinite(loss):raise ValueError('R18_NONFINITE_LOSS')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.)
        if not torch.isfinite(norm) or norm<=0:raise ValueError('R18_INVALID_GRADIENT')
        if gradient_evidence is None:
            params=dict(model.named_parameters());names=['head.final_layer.weight','head.cls_x.weight']+['backbone.stage4.0.conv.weight'];gradient_evidence={n:float(params[n].grad.abs().mean()) for n in names}
            if any(v<=0 for v in gradient_evidence.values()) or any(p.grad is None for p in model.parameters() if p.requires_grad) or any(p.grad is not None for p in model.parameters() if not p.requires_grad):raise ValueError('R18_TRAINABLE_OR_FROZEN_GRADIENT_POLICY_FAILED')
        optimizer.step();step+=1;lrstate=schedule(optimizer,step,cfg,groups);last_loss=float(loss)
        if step%20==0 or step==end:
            with (run/'progress.jsonl').open('a') as f:f.write(json.dumps({'step':step,'MC_replay_epoch_fraction':step/313,'loss':last_loss,'grad_norm':float(norm),'next_lrs':lrstate['next_lrs'],'phase':phase,'time_UTC':now()},allow_nan=False)+'\n')
        do_eval=phase=='resume' and (step%cfg['evaluation_interval_steps']==0 or step==end)
        if do_eval:
            latest,rows=evaluate(model,dev,'cuda');checks=eligibility(latest,baseline);ob={'step':step,'MC_replay_epoch':step/313,'metrics':latest,'guards':checks,'time_UTC':now()}
            with (run/'validation_curve.jsonl').open('a') as f:f.write(json.dumps(ob,allow_nan=False)+'\n')
            value=latest['primary_macro_field_NME']
            if all(checks.values()) and value<best['metric_NME']-1e-6:
                save(run/'best.pth',{'kind':'R18_inference_delta_v1','delta':{k:v.cpu() for k,v in delta(model).items()},'arm':arm,'step':step,'parent_sha256':cfg['parent_checkpoint_sha256'],'metrics':latest})
                best={'step':step,'path':str(run/'best.pth'),'sha256':sha(run/'best.pth'),'is_original_pretrained':False,'is_initial_parent':False,'metric_NME':value,'metrics':latest};atomic(run/'best_ref.json',best);atomic(run/'best_predictions.json',rows)
            if step==end:atomic(run/'final_predictions.json',rows)
        if step==end or (phase=='resume' and step%cfg['checkpoint_interval_steps']==0):
            guard();model.eval()
            if digest(frozen_state(model))!=frozen_digest:raise ValueError('R18_FROZEN_FEATURE_OR_BN_STATE_CHANGED')
            with torch.no_grad():probeout=tuple(v.cpu() for v in model(probe))
            rng=rng_state();part=delta(model);state={'kind':'R18_full_delta_state_v1','arm':arm,'step':step,'delta':part,'delta_digest':digest(part),'full_model_digest':digest(model.state_dict()),'frozen_state_digest':frozen_digest,'optimizer':optimizer.state_dict(),'optimizer_digest':digest(optimizer.state_dict()),'optimizer_groups':groups,'scheduler':lrstate,'rng':rng,'rng_digest':digest(rng),'output_probe':probeout,'baseline':baseline,'best':best,
                'source_sha256':install['source_sha256'],'config_sha256':install['config_sha256'],'data_manifest_sha256':cfg['data_manifest_sha256'],'parent_sha256':cfg['parent_checkpoint_sha256'],'initial_delta_sha256':cfg['initial_delta_sha256'],'sampler_cursor':{'next_global_step':step,'stateless_mixture_and_augmentation':True,'seed':cfg['seed']}}
            save(run/'last.pth',state);del state,part
    del iterator,loader
    result={'arm':arm,'step':step,'phase_optimizer_steps':step-start,'unique_optimizer_steps_including_short16':step,'MC_replay_epochs':step/313,'trainable_parameters':cfg['arms'][arm]['trainable_parameters'],'gradient_evidence':gradient_evidence,'optimizer_groups':groups,'restore':restore,'best':best,'final_metrics':latest,'checkpoint_sha256':sha(run/'last.pth'),'elapsed_this_process_seconds':time.monotonic()-began,'last_loss':last_loss,'frozen_state_unchanged':True,'parent_required_for_delta':cfg['parent_checkpoint_sha256'],'time_UTC':now(),'not_independent_human_field_acceptance':True}
    if phase=='resume':atomic(run/'completion.json',result)
    return result
