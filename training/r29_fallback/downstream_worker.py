import sys,os,time,traceback
from fractions import Fraction
import numpy as np,torch,av
from runtime import *
@torch.inference_mode()
def main():
 cfg=ensure();guard();torch.set_num_threads(2);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 torch.cuda.set_per_process_memory_fraction(cfg['memory_cap_bytes']/torch.cuda.get_device_properties(0).total_memory,0)
 sys.path.insert(0,str(ROOT/'data/raw_pipeline_runtime_v1/site'));sys.path.insert(0,str(ROOT/'src/raw_pipeline_3min_r26_v1/project'))
 from common.glove_evidence import GloveHandEvidence
 from infer_service.engine.local_action_backend import LocalActionBackend
 profile=read(ROOT/'src/raw_pipeline_3min_r26_v1/profile_server.json');rawmanifest=Path(profile['local_action']['manifest']);mcfg=read(rawmanifest)
 provider=GloveHandEvidence(rawmanifest.parent/mcfg['glove_manifest']['file'],'cuda:0',score_threshold=.5,roi_normalized=[.375,.1,.375,1400/3000])
 delta=torch.load(RUN/'inference.pth',map_location='cpu',weights_only=True)['model'];base=provider.detector.state_dict();assert set(delta)=={k for k in base if not k.startswith('backbone.body.')};base.update(delta);provider.detector.load_state_dict(base,strict=True)
 heads={a:LocalActionBackend(cfg['action_parent'],cfg['action_sha256'],'cuda:0',.8,.2,.12) for a in ['baseline','adapted']}
 metadata=read(ROOT/'data/local_action_expanded_round25_v1/dataset.json');meta={r['id']:r for r in metadata['clips']};beg=time.monotonic();results=[];total=0
 assets=[]
 for cid in ['clip063','clip064','clip065','clip066','clip067','clip068']:
  assets.append(dict(id=cid,video=ROOT/'data/local_action_expanded_round25_v1/videos'/(cid+'.mp4'),features=ROOT/'runs/hand_tool_local_action_round25_parallel_v1/features'/(cid+'.npz'),source_sha256=meta[cid]['source_sha256'],mapping=None))
 assets.append(dict(id='SEP11_HOLDOUT',video=ROOT/'data/cross_domain_action_round28_v1/SEP11_HOLDOUT.mkv',features=ROOT/'runs/cross_domain_action_round28_seq_gpu2_v1/SEP11_HOLDOUT_features.npz',source_sha256='827c45622cd3fef5b4f5e7a1855b803905ffda33afa1e23f71370933ff1a679a',mapping=read(ROOT/'data/cross_domain_action_round28_v1/SEP11_HOLDOUT_source_map.json')['frames']))
 for asset in assets:
  cid=asset['id'];a=dict(np.load(asset['features']));ix={int(p):i for i,p in enumerate(a['source_pts'])};records=[];newh=[];newvalid=[]
  for h in heads.values():h.reset()
  with av.open(str(asset['video'])) as c:
   c.streams.video[0].codec_context.thread_count=2
   for fi,f in enumerate(c.decode(video=0)):
    mapped=None if asset['mapping'] is None else asset['mapping'][fi]
    p=int(f.pts) if mapped is None else mapped['source_pts'];tb=str(f.time_base) if mapped is None else mapped['time_base']
    if p not in ix:continue
    i=ix[p]
    if 'history_reset' in a and a['history_reset'][i]:
     for h in heads.values():h.reset()
    quality=bool(a.get('quality_valid',np.ones(len(ix),bool))[i]);evidence={'hands':[]}
    if quality:
     rgb=f.to_ndarray(format='rgb24');evidence=provider.predict(np.ascontiguousarray(rgb[:,:,::-1]),True,rgb)
     hands=evidence.pop('action_features');valid=evidence.pop('action_valid')
    else:hands=np.zeros((6,816),np.float32);valid=np.zeros(6,bool)
    details={}
    for arm,hh,vv in [('baseline',a['hands'][i],a['hand_valid'][i]),('adapted',hands,valid)]:
     d=heads[arm].process_features(a['context'][i],hh,vv,bool(a['context_valid'][i]),p,tb,1,score_enabled=False)
     details[arm]=dict(probability=float(d['probability']),accepted=bool(d['accepted']),action_code=d['action_code'],current_hand_count=int(vv.sum()),history_samples=d['history_samples'])
    records.append(dict(clip_id=cid,feature_index=i,native_index=fi,source_pts=p,time_base=tb,quality_valid=quality,models=details,new_hand_boxes=[h['box_xyxy'] for h in evidence['hands']],new_hand_scores=[h['detection_score'] for h in evidence['hands']]))
    newh.append(hands);newvalid.append(valid);total+=1
    if total%40==0:guard();atomic(RUN/'downstream_progress.json',dict(samples=total,clip=cid,feature_index=i,time_UTC=now(),wall_seconds=time.monotonic()-beg))
  assert len(records)==len(ix)
  dest=RUN/(cid+'_adapted_features.npz');np.savez_compressed(dest,hands=np.stack(newh),hand_valid=np.stack(newvalid),source_pts=a['source_pts'])
  atomic(RUN/(cid+'_predictions.json'),dict(rows=records,source_sha256=asset['source_sha256'],baseline_features_sha256=sha(asset['features']),adapted_hand_features_sha256=sha(dest),context_reused_unchanged=True,hand_detector_and_pose_fresh_on_raw=True,action_head_R26_unchanged=True))
  results.append(dict(id=cid,samples=len(records),adapted_hand_observed=sum(r['models']['adapted']['current_hand_count']>0 for r in records),baseline_hand_observed=sum(r['models']['baseline']['current_hand_count']>0 for r in records)))
 atomic(RUN/'downstream_completion.json',dict(clips=results,samples=total,wall_seconds=time.monotonic()-beg,new_optimizer_steps=0,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),physical_connection_accuracy=None,per_hose_accuracy=None,semantic_scope='macro operation; fixed context replay with fresh raw hand/pose observations',time_UTC=now()))
if __name__=='__main__':
 try:main()
 except Exception as e:
  atomic(ROOT/'receipts'/(INTENT+'_downstream_failure.json'),dict(reason=str(e)[:180],type=type(e).__name__,frames=[dict(file=Path(f.filename).name,line=f.lineno) for f in traceback.extract_tb(e.__traceback__)],time_UTC=now()));raise
