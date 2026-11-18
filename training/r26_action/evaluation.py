import json,numpy as np,torch
from runtime import atomic
from data import action_forward
from continuity import ContinuousEvidenceTimer

def metric(y,p):
 y=np.asarray(y);p=np.asarray(p)>=.5;pos=y==1;tp=int((p&pos).sum());fp=int((p&~pos).sum());fn=int((~p&pos).sum());tn=int((~p&~pos).sum())
 return dict(TP=tp,FP=fp,FN=fn,TN=tn,N=len(y),F1=2*tp/max(1,2*tp+fp+fn),precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),accuracy=(tp+tn)/max(1,len(y)))

def boundary_score(pred,bags):
 edges=[]
 for i,p in enumerate(pred):
  for j,b in enumerate(bags):
   if p['kind']!=b['kind']:continue
   lo,hi=b['boundary_source_bracket'];d=max(lo-p['time'],0,p['time']-hi)
   if d<=1:edges.append((d,i,j))
 usedp=set();usedb=set();pairs=[]
 # True windows are separated by many seconds, so1s matching neighborhoods do not overlap.
 for d,i,j in sorted(edges):
  if i in usedp or j in usedb:continue
  usedp.add(i);usedb.add(j);pairs.append(dict(prediction=pred[i],truth_interval=bags[j]['id'],distance_outside_uncertainty_interval=d))
 tp=len(pairs);fp=len(pred)-tp;fn=len(bags)-tp
 return dict(TP=tp,FP=fp,FN=fn,F1=2*tp/max(1,2*tp+fp+fn),precision=tp/max(1,len(pred)),recall=tp/max(1,len(bags)),true_boundaries=len(bags),predicted_boundaries=len(pred),tolerance_seconds=1,pairs=pairs,scope='AI_operation_boundary_intervals_NOT_physical_connection')

@torch.inference_mode()
def evaluate(model,data,output=None):
 model.eval();n=len(data.labels['target']);probs=[];bound=[]
 for start in range(0,n,32):
  anchors=np.arange(start,min(n,start+32));indices=anchors[:,None]+np.arange(-31,1)[None];pad=indices<0;indices=indices.clip(min=0)
  ind=torch.from_numpy(indices).to(data.device);pm=torch.from_numpy(pad).to(data.device);b={k:v[ind].clone() for k,v in data.field.items()}
  for x in b.values():x[pm]=0
  out=model(**b);probs.extend(out['logits'][:,-1].softmax(-1)[:,1].cpu().tolist());bound.extend(out['boundary_logits'][:,-1].sigmoid().cpu().tolist())
 probs=np.array(probs);bound=np.array(bound);times=data.labels['source_elapsed_seconds'];target=data.labels['target'];split=data.labels['split'];support=data.field['hand_valid'].any(-1).cpu().numpy();accepted=support&((probs<=.2)|(probs>=.8));results={}
 for s,name in [(0,'field_train'),(1,'field_blocked_development')]:
  mask=(split==s)&(target>=0);sel=(split==s);bags=[b for b in data.bags.values() if b['split']==s];pred=[];raw=[];last=[-1e9,-1e9];prior=[False,False];previous=None
  for i in np.where(sel)[0]:
   for channel,kind in [(0,'start'),(1,'end')]:
    flag=bound[i,channel]>=.5
    if flag and not prior[channel] and times[i]-last[channel]>=.5:pred.append(dict(kind=kind,time=float(times[i])));last[channel]=times[i]
    prior[channel]=flag
   positive=probs[i]>=.5
   if previous is not None and positive!=previous:raw.append(dict(kind='start' if positive else 'end',time=float(times[i])))
   previous=positive
  results[name]=dict(action=metric(target[mask],probs[mask]),accepted_action=metric(target[mask&accepted],probs[mask&accepted]),accepted_label_coverage=float(accepted[mask].mean()),accepted_continuous_coverage=float(accepted[sel].mean()),supervised_action_points=int(mask.sum()),continuous_samples=int(sel.sum()),learned_boundary=boundary_score(pred,bags),action_transition_boundary=boundary_score(raw,bags))
 oldq=[['old',key,i] for key,item in data.old.items() if item['meta']['split']!='train' for i in range(len(item['target']))];op=[];oy=[]
 for beg in range(0,len(oldq),32):
  b=data.batch(oldq[beg:beg+32]);op.extend(action_forward(model,b).softmax(-1)[:,1].cpu().tolist());oy.extend(b['target'].cpu().tolist())
 results['old6_opened_regression']=metric(oy,op)
 # Audit-only reconnect labels never contributed to gradients.
 recon=(times>=191.03)&(times<196.03);results['human_reconnect_audit']=dict(raw_positive_fraction=float((probs[recon]>=.5).mean()),minimum_probability=float(probs[recon].min()),learned_boundary_crossings_not_scored_as_physical_truth=True,actual_connection_accuracy=None)
 timer=ContinuousEvidenceTimer();rows=[];events=[]
 for i,t in enumerate(times):
  if not 150<=t<450:continue
  e=timer.update(bool(support[i] and probs[i]>=.8),int(data.labels['source_pts'][i]),str(data.labels['time_base'][i]),observation_state='observed' if support[i] else 'unknown')
  if e:events.append(e)
  rows.append(dict(source_pts=int(data.labels['source_pts'][i]),time_base=str(data.labels['time_base'][i]),source_elapsed_seconds=float(t),operation_probability=float(probs[i]),start_boundary_probability=float(bound[i,0]),end_boundary_probability=float(bound[i,1]),current_hand_observed=bool(support[i]),accepted=bool(accepted[i]),operation_label=int(target[i]),split=int(split[i]),research_action_duration_count=timer.count,actual_connection_verified=False,physical_hose_id=None))
 results['whole5minute_research_candidates']=dict(candidates=len(events),events=events,not_physical_contact_or_unique_hose_counts=True)
 if output:
  output.mkdir(exist_ok=True);atomic(output/'summary.json',results)
  with (output/'predictions.jsonl').open('w') as f:
   for r in rows:f.write(json.dumps(r,allow_nan=False)+'\n')
 return results
