import json,hashlib,numpy as np,torch
from runtime import DATA,read,sha

class Data:
 def __init__(self,device='cuda'):
  self.device=device;self.meta=read(DATA/'dataset.json');self.old={}
  for row in self.meta['old_feature_references']:
   if sha(row['file'])!=row['sha256']:raise ValueError('OLD_FROZEN_FEATURE_SHA')
   with np.load(row['file'],allow_pickle=False) as archive:a={k:archive[k] for k in archive.files}
   self.old[row['id']]={'meta':row,'tensors':{k:torch.from_numpy(a[k]).to(device) for k in ['context','hands','hand_valid','context_valid','delta_pts']},'target':a['target']}
  with np.load(DATA/'context.npz',allow_pickle=False) as archive:c={k:archive[k] for k in archive.files}
  with np.load(self.meta['hand_cache'],allow_pickle=False) as archive:h={k:archive[k] for k in archive.files}
  with np.load(DATA/'labels.npz',allow_pickle=False) as archive:labels={k:archive[k] for k in archive.files}
  assert np.array_equal(c['source_pts'],h['source_pts']) and np.array_equal(c['time_base'],h['time_base']) and np.array_equal(c['source_pts'],labels['source_pts'])
  c.update(hands=h['hands'],hand_valid=h['hand_valid'])
  self.field={k:torch.from_numpy(c[k]).to(device) for k in ['context','hands','hand_valid','context_valid','delta_pts']};self.labels=labels
  self.bags={r['id']:r for r in read(DATA/'fine_boundary_bags.json')}
 def batch(self,queries):
  rows={k:[] for k in self.field};labels=[]
  for q in queries:
   if q[0]=='field':t=self.field;i=q[1];label=int(self.labels['target'][i])
   else:t=self.old[q[1]]['tensors'];i=q[2];label=int(self.old[q[1]]['target'][i])
   assert label in (0,1)
   ind=torch.arange(i-31,i+1,device=self.device);pad=ind<0;ind=ind.clamp_min(0)
   for k in rows:x=t[k][ind].clone();x[pad]=0;rows[k].append(x)
   labels.append(label)
  return {**{k:torch.stack(x) for k,x in rows.items()},'target':torch.tensor(labels,device=self.device)}
 def boundary_batch(self,names):
  bags=[self.bags[n] for n in names];length=max(b['last_index']-b['input_start_index']+1 for b in bags)
  batch={k:[] for k in self.field};negative=[];positive=[]
  for b in bags:
   lo=b['input_start_index'];hi=b['last_index']+1;n=hi-lo;pad=length-n
   for k in batch:
    x=self.field[k][lo:hi];shape=(pad,)+tuple(x.shape[1:]);batch[k].append(torch.cat([torch.zeros(shape,dtype=x.dtype,device=self.device),x]))
   nm=torch.zeros((length,2),dtype=torch.bool,device=self.device);pm=torch.zeros_like(nm)
   nm[pad+b['first_index']-lo:]=True
   for i in b['positive_indices']:nm[pad+i-lo,b['channel']]=False;pm[pad+i-lo,b['channel']]=True
   negative.append(nm);positive.append(pm)
  return {k:torch.stack(x) for k,x in batch.items()},torch.stack(negative),torch.stack(positive)

def action_forward(model,b):return model(**{k:v for k,v in b.items() if k!='target'})['logits'][:,-1]

def boundary_loss(model,data,names):
 b,nm,pm=data.boundary_batch(names);logits=model(**b)['boundary_logits'];pos=[]
 for j in range(len(names)):
  for channel in range(2):
   if pm[j,:,channel].any():pos.append(torch.nn.functional.softplus(-logits[j,pm[j,:,channel],channel].max()))
 losspos=torch.stack(pos).mean();lossneg=torch.nn.functional.softplus(logits[nm]).mean()
 return losspos+lossneg,{'positive_bags':len(pos),'negative_points':int(nm.sum())}
