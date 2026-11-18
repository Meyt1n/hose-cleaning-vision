import torch,numpy as np,random,hashlib,os
from pathlib import Path
def digest(value):
 h=hashlib.sha256()
 def add(x):
  if isinstance(x,torch.Tensor):
   y=x.detach().cpu().contiguous();h.update(str(('tensor',y.dtype,tuple(y.shape))).encode());h.update(y.numpy().tobytes())
  elif isinstance(x,np.ndarray):h.update(str(('array',x.dtype,x.shape)).encode());h.update(x.tobytes())
  elif isinstance(x,dict):
   for k in sorted(x,key=lambda k:str(k)):add(k);add(x[k])
  elif isinstance(x,(tuple,list)):
   h.update(str(type(x)).encode())
   for v in x:add(v)
  else:h.update(repr((type(x).__name__,x)).encode())
 add(value);return h.hexdigest()
def rng_state():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
def restore_rng(d):random.setstate(d['python']);np.random.set_state(d['numpy']);torch.set_rng_state(d['torch']);torch.cuda.set_rng_state_all(d['cuda'])
def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def save_checkpoint(path,d):
 path=Path(path);tmp=path.with_name(path.name+'.atomic.'+str(os.getpid()))
 with tmp.open('xb') as f:torch.save(d,f);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def adamw(model,cfg):
 named=list(model.named_parameters());decay=[(n,p) for n,p in named if p.ndim>1];no=[(n,p) for n,p in named if p.ndim<=1]
 opt=torch.optim.AdamW([{'params':[p for n,p in decay],'names':[n for n,p in decay],'name':'decay','weight_decay':cfg['weight_decay']},{'params':[p for n,p in no],'names':[n for n,p in no],'name':'bias_norm','weight_decay':0.}],lr=cfg['lr'])
 return opt
