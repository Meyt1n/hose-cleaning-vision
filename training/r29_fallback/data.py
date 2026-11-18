from pathlib import Path
import sys,json,numpy as np,torch
from PIL import Image,ImageEnhance
from runtime import ROOT,DATA,sha,read
class Data:
 def __init__(self,cfg):
  self.cfg=cfg;self.rows=read(DATA/'visible_hand_labels.json')['rows']
  self.old=[json.loads(s) for s in (ROOT/'data/glove_hand_detection_round19_v1/manifest.jsonl').read_text().splitlines()]
  self.groups={s:[r for r in self.old if r['split']==s] for s in {r['split'] for r in self.old}}
  self.hard={d:[r for r in self.rows if r['split']=='train_hard' and r['source_date']==d] for d in ['2026-09-09','2026-09-11']}
  self.roots={'current':ROOT/'data/glove_hand_detection_round19_v1','mc_parent':ROOT/'data/mc_glove_hand5_round16_v1','r17_parent':ROOT/'data/glove_mixed_round17_v1'}
 def image(self,r):
  p=DATA/r['image'] if 'image_location' not in r else self.roots[r['image_location']]/r['image']
  if sha(p)!=r['image_sha256']:raise ValueError('R29_SAMPLE_SHA_CHANGED')
  return Image.open(p).convert('RGB')
 def load(self,r,seed=None):
  im=self.image(r);b=np.array(r['boxes_xyxy'],np.float32).reshape(-1,4)
  if seed is not None:
   rng=np.random.RandomState(seed)
   if rng.rand()<.5:im=im.transpose(Image.Transpose.FLIP_LEFT_RIGHT);b[:,[0,2]]=im.width-b[:,[2,0]]
   for enhancer in [ImageEnhance.Brightness,ImageEnhance.Contrast,ImageEnhance.Color]:im=enhancer(im).enhance(float(rng.uniform(.85,1.15)))
  return torch.from_numpy(np.array(im)).permute(2,0,1).float()/255.,{'boxes':torch.from_numpy(b),'labels':torch.ones(len(b),dtype=torch.int64)}
 def batch(self,step):
  rng=np.random.RandomState(self.cfg['seed']+step*1009);date=['2026-09-09','2026-09-11'][step%2];pool=self.hard[date];r0=pool[int(rng.randint(len(pool)))]
  group='train_field_old' if step%4==0 else ['train_mc_box','train_freihand_box','train_hagrid_box'][int(rng.choice(3,p=[.5,.25,.25]))]
  pool=self.groups[group];r1=pool[int(rng.randint(len(pool)))];rows=[r0,r1];values=[self.load(r,self.cfg['seed']+step*997+i*13) for i,r in enumerate(rows)]
  return [v[0] for v in values],[v[1] for v in values],[r['id'] for r in rows],['hard_'+date,group]
 def evaluation_rows(self):
  return [r for r in self.rows if r['split']!='excluded_unknown']+self.groups['development_hagrid_box']+self.groups['development_field_old']
