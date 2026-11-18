from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image,ImageEnhance,ImageFilter
from pose_geometry import affine
from occlusion import apply_occluder
class MixedData(torch.utils.data.Dataset):
    def __init__(self,cfg,training,arm='base'):
        self.cfg=cfg;self.root=Path(cfg['data_root']);self.training=training;self.arm=arm;self.perms={};rows=[json.loads(x) for x in (self.root/'manifest.jsonl').read_text().splitlines()];self.rows=[r for r in rows if r['split'].startswith('train_')==training]
        self.pools=[[r for r in self.rows if r['split']==s] for s in ['train_mc','train_freihand','train_field_AI']] if training else None
    def __len__(self):return len(self.rows)
    def __getitem__(self,key):
        if self.training:
            step,slot=key;pool=0 if slot<16 else 1 if slot<28 else 2;count=[16,12,4][pool];offset=[0,16,28][pool];index=step*count+slot-offset;cycle=index//len(self.pools[pool]);tag=(pool,cycle)
            if tag not in self.perms:self.perms[tag]=np.random.RandomState(self.cfg['seed']+pool*100000+cycle).permutation(len(self.pools[pool]))
            row=self.pools[pool][int(self.perms[tag][index%len(self.pools[pool])])]
        else:row=self.rows[key]
        roots={'mc_parent':Path(self.cfg['mc_root']),'r17_parent':Path(self.cfg['r17_data_root']),'current':self.root};im=Image.open(roots[row['image_location']]/row['image'].replace('\\','/')).convert('RGB');xy=np.array(row['keypoints_xy_256'],dtype='float64');weights=np.array(row['joint_weights'],dtype='float32');applied=False;cover=np.zeros(21,dtype='float32')
        if im.size!=(256,256):raise ValueError('R18_INPUT_SIZE_CHANGED')
        if self.training:
            rng=np.random.RandomState((self.cfg['seed']+step*2654435761+slot*2246822519)&0xffffffff);angle=30 if pool==2 else 90;im,xy=affine(im,xy,rng.uniform(-angle,angle),rng.uniform(.9,1.1),rng.uniform(-8,8,2),rng.rand()<.5)
            for enhancer in [ImageEnhance.Brightness,ImageEnhance.Contrast,ImageEnhance.Color]:im=enhancer(im).enhance(rng.uniform(.85,1.15))
            if rng.rand()<.1:im=im.filter(ImageFilter.GaussianBlur(rng.uniform(.2,.9)))
            occrng=np.random.RandomState((self.cfg['seed']^0x5f3759df^step*374761393^slot*668265263)&0xffffffff)
            if self.arm=='occlusion' and occrng.rand()<self.cfg['occlusion_probability']:
                im,_,info=apply_occluder(im,xy,weights,occrng);applied=info['applied'];cover[info['covered_known_indices']]=1.
        else:cover[row.get('synthetic_occlusion',{}).get('covered_known_indices',[])]=1.
        weights*=np.all((xy>=0)&(xy<256),axis=1)
        return {'image':torch.from_numpy(np.array(im,copy=True)).permute(2,0,1),'xy':torch.from_numpy(xy.astype('float32')),'weights':torch.from_numpy(weights),'id':row['id'],'split':row['split'],'condition':row.get('condition','public_or_original_field'),'normalizer':np.float32(row['normalizer_pixels']),'sigma':np.float32(row['sigma_pixels']),'synthetic_cover_mask':torch.from_numpy(cover),'synthetic_occlusion_applied':applied}
class Batches:
    def __init__(self,start,end):self.start,self.end=start,end
    def __iter__(self):
        for step in range(self.start,self.end):yield [(step,slot) for slot in range(32)]
    def __len__(self):return self.end-self.start
def worker_init(_):torch.set_num_threads(1)
def train_loader(data,start,end):return torch.utils.data.DataLoader(data,batch_sampler=Batches(start,end),num_workers=2,persistent_workers=True,pin_memory=True,multiprocessing_context='spawn',prefetch_factor=2,worker_init_fn=worker_init)
def normalize(image,device):
    x=image.to(device,non_blocking=True).float();return (x-x.new_tensor([123.675,116.28,103.53]).view(1,3,1,1))/x.new_tensor([58.395,57.12,57.375]).view(1,3,1,1)
def targets(xy,weights,sigma):
    grid=torch.arange(512,device=xy.device).float();centers=xy*2;sigma=sigma[:,None,None];labels=tuple(torch.exp(-.5*((grid[None,None,:]-centers[:,:,d,None])/sigma)**2) for d in [0,1]);balanced=weights*(21/weights.sum(1).clamp_min(1))[:,None];return labels,balanced
