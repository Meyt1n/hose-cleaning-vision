# -*- coding: utf-8 -*-
from pathlib import Path
import json,hashlib
import numpy as np
import torch
from PIL import Image,ImageEnhance


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Data:
    def __init__(self,cfg):
        self.cfg=cfg
        self.rows=[json.loads(x) for x in (Path(cfg['data_root'])/'manifest.jsonl').read_text().splitlines()]
        self.groups={s:[r for r in self.rows if r['split']==s] for s in {r['split'] for r in self.rows}}
        self.roots={'current':Path(cfg['data_root']),'mc_parent':Path(cfg['mc_root']),'r17_parent':Path(cfg['r17_data_root'])}
        self.byid={r['id']:r for r in self.rows}

    def image(self,r):
        f=self.roots[r['image_location']]/r['image']
        if sha(f)!=r['image_sha256']:raise ValueError('FROZEN_SAMPLE_CHANGED')
        return Image.open(f).convert('RGB')

    def load(self,r,augmentation_seed=None):
        image=self.image(r);boxes=np.asarray(r['boxes_xyxy'],np.float32).copy()
        if augmentation_seed is not None:
            rng=np.random.RandomState(augmentation_seed)
            if rng.rand()<.5:
                image=image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                boxes[:,[0,2]]=image.width-boxes[:,[2,0]]
            for enhancer in [ImageEnhance.Brightness,ImageEnhance.Contrast,ImageEnhance.Color]:image=enhancer(image).enhance(float(rng.uniform(.85,1.15)))
        tensor=torch.from_numpy(np.array(image,copy=True)).permute(2,0,1).float()/255.
        target={'boxes':torch.from_numpy(boxes),'labels':torch.ones(len(boxes),dtype=torch.int64)}
        return tensor,target

    def batch(self,step,arm):
        rows=[];sources=[]
        for slot in range(2):
            rng=np.random.RandomState(self.cfg['seed']+step*1009+slot*37)
            public=['train_mc_box','train_freihand_box','train_hagrid_box'][int(rng.choice(3,p=[.5,.25,.25]))]
            source='train_field_old' if arm=='field_mix' and slot==1 and step%2==1 else public
            group=self.groups[source];rows.append(group[int(rng.randint(len(group)))]);sources.append(source)
        samples=[self.load(r,self.cfg['seed']+step*997+slot*13) for slot,r in enumerate(rows)]
        return [s[0] for s in samples],[s[1] for s in samples],[r['id'] for r in rows],sources

    def evaluation_rows(self):
        result=[]
        for s in ['development_hagrid_box','development_field_old','regression_field_opened']:
            result+=self.groups[s]
        for s in ['development_mc_box_new115','development_freihand_box']:
            result+=sorted(self.groups[s],key=lambda r:r['id'])[:32]
        return result
