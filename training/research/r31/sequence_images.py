"""Known whole-image transforms, not native finger articulation or new field GT."""
import numpy as np
from PIL import Image,ImageEnhance,ImageFilter,ImageDraw
def sequence(row):
 im=Image.open(row['image']).convert('RGB');assert im.size==(256,256)
 xy=np.asarray(row['xy'],np.float64);w=np.asarray(row['weights'],np.float32)
 r=np.random.RandomState(row['seed']);T=16;pressure=row['split']=='public_pressure'
 # Keep an identity/near-correct stratum; deterministic same images for both arms.
 mode=int(r.randint(4));angle=r.uniform(-35,35) if row['source']!='field_AI' else r.uniform(-12,12)
 amplitude=r.uniform(5,18)*(1.7 if pressure else 1.);velocity=r.uniform(-1,1,2)*amplitude
 rotation=r.uniform(-8,8);images=[];truth=[];weights=[];tags=[]
 for t in range(T):
  q=t/(T-1);motion=(float(t>=8) if mode==2 else q)
  a=np.deg2rad(angle+rotation*motion);M=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
  shift=velocity*motion;offset=np.array([128,128])+shift-M@np.array([128,128])
  if mode==0:M=np.eye(2);offset=np.zeros(2)
  inv=np.linalg.inv(M);back=-inv@offset
  image=im.transform((256,256),Image.Transform.AFFINE,tuple(np.c_[inv,back].reshape(-1)),resample=Image.Resampling.BILINEAR)
  target=xy@M.T+offset;known=w*np.all((target>=0)&(target<256),1)
  occluded=mode!=0 and (6<=t<=11 or pressure and t>=7)
  if occluded:
   j=int(r.choice(np.flatnonzero(w>0)));cx,cy=target[j];wid=26 if not pressure else 42
   ImageDraw.Draw(image).rectangle((cx-wid/2,cy-46,cx+wid/2,cy+46),fill=(50,65,80))
  if mode!=0:
   image=ImageEnhance.Brightness(image).enhance(1+.12*np.sin(t*.4))
   if t%5==2:image=image.filter(ImageFilter.GaussianBlur(1.0 if not pressure else 1.8))
  images.append(image);truth.append(target.astype('float32'));weights.append(known);tags.append(dict(fast=mode==2 and 8<=t<=10,occluded=occluded,identity=mode==0))
 return images,np.stack(truth),np.stack(weights),tags
