"""Procedural input occlusion. Known source labels remain labels; unknown never gains weight."""
import numpy as np
from PIL import Image,ImageDraw
def apply_occluder(image,xy,weights,rng):
    eligible=np.flatnonzero((np.asarray(weights)>0)&np.all((np.asarray(xy)>=0)&(np.asarray(xy)<256),axis=1))
    if not len(eligible):return image.copy(),np.zeros((256,256),dtype=np.uint8),{'applied':False,'covered_known_indices':[]}
    joint=int(eligible[int(rng.randint(len(eligible)))]);center=np.asarray(xy[joint],dtype='float64')+rng.uniform(-5,5,2);angle=rng.uniform(-np.pi,np.pi)
    width=rng.uniform(12,28);length=rng.uniform(70,150);a=np.array([np.cos(angle),np.sin(angle)])*length/2;b=np.array([-np.sin(angle),np.cos(angle)])*width/2
    polygon=[tuple(center+u*a+v*b) for u,v in [(-1,-1),(1,-1),(1,1),(-1,1)]];mask=Image.new('L',(256,256),0);ImageDraw.Draw(mask).polygon(polygon,fill=255);mask=np.array(mask,dtype=np.uint8);color=np.array([(40,45,50),(155,160,166),(35,65,95)][int(rng.randint(3))],dtype='int16');noise=rng.randint(-8,9,(256,256,1));rgb=np.clip(color[None,None,:]+noise,0,255).astype('uint8');out=np.array(image,copy=True);out[mask>0]=rgb[mask>0]
    rounded=np.rint(xy).astype(int);inside=np.all((rounded>=0)&(rounded<256),axis=1);covered=[int(k) for k in eligible if inside[k] and mask[rounded[k,1],rounded[k,0]]>0]
    return Image.fromarray(out),mask,{'applied':True,'generator':'opaque_tool_strip_v1','chosen_known_joint':joint,'covered_known_indices':covered,'mask_area_fraction':float((mask>0).mean()),'polygon_xy256':[list(p) for p in polygon],'synthetic_not_real_occlusion_GT':True}
