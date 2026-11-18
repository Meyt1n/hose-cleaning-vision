"""Research prototype only: joint visual tokens and strictly causal temporal attention.

Inspired by the separation in DSTA (CVPR2024); this is an original 21-hand-joint,
past/current implementation, not the official body model or a trained release.
The 'provided' flag means an upstream observation exists, never visibility truth.
"""
import torch
from torch import nn
from torch.nn import functional as F

class FrozenHandJointTokens(nn.Module):
 def __init__(self,pose):
  super().__init__();self.pose=pose.eval().requires_grad_(False)
 @torch.inference_mode()
 def forward(self,normalized_RGB):
  feat=self.pose.backbone(normalized_RGB);head=self.pose.head
  token=head.gau(head.mlp(head.final_layer(feat[-1]).flatten(2)))
  sx,sy=head.cls_x(token),head.cls_y(token)
  xy=torch.stack([sx.argmax(-1),sy.argmax(-1)],-1).float()/2
  stats=[]
  for z in [sx,sy]:
   p=z.softmax(-1);top=z.topk(2,-1).values
   stats.extend([top[...,0],top[...,0]-top[...,1],-(p*p.clamp_min(1e-12).log()).sum(-1)/torch.log(z.new_tensor(512.))])
  return {'joint_tokens':token,'simcc_x':sx,'simcc_y':sy,'xy256':xy,'uncalibrated_distribution_stats':torch.stack(stats,-1)}

class JointFeatureCausal(nn.Module):
 def __init__(self,width=64,heads=4):
  super().__init__();self.width=width;self.heads=heads
  self.token=nn.Linear(256,width);self.pose=nn.Linear(2+6+1,width);self.time=nn.Linear(1,width);self.joint=nn.Parameter(torch.zeros(21,width));self.norm=nn.LayerNorm(width)
  self.qkv=nn.Linear(width,3*width);self.temporal=nn.Linear(width,width);self.spatial=nn.Linear(width,width);self.final=nn.LayerNorm(width)
  self.delta=nn.Linear(width,2);self.gate=nn.Linear(width,1);nn.init.zeros_(self.delta.weight);nn.init.zeros_(self.delta.bias);nn.init.zeros_(self.gate.weight);nn.init.constant_(self.gate.bias,-4.)
  adjacency=torch.eye(21)
  for finger in [[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]:
   for a,b in zip(finger,finger[1:]):adjacency[a,b]=adjacency[b,a]=1
  self.register_buffer('adjacency',adjacency)
 def forward_sequence(self,tokens,raw_station_xy,ROI_center_wh,provided,delta_seconds,distribution_stats):
  # tokens B,T,21,256; raw pixel XY B,T,21,2; ROI pixel center_x,y,width,height.
  B,T,J,_=tokens.shape;assert J==21
  side=ROI_center_wh[...,2:].amax(-1).clamp_min(20.);local=(raw_station_xy-ROI_center_wh[...,:2].unsqueeze(-2))/side[...,None,None]
  observed=provided.bool();m=observed.to(tokens.dtype);prefix=torch.ones(T,T,dtype=delta_seconds.dtype,device=delta_seconds.device).triu();time=delta_seconds@prefix
  x=self.token(tokens)+self.pose(torch.cat([local,distribution_stats,m[...,None]],-1))+self.time(time[...,None])[...,None,:]+self.joint;x=self.norm(x)*m[...,None]
  q,k,v=self.qkv(x).chunk(3,-1);d=self.width//self.heads
  def reshape(a):return a.reshape(B,T,J,self.heads,d).permute(0,2,3,1,4)
  q,k,v=map(reshape,[q,k,v]);score=q@k.transpose(-1,-2)/(d**.5)
  allowed=torch.ones(T,T,dtype=torch.bool,device=x.device).tril()[None,None,None,:,:]&observed.permute(0,2,1)[:,:,None,None,:]
  weight=score.masked_fill(~allowed,torch.finfo(score.dtype).min).softmax(-1)*allowed;weight=weight/weight.sum(-1,keepdim=True).clamp_min(torch.finfo(score.dtype).eps)
  temp=(weight@v).permute(0,3,1,2,4).reshape(B,T,J,self.width)
  # Same-frame finger-neighbour context; never pulls another hand's token.
  neighbour=torch.einsum('ij,btjc->btic',self.adjacency,x);denom=torch.einsum('ij,btj->bti',self.adjacency,m).clamp_min(1.)
  spatial=neighbour/denom[...,None];y=self.final(x+self.temporal(temp)+self.spatial(spatial))
  history_available=(m.transpose(1,2)@prefix).transpose(1,2)>=2;gate=torch.sigmoid(self.gate(y))*m[...,None]
  correction=.15*torch.tanh(self.delta(y))*side[...,None,None]*gate
  return {'xy_station':raw_station_xy+correction,'correction_gate':gate,'provided_observation':observed,'history_available':history_available,'future_frames_used':0,'uncertainty_or_visibility_calibrated':False}
 def forward(self,*args):
  result=self.forward_sequence(*args);return {k:(v[:,-1] if torch.is_tensor(v) else v) for k,v in result.items()}
