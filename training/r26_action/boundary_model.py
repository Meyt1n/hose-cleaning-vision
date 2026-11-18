import torch
from torch import nn
from local_action_model import HandToolTemporalHead

class BoundaryModel(HandToolTemporalHead):
    def __init__(self):
        super().__init__()
        self.boundary=nn.Linear(96,2)
        nn.init.normal_(self.boundary.weight,mean=0,std=.01)
        nn.init.constant_(self.boundary.bias,-3.)
    def forward(self,context,hands,hand_valid,context_valid,delta_pts):
        out=super().forward(context,hands,hand_valid,context_valid,delta_pts)
        out['boundary_logits']=self.boundary(out.pop('shared'))
        return out
