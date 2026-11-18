import math,torch
from state_utils import digest,rng_state,set_rng,save
def configure(model,arm):
    for name,p in model.named_parameters():p.requires_grad_(name.startswith('head.') or name.startswith('backbone.stage4.'))
def make_optimizer(model,cfg,arm):
    groups=[];spec=[]
    for prefix,base in [('backbone.stage4.',cfg['backbone_lr']),('head.',cfg['head_lr'])]:
        for nodecay in [False,True]:
            pairs=[(n,p) for n,p in model.named_parameters() if p.requires_grad and n.startswith(prefix) and (p.ndim<=1 or n.endswith('.bias'))==nodecay]
            if not pairs:continue
            name=prefix+('norm_bias' if nodecay else 'decay');groups.append({'params':[p for n,p in pairs],'weight_decay':0. if nodecay else .01,'lr':base,'group_name':name});spec.append({'name':name,'base_lr':base,'weight_decay':0. if nodecay else .01,'parameter_names':[n for n,p in pairs],'numel':sum(p.numel() for n,p in pairs)})
    if len(groups)!=4 or sum(x['numel'] for x in spec)!=cfg['arms'][arm]['trainable_parameters']:raise ValueError('ACTUAL_TRAINABLE_GROUPS_MISMATCH')
    return torch.optim.AdamW(groups),spec
def schedule(opt,step,cfg,spec):
    rates=[]
    for group,row in zip(opt.param_groups,spec):
        base=row['base_lr'];end=base*.1;warm=cfg['warmup_steps'];rate=end+(base-end)*step/warm if step<warm else end+(base-end)*(1+math.cos(math.pi*(step-warm)/(cfg['steps']-warm)))/2;group['lr']=rate;rates.append(rate)
    return {'kind':'R18_actual_named_group_warmup_cosine','step':step,'steps':cfg['steps'],'next_lrs':rates,'group_names':[x['name'] for x in spec]}
def delta(model):
    state=model.state_dict();return {n:state[n] for n,p in model.named_parameters() if p.requires_grad}
def frozen_state(model):
    names={n for n,p in model.named_parameters() if p.requires_grad};return {n:v for n,v in model.state_dict().items() if n not in names}
def load_delta(model,part):
    expected={n for n,p in model.named_parameters() if p.requires_grad}
    if set(part)!=expected:raise ValueError('NAMED_TRAINABLE_DELTA_KEYS_CHANGED')
    state=model.state_dict();state.update(part);model.load_state_dict(state,strict=True)
