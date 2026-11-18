# -*- coding: utf-8 -*-
import os,sys,argparse,time,traceback
from pathlib import Path
from runtime import ROOT,SOURCE,RUN,INTENT,ensure_frozen,read,atomic,identity,check_fds,query_gpu,check_snapshot,disk_gate,now


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['base','occlusion'],required=True);p.add_argument('--phase',choices=['short_test','resume'],required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
    cfg,base=ensure_frozen()
    if cfg['arms'][a.arm]['physical_gpu']!=a.gpu or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(a.gpu):raise ValueError('POINT_GPU_ASSIGNMENT_CHANGED')
    owner=read(ROOT/'receipts'/f'{INTENT}_owner.json');checks=[]
    def guard():
        if identity(owner['pid'])!=owner['identity']:raise ValueError('POINT_OWNER_IDENTITY_CHANGED')
        check_fds(owner['pid'],a.gpu);disk_gate(atomic_save=False)
        try:
            s=query_gpu(a.gpu);check_snapshot(s,a.gpu,os.getpid());checks.append(s['free_MiB']);return
        except (ValueError,OSError):pass
        observations=[];stable=0
        for _ in range(12):
            atomic(ROOT/'receipts'/f'{INTENT}_{a.arm}_resource_pause.json',{'active':True,'time_utc':now(),'stable':stable,'observations':observations});time.sleep(30);check_fds(owner['pid'],a.gpu)
            try:s=query_gpu(a.gpu);check_snapshot(s,a.gpu,os.getpid());stable+=1;s['passed']=True
            except (ValueError,OSError):s={'passed':False};stable=0
            observations.append(s)
            if stable>=3:
                atomic(ROOT/'receipts'/f'{INTENT}_{a.arm}_resource_pause.json',{'active':False,'observations':observations});return
        raise ValueError('POINT_RESOURCE_NOT_RESTABILIZED')
    guard();import torch
    torch.set_num_threads(4)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('ONE_ALLOCATED_SERVER_GPU_REQUIRED')
    torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(cfg['framework_memory_cap_bytes']/torch.cuda.get_device_properties(0).total_memory,0);torch.cuda.reset_peak_memory_stats()
    from mix_train import train
    result=train(cfg,RUN/a.arm,a.arm,a.phase,guard,read(ROOT/'receipts'/f'{INTENT}_installation.json'))
    atomic(ROOT/'receipts'/f'{INTENT}_{a.arm}_{a.phase}_pass.json',{'arm':a.arm,'phase':a.phase,'physical_gpu':a.gpu,'identity':identity(os.getpid()),'result':result,
        'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'real_shared_locks_inherited':True,'resource_checks':len(checks),'time_utc':now()})


if __name__=='__main__':
    try:main()
    except Exception as e:
        atomic(ROOT/'receipts'/f'{INTENT}_worker_failure_{os.getpid()}.json',{'identity':identity(os.getpid()),'error_class':type(e).__name__,'reason':str(e) if str(e).isupper() and len(str(e))<160 else None,
            'frames':[{'file':Path(f.filename).name,'line':f.lineno} for f in traceback.extract_tb(e.__traceback__)]});raise
