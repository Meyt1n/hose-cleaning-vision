# -*- coding: utf-8 -*-
import os,sys,time,fcntl,subprocess,shutil,hashlib,traceback
from pathlib import Path
from runtime import ROOT,SOURCE,RUN,CONFIG,INTENT,GLOBAL_LOCK,read,atomic,identity,ensure_frozen,query_gpu,check_snapshot,now


def status(stage,**extra):atomic(ROOT/'receipts'/f'{INTENT}_service.json',{'stage':stage,'intent_id':INTENT,'identity':identity(os.getpid()),'time_utc':now(),**extra})


def launch(arm,phase,gpu,global_fd,card_fd):
    args=[str(ROOT/'env/bin/python'),'-B','-u',str(SOURCE/'worker.py'),'--arm',arm,'--phase',phase,'--gpu',str(gpu)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),HOSE_GLOBAL_DISPATCH_FD=str(global_fd),HOSE_GPU_LEASE_FD=str(card_fd),PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TMPDIR=str(ROOT/'tmp'));env.pop('PYTHONPATH',None)
    with (ROOT/'logs'/f'{INTENT}_{arm}_{phase}.log').open('xb') as f:p=subprocess.Popen(args,cwd=SOURCE,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,pass_fds=(global_fd,card_fd),start_new_session=True)
    expected=hashlib.sha256(('\0'.join(args)+'\0').encode()).hexdigest();deadline=time.monotonic()+8
    while True:
        who=identity(p.pid)
        if who['command_sha256']==expected:break
        if p.poll() is not None or time.monotonic()>deadline:raise ValueError('POINT_WORKER_EXEC_IDENTITY_NOT_CONFIRMED')
        time.sleep(.02)
    atomic(ROOT/'receipts'/f'{INTENT}_{arm}_{phase}_process.json',{'identity':who,'parent_identity':identity(os.getpid()),'physical_gpu':gpu,'argv':args,'time_utc':now()})
    return {'arm':arm,'phase':phase,'gpu':gpu,'process':p,'identity':who}


def publish(children):
    states={}
    for item in children:
        folder=RUN/item['arm'];p=folder/'progress.jsonl';last=None
        if p.is_file():
            lines=p.read_text().splitlines()
            if lines:last=__import__('json').loads(lines[-1])
        states[item['arm']]={'physical_gpu':item['gpu'],'phase':item['phase'],'process_live':item['process'].poll() is None,'identity':item['identity'],'last_progress':last}
    status('PARALLEL_GLOVE_POSE_TRAINING',workers=states)


def passed(item):
    if item['process'].returncode:raise ValueError('POINT_BRANCH_FAILED_RECONCILE_ONLY')
    return read(ROOT/'receipts'/f'{INTENT}_{item["arm"]}_{item["phase"]}_pass.json')


def main():
    cfg,_=ensure_frozen();intent=read(ROOT/'receipts'/f'{INTENT}_intent.json')
    if RUN.exists() or intent['physical_gpus']!=[1,2]:raise ValueError('POINT_EXISTING_RUN_OR_ASSIGNMENT_RECONCILE_ONLY')
    if shutil.disk_usage(ROOT).free<cfg['expected_added_storage_upper_bound_bytes']+cfg['required_growth_margin_bytes']:raise ValueError('POINT_TOTAL_DISK_BUDGET_FAILED')
    fds=[];children=[]
    try:
        for path in [GLOBAL_LOCK,Path('/var/tmp/score66_gpu1.lock'),Path('/var/tmp/score66_gpu2.lock')]:
            if not path.is_file() or path.is_symlink():raise ValueError('EXISTING_POINT_SHARED_LOCK_REQUIRED')
            fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW);fds.append(fd);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        owner={'pid':os.getpid(),'identity':identity(os.getpid()),'physical_gpus':[1,2],'lock_inodes':[os.fstat(fd).st_ino for fd in fds],'time_utc':now()};atomic(ROOT/'receipts'/f'{INTENT}_owner.json',owner)
        stable=[];observations=[]
        for _ in range(12):
            ob={'time_utc':now(),'cards':[query_gpu(1),query_gpu(2)]}
            try:
                for s in ob['cards']:check_snapshot(s,s['physical_index'])
                ob['passed']=True;stable.append(ob)
            except ValueError as e:ob['passed']=False;ob['reason']=str(e);stable=[]
            observations.append(ob);atomic(ROOT/'receipts'/f'{INTENT}_resource_observations.json',{'interval_seconds':30,'observations':observations});status('RESOURCE_OBSERVATIONS',consecutive_passes=len(stable),latest=ob,gpu_training_started=False)
            if len(stable)>=3:break
            time.sleep(30)
        if len(stable)<3:raise ValueError('POINT_THREE_STABLE_RESOURCE_OBSERVATIONS_REQUIRED')
        atomic(ROOT/'receipts'/f'{INTENT}_resource_pass.json',{'observations':stable[-3:],'interval_seconds':30,'real_global_and_per_card_locks_held':True});RUN.mkdir()
        children=[launch(arm,'short_test',cfg['arms'][arm]['physical_gpu'],fds[0],fds[cfg['arms'][arm]['physical_gpu']]) for arm in ['base','occlusion']]
        while any(c['process'].poll() is None for c in children):publish(children);time.sleep(5)
        for c in children:
            if passed(c)['result']['step']!=16:raise ValueError('FIXED_FIT_SHORT_BUDGET_CHANGED')
        ensure_frozen()
        for gpu in [1,2]:check_snapshot(query_gpu(gpu),gpu)
        children=[launch(arm,'resume',cfg['arms'][arm]['physical_gpu'],fds[0],fds[cfg['arms'][arm]['physical_gpu']]) for arm in ['base','occlusion']]
        while any(c['process'].poll() is None for c in children):publish(children);time.sleep(5)
        results=[passed(c) for c in children]
        if any(r['result']['phase_optimizer_steps']!=cfg['steps']-cfg['short_steps'] or r['result']['step']!=cfg['steps'] for r in results):raise ValueError('FIXED_FIT_FINAL_BUDGET_CHANGED')
        completion={'status':'COMPLETED','intent_id':INTENT,'physical_gpus':[1,2],'phase_optimizer_steps':2*cfg['steps'],'steps_per_arm':cfg['steps'],'short16_included_each':True,'actual_hand21point_pose_task':True,
            'branches':results,'final_gpu_snapshots':[query_gpu(1),query_gpu(2)],'time_utc':now(),'independent_site_acceptance':False,'three_second_completion_measured':False}
        atomic(RUN/'completion.json',completion);atomic(ROOT/'receipts'/f'{INTENT}_completion.json',completion);status('COMPLETED',summary=completion)
    finally:
        for c in children:
            while c['process'].poll() is None:time.sleep(5)
        for fd in reversed(fds):os.close(fd)


if __name__=='__main__':
    try:main()
    except Exception as e:
        failure={'error_class':type(e).__name__,'reason':str(e) if str(e).isupper() and len(str(e))<160 else None,'frames':[{'file':Path(f.filename).name,'line':f.lineno} for f in traceback.extract_tb(e.__traceback__)]}
        atomic(ROOT/'receipts'/f'{INTENT}_failure.json',failure);status('FAILED_RECONCILE_ONLY',failure=failure);raise
