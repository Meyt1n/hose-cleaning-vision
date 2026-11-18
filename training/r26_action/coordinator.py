import os,time,fcntl,subprocess,hashlib,traceback
from runtime import *

def status(stage,**extra):atomic(ROOT/'receipts'/(INTENT+'_service.json'),dict(stage=stage,identity=identity(os.getpid()),time_UTC=now(),**extra))
def launch(arm,phase,gpu,fds):
 args=[str(ROOT/'env/bin/python'),'-B','-u',str(SOURCE/'train_worker.py'),'--arm',arm,'--phase',phase]
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),HOSE_GLOBAL_DISPATCH_FD=str(fds[0]),HOSE_GPU_LEASE_FD=str(fds[gpu]),LOCAL_ACTION_OWNER_PID=str(os.getpid()),PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TMPDIR=str(ROOT/'tmp'));env.pop('PYTHONPATH',None)
 with (ROOT/'logs'/(INTENT+'_'+arm+'_'+phase+'.log')).open('xb') as f:p=subprocess.Popen(args,cwd=SOURCE,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,pass_fds=(fds[0],fds[gpu]),start_new_session=True)
 expected=hashlib.sha256(('\0'.join(args)+'\0').encode()).hexdigest();deadline=time.monotonic()+8
 while True:
  who=identity(p.pid)
  if who['command_sha256']==expected:break
  if p.poll() is not None or time.monotonic()>deadline:raise ValueError('R26_CHILD_IDENTITY')
  time.sleep(.02)
 atomic(ROOT/'receipts'/(INTENT+'_'+arm+'_'+phase+'_process.json'),dict(identity=who,owner=identity(os.getpid()),gpu=gpu,time_UTC=now()))
 return dict(process=p,identity=who,arm=arm,phase=phase)
def main():
 cfg,meta=ensure();fds=[];children=[]
 if RUN.exists():raise ValueError('R26_RUN_ALREADY_EXISTS')
 if shutil.disk_usage(ROOT).free<cfg['growth_margin_bytes']+cfg['runtime_storage_upper_bytes']:raise ValueError('R26_INITIAL_STORAGE_BUDGET')
 try:
  for path in [GLOBAL_LOCK,Path('/var/tmp/score66_gpu1.lock'),Path('/var/tmp/score66_gpu2.lock')]:
   if not path.is_file() or path.is_symlink():raise ValueError('R26_REAL_SHARED_LOCK_MISSING')
   fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW);fds.append(fd);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  observations=[]
  for i in range(3):
   if i:time.sleep(30)
   cards=[query_gpu(g) for g in [1,2]]
   for snap in cards:check(snap,snap['physical_index'])
   observations.append(dict(time_UTC=now(),cards=cards));atomic(ROOT/'receipts'/(INTENT+'_resource_observations.json'),dict(minimum_interval_seconds=30,observations=observations,real_global_and_perGPU_locks_held=True));status('RESOURCE_OBSERVATIONS',passed=i+1)
  RUN.mkdir()
  for phase in ['short','resume']:
   ensure()
   for gpu in [1,2]:check(query_gpu(gpu),gpu)
   children=[launch(a,phase,info['physical_gpu'],fds) for a,info in cfg['arms'].items()]
   while any(c['process'].poll() is None for c in children):
    status('TRAINING_'+phase.upper(),workers=[dict(identity=c['identity'],live=c['process'].poll() is None,arm=c['arm'],phase=c['phase']) for c in children]);time.sleep(3)
   if any(c['process'].returncode for c in children):raise ValueError('R26_WORKER_FAILED_RECONCILE_ONLY')
   results=[read(RUN/a/(phase+'_completion.json')) for a in cfg['arms']]
   if len({r['query_plan_phase_digest'] for r in results})!=1 or len({r['initial_model_digest'] for r in results})!=1:raise ValueError('R26_CONTROL_RANDOM_QUERY_OR_INITIALIZATION_MISMATCH')
   if any(r['step']!=(cfg['short_steps'] if phase=='short' else cfg['steps']) for r in results):raise ValueError('R26_STEP_BUDGET')
   if any(r['peak_cuda_allocated_bytes']>=cfg['memory_cap_bytes'] for r in results):raise ValueError('R26_MEMORY_GATE')
  completion=dict(status='COMPLETED',time_UTC=now(),actual_optimizer_steps=2*cfg['steps'],short16_included_once=True,both_full_restores_passed=True,branches={a:read(RUN/a/'resume_completion.json') for a in cfg['arms']},scope=meta['scope'],actual_connection_accuracy=None,per_hose_completion=None)
  atomic(RUN/'completion.json',completion);status('COMPLETED',actual_optimizer_steps=completion['actual_optimizer_steps'])
 finally:
  for child in children:
   while child['process'].poll() is None:time.sleep(3)
  for fd in reversed(fds):os.close(fd)
if __name__=='__main__':
 try:main()
 except Exception as e:
  atomic(ROOT/'receipts'/(INTENT+'_failure.json'),dict(type=type(e).__name__,reason=str(e)[:180],frames=[dict(file=Path(f.filename).name,line=f.lineno) for f in traceback.extract_tb(e.__traceback__)]));status('FAILED_RECONCILE_ONLY');raise
