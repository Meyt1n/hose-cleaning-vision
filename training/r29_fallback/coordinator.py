import os,time,fcntl,subprocess,hashlib,traceback
from runtime import *
def status(stage,**extra):atomic(ROOT/'receipts'/(INTENT+'_service.json'),dict(stage=stage,identity=identity(os.getpid()),time_UTC=now(),**extra))
def launch(kind,fds):
 args=[str(ROOT/'env/bin/python'),'-B','-u',str(SOURCE/('downstream_worker.py' if kind=='downstream' else 'train_worker.py'))]
 if kind!='downstream':args.append(kind)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',HOSE_GLOBAL_DISPATCH_FD=str(fds[0]),HOSE_GPU_LEASE_FD=str(fds[1]),LOCAL_ACTION_OWNER_PID=str(os.getpid()),PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TMPDIR=str(ROOT/'tmp'));env.pop('PYTHONPATH',None)
 with (ROOT/'logs'/(INTENT+'_'+kind+'.log')).open('xb') as log:
  child=subprocess.Popen(args,cwd=SOURCE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=tuple(fds))
 expected=hashlib.sha256((chr(0).join(args)+chr(0)).encode()).hexdigest();deadline=time.monotonic()+8
 while True:
  who=identity(child.pid)
  if who['command_sha256']==expected:break
  if child.poll() is not None or time.monotonic()>deadline:raise ValueError('R29_CHILD_IDENTITY')
  time.sleep(.02)
 atomic(ROOT/'receipts'/(INTENT+'_'+kind+'_process.json'),dict(identity=who,owner=identity(os.getpid()),gpu=1,time_UTC=now()));return child,who
def main():
 cfg=ensure();fds=[];child=None
 if RUN.exists():raise ValueError('R29_RUN_ALREADY_EXISTS')
 if shutil.disk_usage(ROOT).free<cfg['runtime_storage_upper_bytes']+cfg['growth_margin_bytes']:raise ValueError('R29_INITIAL_STORAGE')
 try:
  for path in [GLOBAL_LOCK,Path('/var/tmp/score66_gpu1.lock')]:
   if not path.is_file() or path.is_symlink():raise ValueError('R29_REAL_LOCK_MISSING')
   fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW);fds.append(fd);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  obs=[]
  for i in range(3):
   if i:time.sleep(30)
   snap=query_gpu(1);check(snap);obs.append(snap);atomic(ROOT/'receipts'/(INTENT+'_observations.json'),dict(observations=obs,minimum_interval_seconds=30,real_global_and_gpu1_locks_held=True));status('RESOURCE_OBSERVATIONS',passed=i+1)
  RUN.mkdir()
  for phase in ['short','resume','downstream']:
   ensure();check(query_gpu(1));child,who=launch(phase,fds)
   while child.poll() is None:status(phase.upper(),worker=who);time.sleep(3)
   if child.returncode:raise ValueError('R29_WORKER_FAILED_RECONCILE_'+phase)
   if phase in ['short','resume']:
    done=read(RUN/(phase+'_completion.json'));assert done['step']==(cfg['short_steps'] if phase=='short' else cfg['steps']) and done['peak_cuda_allocated_bytes']<cfg['memory_cap_bytes']
  atomic(RUN/'completion.json',dict(status='COMPLETED',actual_optimizer_steps=cfg['steps'],short16_counted_once=True,full_restore=read(RUN/'restore_verification.json'),train=read(RUN/'resume_completion.json'),downstream=read(RUN/'downstream_completion.json'),actual_connection_accuracy=None,per_hose_accuracy=None,time_UTC=now()));status('COMPLETED',actual_optimizer_steps=cfg['steps'])
 finally:
  if child is not None:
   while child.poll() is None:time.sleep(3)
  for fd in reversed(fds):os.close(fd)
if __name__=='__main__':
 try:main()
 except Exception as e:
  atomic(ROOT/'receipts'/(INTENT+'_failure.json'),dict(type=type(e).__name__,reason=str(e)[:200],frames=[dict(file=Path(f.filename).name,line=f.lineno) for f in traceback.extract_tb(e.__traceback__)],time_UTC=now()));status('FAILED_RECONCILE_ONLY');raise
