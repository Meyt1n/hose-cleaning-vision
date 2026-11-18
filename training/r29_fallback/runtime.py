from pathlib import Path
import os,sys,shutil,importlib.util
ROOT = Path(os.environ["HOSE_TRAIN_ROOT"]).resolve(strict=True)
INTENT = os.environ["HOSE_INTENT"]
SOURCE = Path(__file__).resolve().parent
DATA = Path(os.environ["HOSE_DATA_ROOT"]).resolve(strict=True)
RUN = ROOT / "runs" / INTENT
CONFIG = ROOT / "configs" / f"{INTENT}.json"
spec=importlib.util.spec_from_file_location('old_resource_helpers',Path(os.environ["HOSE_RESOURCE_HELPER"]).resolve(strict=True));old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
sha=old.sha;read=old.read;atomic=old.atomic;now=old.now;identity=old.identity;live=old.live;query_gpu=old.query_gpu;check_fds=old.check_fds;GLOBAL_LOCK=old.GLOBAL_LOCK
def ensure():
 cfg=read(CONFIG);ins=read(ROOT/'receipts'/(INTENT+'_installation.json'))
 if sha(CONFIG)!=ins['config_sha256']:raise ValueError('R29_CONFIG_CHANGED')
 for base,key in [(SOURCE,'source_members'),(DATA,'data_members')]:
  for row in ins[key]:
   if sha(base/row['name'])!=row['sha256']:raise ValueError('R29_FROZEN_MEMBER_CHANGED')
 for p,h in cfg['reference_files'].items():
  if sha(p)!=h:raise ValueError('R29_REFERENCE_CHANGED')
 return cfg
def disk(save=False):
 cfg=read(CONFIG);need=cfg['growth_margin_bytes']+(170*1024**2 if save else 0)
 if shutil.disk_usage(ROOT).free<need:raise ValueError('R29_DISK_RESERVE')
def check(snap,own_pid=None):
 cfg=read(CONFIG)
 if snap['physical_index']!=1 or snap['uuid_sha256']!=cfg['gpu_uuid_sha256'] or 'A40' not in snap['model']:raise ValueError('R29_GPU_IDENTITY')
 if snap['free_MiB']<8192:raise ValueError('R29_FREE_GPU_RESERVE')
 for r in snap['processes']:
  if r['pid']==own_pid:
   if r['memory_MiB'] is None or r['memory_MiB']>cfg['own_memory_limit_MiB']:raise ValueError('R29_OWN_MEMORY')
  else:
   b=cfg['co_residents'].get(str(r['pid']))
   if not b or any(r[k]!=b[k] for k in ['start_token','command_sha256']):raise ValueError('R29_NEW_CORESIDENT')
   if r['memory_MiB'] is None or r['memory_MiB']>b['memory_upper_MiB']:raise ValueError('R29_CORESIDENT_MEMORY')
def guard():
 check_fds(int(os.environ['LOCAL_ACTION_OWNER_PID']),1);check(query_gpu(1),os.getpid());disk()
