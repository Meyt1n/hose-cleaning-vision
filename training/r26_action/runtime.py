from pathlib import Path
import os,json,hashlib,shutil,importlib.util
ROOT = Path(os.environ["HOSE_TRAIN_ROOT"]).resolve(strict=True)
INTENT = os.environ["HOSE_INTENT"]
SOURCE = Path(__file__).resolve().parent
DATA = Path(os.environ["HOSE_DATA_ROOT"]).resolve(strict=True)
RUN = ROOT / "runs" / INTENT
CONFIG = ROOT / "configs" / f"{INTENT}.json"
spec=importlib.util.spec_from_file_location('existing_real_resource_helpers',Path(os.environ["HOSE_RESOURCE_HELPER"]).resolve(strict=True))
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
sha=old.sha;read=old.read;atomic=old.atomic;identity=old.identity;live=old.live;now=old.now;query_gpu=old.query_gpu;check_fds=old.check_fds;GLOBAL_LOCK=old.GLOBAL_LOCK
def ensure():
 cfg=read(CONFIG);install=read(ROOT/'receipts'/(INTENT+'_installation.json'))
 if sha(CONFIG)!=install['config_sha256']:raise ValueError('LOCAL_ACTION_CONFIG_CHANGED')
 for x in install['source_members']:
  if sha(SOURCE/x['name'])!=x['sha256']:raise ValueError('LOCAL_ACTION_SOURCE_CHANGED')
 for x in cfg['data_files']:
  if sha(DATA/x['name'])!=x['sha256']:raise ValueError('LOCAL_ACTION_DATA_CHANGED')
 for p,h in cfg['reference_files'].items():
  if sha(p)!=h:raise ValueError('LOCAL_ACTION_REFERENCE_CHANGED')
 return cfg,read(DATA/'dataset.json')
def disk(save=False):
 cfg=read(CONFIG);need=cfg['growth_margin_bytes']+(cfg['atomic_save_budget_bytes'] if save else 0)
 if shutil.disk_usage(ROOT).free<need:raise ValueError('LOCAL_ACTION_DISK_BUDGET_FAILED')
def check(snap,gpu,own_pid=None):
 cfg=read(CONFIG)
 if snap['physical_index']!=gpu or snap['uuid_sha256']!=cfg['gpu_uuid_sha256'][str(gpu)] or 'A40' not in snap['model']:raise ValueError('LOCAL_ACTION_GPU_IDENTITY')
 if snap['free_MiB']<4096:raise ValueError('LOCAL_ACTION_GPU_FREE_RESERVE')
 for item in snap['processes']:
  if item['pid']==own_pid:
   if item['memory_MiB'] is None or item['memory_MiB']>cfg['own_memory_limit_MiB']:raise ValueError('LOCAL_ACTION_OWN_MEMORY_LIMIT')
  else:
   baseline=cfg['co_residents'][str(gpu)].get(str(item['pid']))
   if not baseline or any(item[k]!=baseline[k] for k in ['start_token','command_sha256']):raise ValueError('LOCAL_ACTION_UNKNOWN_GPU_PROCESS')
   if item['memory_MiB'] is None or item['memory_MiB']>baseline['memory_upper_MiB']:raise ValueError('LOCAL_ACTION_CORESIDENT_CHANGED')
 return True
def guard(gpu):
 check_fds(int(os.environ['LOCAL_ACTION_OWNER_PID']),gpu);check(query_gpu(gpu),gpu,os.getpid());disk()
def tensor_digest(state):
 h=hashlib.sha256()
 for k in sorted(state):
  a=state[k].detach().cpu().contiguous().numpy();h.update(str(k).encode());h.update(str((a.dtype,a.shape)).encode());h.update(a.tobytes())
 return h.hexdigest()
