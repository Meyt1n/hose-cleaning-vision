from pathlib import Path
import os, json, hashlib, subprocess, datetime, zipfile, shutil

ROOT = Path(os.environ["HOSE_TRAIN_ROOT"]).resolve(strict=True)
INTENT = os.environ["HOSE_INTENT"]
SOURCE = Path(__file__).resolve().parent
RUN = ROOT / "runs" / INTENT
CONFIG = ROOT / "configs" / f"{INTENT}.json"
CONFIG_SHA = os.environ["HOSE_CONFIG_SHA256"]
GLOBAL_LOCK = Path(os.environ["HOSE_GLOBAL_LOCK"]).resolve(strict=True)
with Path(os.environ["HOSE_RESOURCE_BASELINE"]).open(encoding="utf-8") as baseline:
    BASELINES = {int(gpu): {int(pid): entry for pid, entry in rows.items()}
                 for gpu, rows in json.load(baseline).items()}

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(2**20), b''): h.update(part)
    return h.hexdigest()
def atomic(path, data):
    path = Path(path); part = path.with_name(path.name + '.pending.' + str(os.getpid()))
    with part.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False); stream.flush(); os.fsync(stream.fileno())
    os.replace(part, path)
def identity(pid):
    p = Path('/proc') / str(pid); a = (p/'stat').read_text().rsplit(')', 1)[1].split()
    command = (p/'cmdline').read_bytes(); b = (p/'stat').read_text().rsplit(')', 1)[1].split()
    if a[19] != b[19] or a[0] == 'Z': raise ValueError('PROCESS_IDENTITY_CHANGED')
    return {'pid': pid, 'start_token': a[19], 'command_sha256': hashlib.sha256(command).hexdigest()}
def live(who):
    try: return bool(who) and identity(who['pid']) == who
    except (OSError, ValueError): return False
def check_archive(archive, digest, target):
    if sha(archive) != digest: raise ValueError('FROZEN_SOURCE_ARCHIVE_CHANGED')
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if sha(Path(target)/name) != hashlib.sha256(z.read(name)).hexdigest(): raise ValueError('FROZEN_SOURCE_MEMBER_CHANGED')
def ensure_frozen():
    if sha(CONFIG)!=CONFIG_SHA:raise ValueError('R18_CONFIG_CHANGED')
    cfg=read(CONFIG);install=read(ROOT/'receipts'/f'{INTENT}_installation.json')
    check_archive(cfg['source_archive_path'],install['source_sha256'],SOURCE)
    check_archive(cfg['data_archive_path'],cfg['data_archive_sha256'],Path(cfg['data_root']))
    for file,key in [('manifest.jsonl','data_manifest_sha256'),('dataset.json','dataset_metadata_sha256')]:
        if sha(Path(cfg['data_root'])/file)!=cfg[key]:raise ValueError('R18_DATA_CHANGED')
    if sha(Path(cfg['mc_root'])/'manifest.jsonl')!=cfg['mc_manifest_sha256'] or sha(cfg['mc_archive_path'])!=cfg['mc_archive_sha256']:raise ValueError('FROZEN_MC_PARENT_DATA_CHANGED')
    if sha(cfg['parent_checkpoint'])!=cfg['parent_checkpoint_sha256'] or sha(cfg['plan_file'])!=cfg['plan_sha256']:raise ValueError('R18_PARENT_OR_PLAN_CHANGED')
    check_archive(cfg['r17_data_archive_path'],cfg['r17_data_archive_sha256'],Path(cfg['r17_data_root']))
    if sha(Path(cfg['r17_data_root'])/'manifest.jsonl')!=cfg['r17_manifest_sha256'] or sha(cfg['initial_delta_checkpoint'])!=cfg['initial_delta_sha256']:raise ValueError('R18_INITIAL_STATE_OR_REPLAY_CHANGED')
    return cfg,{}


def query_gpu(gpu):
    q = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-gpu=index,name,memory.used,memory.free,utilization.gpu,uuid', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
    if q.returncode: raise ValueError('GPU_SNAPSHOT_FAILED')
    c = [s.strip() for s in q.stdout.strip().split(',')]; uuid = c[5]
    result = {'physical_index': int(c[0]), 'model': c[1], 'used_MiB': int(c[2]), 'free_MiB': int(c[3]), 'utilization_percent': int(c[4]), 'time_utc': now(), 'processes': []}
    q = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid,used_gpu_memory,gpu_uuid', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
    if q.returncode: raise ValueError('GPU_PROCESS_SNAPSHOT_FAILED')
    for line in q.stdout.splitlines():
        c = [s.strip() for s in line.split(',')]
        if len(c) != 3 or not c[0].isdigit() or c[2] != uuid: continue
        item = identity(int(c[0])); item['memory_MiB'] = int(c[1]) if c[1].isdigit() else None
        result['processes'].append(item)
    return result
def check_snapshot(snap, gpu, own_pid=None):
    if gpu not in BASELINES or snap['physical_index'] != gpu or 'A40' not in snap['model']: raise ValueError('ALLOCATED_GPU_IDENTITY_MISMATCH')
    if snap['free_MiB'] < (4096 if own_pid is not None else 14336): raise ValueError('GPU_MEMORY_RESERVE_FAILED')
    for item in snap['processes']:
        if own_pid is not None and item['pid'] == own_pid:
            if item['memory_MiB'] is None or item['memory_MiB'] > 10240: raise ValueError('OWN_PROCESS_MEMORY_BOUND_EXCEEDED')
            continue
        expected = BASELINES[gpu].get(item['pid'])
        if not expected or any(item[key] != expected[key] for key in ('start_token', 'command_sha256')): raise ValueError('UNKNOWN_GPU_PROCESS')
        if item['memory_MiB'] is None or item['memory_MiB'] > expected['memory_upper_MiB']: raise ValueError('CORESIDENT_MEMORY_BOUND_CHANGED')
    return True
def check_fds(owner_pid, gpu):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu): raise ValueError('PHYSICAL_GPU_VISIBILITY_MISMATCH')
    kernel = Path('/proc/locks').read_text().splitlines()
    for envname, path in (('HOSE_GLOBAL_DISPATCH_FD', GLOBAL_LOCK), ('HOSE_GPU_LEASE_FD', Path(f'/var/tmp/score66_gpu{gpu}.lock'))):
        fd = int(os.environ.get(envname, '-1'))
        if fd < 0: raise ValueError('INHERITED_REAL_LOCK_FD_REQUIRED')
        a = os.fstat(fd); b = path.stat()
        if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino): raise ValueError('LOCK_FD_INODE_MISMATCH')
        matched = False
        for line in kernel:
            parts = line.split()
            if len(parts) >= 6 and parts[1] == 'FLOCK' and parts[3] == 'WRITE' and parts[4] == str(owner_pid):
                c = parts[5].split(':')
                if len(c) == 3 and int(c[0], 16) == os.major(b.st_dev) and int(c[1], 16) == os.minor(b.st_dev) and int(c[2]) == b.st_ino: matched = True
        if not matched: raise ValueError('OWNER_KERNEL_FLOCK_NOT_HELD')
def disk_gate(atomic_save=False):
    cfg=read(CONFIG)
    needed = cfg['required_growth_margin_bytes'] + (cfg['atomic_checkpoint_upper_bound_bytes'] if atomic_save else 0)
    if shutil.disk_usage(ROOT).free < needed: raise ValueError('ATOMIC_SAVE_DISK_MARGIN_FAILED')
