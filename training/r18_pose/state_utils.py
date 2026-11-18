from pathlib import Path
import torch, numpy as np, hashlib, random, json, os, fcntl, math
from runtime import RUN, disk_gate

def digest(obj):
    h = hashlib.sha256()

    def add(x):
        if torch.is_tensor(x):
            a = x.detach().cpu().contiguous().numpy()
            h.update(str((a.dtype, a.shape)).encode())
            h.update(a.tobytes())
        elif isinstance(x, np.ndarray):
            h.update(str((x.dtype, x.shape)).encode())
            h.update(x.tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=str):
                h.update(str(k).encode())
                add(x[k])
        elif isinstance(x, (list, tuple)):
            for v in x:
                add(v)
        elif isinstance(x, np.generic):
            add(x.item())
        else:
            h.update(repr(x).encode())
    add(obj)
    return h.hexdigest()

def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}

def set_rng(s):
    random.setstate(s['python'])
    np.random.set_state(s['numpy'])
    torch.set_rng_state(s['torch'].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in s['cuda']])

def save(path, data):
    path = Path(path)
    RUN.mkdir(exist_ok=True)
    fd = os.open(RUN / 'checkpoint_write.lock', os.O_CREAT | os.O_RDWR, 384)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        disk_gate(atomic_save=True)
        pending = path.with_name(path.name + '.pending.' + str(os.getpid()))
        with pending.open('xb') as f:
            torch.save(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(pending, path)
    finally:
        os.close(fd)
