"""Install the versioned release weights and validate all SHA256 digests."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile


DEFAULT_MODELS = Path(__file__).resolve().parents[1] / "models"


def verify(path, record):
    if not path.is_file() or path.stat().st_size != record["bytes"]:
        raise ValueError(f"文件缺失或大小错误 {record['file']}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != record["sha256"]:
        raise ValueError(f"SHA256 不匹配 {record['file']}")


def install(archive, model_dir, records):
    with zipfile.ZipFile(archive) as bundle:
        expected = {record["file"] for record in records}
        if len(bundle.namelist()) != len(expected) or set(bundle.namelist()) != expected:
            raise ValueError("Release ZIP 的成员与模型清单不一致")
        with tempfile.TemporaryDirectory(prefix=".weights-", dir=model_dir) as staging:
            staging = Path(staging)
            for record in records:
                relative = record["file"]
                member = bundle.getinfo(relative)
                if member.file_size != record["bytes"] or member.is_dir():
                    raise ValueError(f"压缩包成员大小错误 {relative}")
                target = model_dir / relative
                if target.exists():
                    verify(target, record)
                    continue
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, destination.open("xb") as sink:
                    shutil.copyfileobj(source, sink, length=8 << 20)
                verify(destination, record)
            for record in records:
                relative = record["file"]
                prepared = staging / relative
                if prepared.exists():
                    target = model_dir / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        verify(target, record)
                    else:
                        os.replace(prepared, target)


def main():
    parser = argparse.ArgumentParser(description="安装或核验 Release 带有 SHA256 的推理权重")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--archive", type=Path, help="安装私有 Release 的 inference-weights.zip")
    args = parser.parse_args()
    inventory = json.loads((args.model_dir / "model_inventory.json").read_text(encoding="utf-8"))
    records = inventory["files"]
    if args.archive:
        install(args.archive, args.model_dir, records)
    for record in records:
        verify(args.model_dir / record["file"], record)
        print(f"OK  {record['file']}  {record['bytes']} B")


if __name__ == "__main__":
    main()
