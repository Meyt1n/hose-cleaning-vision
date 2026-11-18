"""Create a new training configuration from a recipe and server-side bindings."""

import argparse
import json
import re
from pathlib import Path, PurePosixPath


RECIPES = ["r18_pose", "r19_detector", "r26_action", "r29_fallback"]


def expand(value, root):
    if isinstance(value, str):
        return value.replace("$TRAIN_ROOT", root)
    if isinstance(value, dict):
        return {expand(key, root): expand(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [expand(item, root) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description="生成新训练配置，不启动优化")
    parser.add_argument("--recipe", choices=RECIPES, required=True)
    parser.add_argument("--train-root", required=True, help="服务器项目根目录的绝对路径")
    parser.add_argument("--intent", required=True, help="新的实验目录名")
    parser.add_argument("--overrides", type=Path, help="当前数据、父权重和资源字段组成的 JSON 对象")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = PurePosixPath(args.train_root)
    if not root.is_absolute() or ".." in root.parts or root == PurePosixPath("/"):
        parser.error("--train-root 需为服务器项目根目录的绝对路径")
    if not re.fullmatch(r"[a-z][a-z0-9_-]{2,63}", args.intent):
        parser.error("--intent 需为 3 到 64 个小写字母、数字、短横线或下划线，以字母开头")
    source = Path(__file__).resolve().parent / args.recipe / "config.reference.json"
    cfg = json.loads(source.read_text(encoding="utf-8"))
    name_key = "intent_id" if "intent_id" in cfg else "intent"
    if cfg[name_key] == args.intent:
        parser.error("请为新一轮使用新实验名")
    if args.overrides:
        overrides = json.loads(args.overrides.read_text(encoding="utf-8"))
        if not isinstance(overrides, dict):
            parser.error("--overrides 需要 JSON 对象")
        cfg.update(overrides)
    cfg[name_key] = args.intent
    cfg = expand(cfg, str(root))
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(cfg, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"配置 {args.output}。请在服务器与新安装回执、数据目录和当前资源分配一起核对。")


if __name__ == "__main__":
    main()
