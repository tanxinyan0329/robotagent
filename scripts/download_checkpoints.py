"""Download ManiSkill's official Panda PPO checkpoints (and optionally demonstrations).

Source: Hugging Face dataset haosulab/ManiSkill_Demonstrations (the same files as
`python -m mani_skill.utils.download_demo`). Each task is a zip; by default only rl/*_ckpt.pt and
rl/*.mp4 samples are kept, under skills/checkpoints/maniskill/<task>/rl/.

    python scripts/download_checkpoints.py                 # tasks listed in skills/catalog.json
    python scripts/download_checkpoints.py PickCube-v1     # one task
    python scripts/download_checkpoints.py --with-demos    # also keep trajectories (*.h5/*.json)
    python scripts/download_checkpoints.py --from-local ~/projects/maniskill/demos   # copy an existing download

If huggingface.co is unreachable: export HF_ENDPOINT=https://hf-mirror.com
Already-present files are skipped, so an interrupted run can simply be restarted.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request
import zipfile

REPO = Path(__file__).resolve().parents[1]
TARGET = REPO / "skills/checkpoints/maniskill"


def catalog_tasks():
    catalog = json.loads((REPO / "skills/catalog.json").read_text(encoding="utf-8"))
    return sorted({s["source_task"] for s in catalog["local_skills"] if s.get("download")})


def keep(name, with_demos):
    path = Path(name)
    if with_demos:
        return not name.endswith("/")
    return len(path.parts) >= 2 and path.parts[-2] == "rl" and (path.name.endswith("_ckpt.pt") or path.suffix == ".mp4")


def wanted_done(task, with_demos):
    rl = TARGET / task / "rl"
    return rl.is_dir() and any(rl.glob("*_ckpt.pt")) and (not with_demos or any((TARGET / task).rglob("*.h5")))


def download(task, with_demos):
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = f"{endpoint}/datasets/haosulab/ManiSkill_Demonstrations/resolve/main/demos/{task}.zip?download=true"
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / f"{task}.zip"
        print(f"[{task}] downloading {url}", flush=True)
        urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as z:
            for member in z.namelist():
                if not keep(member, with_demos):
                    continue
                parts = Path(member).parts
                if parts[0] != task or ".." in parts:
                    continue
                out = TARGET.joinpath(*parts)
                out.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst)


def copy_local(task, source, with_demos):
    src = Path(source).expanduser() / task
    if not src.is_dir():
        raise FileNotFoundError(f"{src} not found")
    for path in src.rglob("*"):
        rel = path.relative_to(src.parent)
        if path.is_file() and keep(rel.as_posix(), with_demos):
            out = TARGET / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tasks", nargs="*", help="ManiSkill task ids (default: all tasks in skills/catalog.json)")
    parser.add_argument("--with-demos", action="store_true", help="also keep demonstration trajectories")
    parser.add_argument("--from-local", metavar="DIR", help="copy from an existing download_demo output directory")
    parser.add_argument("--list", action="store_true", help="only print the task list")
    args = parser.parse_args()
    tasks = args.tasks or catalog_tasks()
    if args.list:
        print("\n".join(tasks))
        return 0
    failed = []
    for task in tasks:
        if wanted_done(task, args.with_demos):
            print(f"[{task}] already present, skipped")
            continue
        try:
            copy_local(task, args.from_local, args.with_demos) if args.from_local else download(task, args.with_demos)
            found = sorted(p.name for p in (TARGET / task / "rl").glob("*_ckpt.pt"))
            print(f"[{task}] {', '.join(found) or 'no PPO checkpoint in this task (demonstrations only)'}")
        except Exception as error:
            failed.append(task)
            print(f"[{task}] FAILED: {error}", file=sys.stderr)
    if failed:
        print("failed: " + " ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
