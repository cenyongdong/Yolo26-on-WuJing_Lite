"""本项目的产物与子进程记录；不访问设备，不在超时时终止 EDA 进程。"""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(str(temporary), str(path))


def run_logged(command, directory, label, env=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    command = [str(part) for part in command]
    started = time.monotonic()
    with (directory / (label + ".log")).open("xb") as output:
        process = subprocess.Popen(command, cwd=str(directory), env=env,
                                   stdout=output, stderr=subprocess.STDOUT)
        record = dict(command=command, cwd=str(directory), pid=process.pid,
                      started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save_json(directory / (label + ".process.json"), record)
        code = process.wait()
    record.update(native_exit=code, elapsed_seconds=time.monotonic() - started)
    save_json(directory / (label + ".exit.json"), record)
    if code:
        raise RuntimeError("{} failed ({}), see {}".format(label, code, directory / (label + ".log")))
    return record

