#!/usr/bin/env python3
"""Run a command fully detached from the launching shell.

    python scripts/daemonize.py --log FILE -- CMD [ARGS...]

The frozen suite takes hours; a plain ``nohup ... &`` still dies when the
launching shell's process group is torn down (and macOS has no ``setsid(1)``).
This does the standard double fork with ``os.setsid`` between them, so the run
becomes a session leader owned by init and survives.

Prints the detached PID on stdout.
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True)
    p.add_argument("--pidfile")
    p.add_argument("--cwd")
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    args = p.parse_args()

    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        p.error("no command given (use: --log FILE -- CMD ARGS...)")

    r, w = os.pipe()
    if os.fork() > 0:                      # launcher: report the PID and exit
        os.close(w)
        with os.fdopen(r) as fh:
            pid = fh.read().strip()
        os.waitpid(-1, 0)
        print(pid)
        return 0

    os.close(r)
    os.setsid()                            # new session, no controlling terminal
    if os.fork() > 0:                      # intermediate: exit so the grandchild
        os._exit(0)                        # cannot reacquire a terminal

    pid = os.getpid()
    with os.fdopen(w, "w") as fh:
        fh.write(str(pid))
    if args.pidfile:
        with open(args.pidfile, "w") as fh:
            fh.write(f"{pid}\n")

    if args.cwd:
        os.chdir(args.cwd)
    fd = os.open(args.log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.execvp(cmd[0], cmd)                 # never returns


if __name__ == "__main__":
    raise SystemExit(main())
