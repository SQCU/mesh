from __future__ import annotations

import json
import os
import shlex
import subprocess
import threading
import time
from collections import deque

def split_source(source: str):
    if source.startswith("/") or source.startswith("."):
        return None, source
    if ":" in source:
        host, path = source.split(":", 1)
        if path.startswith("/") or path.startswith("~"):
            return host, path
    return None, source

def follow_argv(source: str, host_key_alias=None, lines=900):
    host, path = split_source(source)
    remote = f"tail -n {max(1, int(lines))} -F {shlex.quote(path)} 2>/dev/null"
    if host is None:
        return ["/bin/sh", "-c", remote]
    argv = [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
        "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=3",
    ]
    if host_key_alias:
        argv += ["-o", f"HostName={host}", "-o", f"HostKeyAlias={host_key_alias}",
                 "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"]
    return argv + [host_key_alias or host, remote]

class TelemetryFollower:
    def __init__(self, source: str, *, capacity: int = 900, on_frame=None,
                 retry: float = 2.0, host_key_alias=None):
        self.source = source
        self.capacity = int(capacity)
        self.on_frame = on_frame
        self.retry = float(retry)
        self.host_key_alias = host_key_alias
        self.frames = deque(maxlen=self.capacity)
        self.lock = threading.Lock()
        self.state = "starting"
        self.detail = ""
        self.attached_at = None
        self.attempts = 0
        self.lines_seen = 0
        self.parse_errors = 0
        self.consumer_errors = 0
        self.last_frame_at = None
        self.last_error = None
        self.epochs = 0
        self.resp_id_resets = 0
        self._last_resp_id = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="joracle-follow", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            self.attempts += 1
            self.epochs += 1
            argv = follow_argv(self.source, self.host_key_alias, self.capacity)
            try:
                proc = subprocess.Popen(
                    argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, bufsize=1,
                )
            except Exception as exc:
                self._fail(f"spawn failed: {exc}")
                self._sleep_retry()
                continue
            self.state = "attached"
            self.detail = " ".join(argv[-2:])
            self.attached_at = time.time()
            try:
                for line in proc.stdout:
                    if self._stop.is_set():
                        break
                    self._ingest(line)
            except Exception as exc:
                self._fail(f"read failed: {exc}")
            finally:
                detached = True
                try:
                    proc.terminate()
                    stderr = proc.communicate(timeout=3)[1]
                except subprocess.TimeoutExpired:
                    stderr = ""
                    detached = False
                    self._fail(f"detach pending: pid {proc.pid} did not exit after TERM")
                except Exception as exc:
                    stderr = ""
                    self._fail(f"detach failed: {type(exc).__name__}: {exc}")
            if not detached:
                break
            if self._stop.is_set():
                break
            self.state = "reattaching"
            if stderr and stderr.strip():
                self.last_error = stderr.strip().splitlines()[-1][:300]
            self._sleep_retry()
        self.state = "stopped"

    def _sleep_retry(self):
        deadline = time.time() + self.retry
        while time.time() < deadline and not self._stop.is_set():
            time.sleep(0.1)

    def _fail(self, message):
        self.state = "error"
        self.last_error = message
        self.detail = message

    def _ingest(self, line):
        line = line.strip()
        if not line:
            return
        self.lines_seen += 1
        try:
            frame = json.loads(line)
        except Exception:
            self.parse_errors += 1
            return
        if not isinstance(frame, dict) or "resp_id" not in frame:
            self.parse_errors += 1
            return
        resp_id = frame.get("resp_id")
        if isinstance(resp_id, int) and self._last_resp_id is not None and resp_id < self._last_resp_id:
            self.resp_id_resets += 1
        if isinstance(resp_id, int):
            self._last_resp_id = resp_id
        frame["_seen_at"] = time.time()
        frame["_epoch"] = self.epochs
        if self.on_frame is not None:
            try:
                self.on_frame(frame)
            except Exception as exc:
                self.consumer_errors += 1
                self.last_error = f"consumer: {type(exc).__name__}: {exc}"
        with self.lock:
            self.frames.append(frame)
        self.last_frame_at = frame["_seen_at"]

    def snapshot(self, limit=None):
        with self.lock:
            frames = list(self.frames)
        return frames if limit is None else frames[-limit:]

    def latest(self):
        with self.lock:
            return self.frames[-1] if self.frames else None

    def status(self):
        now = time.time()
        return {
            "source": self.source,
            "state": self.state,
            "detail": self.detail,
            "attempts": self.attempts,
            "epochs": self.epochs,
            "resp_id_resets": self.resp_id_resets,
            "lines_seen": self.lines_seen,
            "parse_errors": self.parse_errors,
            "consumer_errors": self.consumer_errors,
            "frames_buffered": len(self.frames),
            "capacity": self.capacity,
            "attached_for": None if self.attached_at is None else round(now - self.attached_at, 1),
            "seconds_since_frame": None if self.last_frame_at is None else round(now - self.last_frame_at, 2),
            "last_error": self.last_error,
        }

__all__ = ["TelemetryFollower", "follow_argv", "split_source"]
