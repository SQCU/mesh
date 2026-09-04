from __future__ import annotations

import json
import os
import shlex
import subprocess
import threading
import time

def run(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=8)
        output = (result.stdout + result.stderr).strip()
        return {"execution_mass": 1, "exception_mass": 0,
                "exit_code": result.returncode, "output": output,
                "output_character_mass": len(output)}
    except Exception as exc:
        output = f"{type(exc).__name__}: {exc}"
        return {"execution_mass": 1, "exception_mass": 1,
                "exit_code": None, "output": output,
                "output_character_mass": len(output)}

def process_rows(output, executable, needles):
    rows = []
    for line in output.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) != 4:
            continue
        command = parts[3]
        try:
            program = os.path.basename(shlex.split(command)[0])
        except (ValueError, IndexError):
            continue
        if executable and not program.startswith(executable):
            continue
        if all(needle in command for needle in needles):
            rows.append(line.strip())
    return rows

def bridge_stats(result):
    for line in reversed(result["output"].splitlines()):
        try:
            value = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict) and ("client" in value or "app" in value):
            return value
    return {}

class RuntimeMeasure:
    def __init__(self, local_repo, remote_host=None, interval=2.0,
                 remote_alias="mesh-mini", remote_repo="/Users/mdot/mesh"):
        self.local_repo = local_repo
        self.remote_host = remote_host
        self.remote_alias = remote_alias
        self.remote_repo = remote_repo
        self.interval = float(interval)
        self.lock = threading.Lock()
        self.value = {"sample_mass": 0}
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, name="joracle-runtime", daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.stop_event.set()

    def snapshot(self):
        with self.lock:
            return dict(self.value)

    def inspect(self):
        local_bridge = run([os.path.join(self.local_repo, "bin", "mesh-bridge.sh"), "status"])
        local_ps = run(["ps", "-axo", "pid=,etime=,comm=,command="])
        games = process_rows(local_ps["output"], "darkplaces-dedicated", ("-xonotic",))
        if self.remote_host:
            ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                   "-o", f"HostName={self.remote_host}",
                   "-o", f"HostKeyAlias={self.remote_alias}",
                   "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", self.remote_alias]
            remote_bridge = run(ssh + [os.path.join(self.remote_repo, "bin", "mesh-bridge.sh"), "status"])
            remote_ps = run(ssh + ["ps", "-axo", "pid=,etime=,comm=,command="])
            responders = process_rows(remote_ps["output"], "python", ("solver.strat.strat_responder",))
        else:
            remote_bridge = local_bridge
            responders = process_rows(local_ps["output"], "python", ("solver.strat.strat_responder",))
        local_stats = bridge_stats(local_bridge)
        remote_stats = bridge_stats(remote_bridge)
        client_mass = int(local_stats.get("client", 0))
        responder_mass = int(remote_stats.get("app", 0))
        return {
            "checked_at": time.time(), "sample_mass": 1,
            "local_bridge": local_bridge, "remote_bridge": remote_bridge,
            "games": games, "responders": responders,
            "game_process_mass": len(games), "responder_process_mass": len(responders),
            "local_bridge_stats": local_stats, "remote_bridge_stats": remote_stats,
            "local_bridge_client_mass": client_mass,
            "remote_bridge_responder_mass": responder_mass,
            "paired_fabric_attachment_mass": min(client_mass, responder_mass),
            "remote_host": self.remote_host,
            "remote_alias": self.remote_alias,
        }

    def run(self):
        while not self.stop_event.is_set():
            value = self.inspect()
            with self.lock:
                self.value = value
            self.stop_event.wait(self.interval)

__all__ = ["RuntimeMeasure", "bridge_stats", "process_rows", "run"]
