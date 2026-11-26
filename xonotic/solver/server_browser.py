#!/usr/bin/env python3
"""Fleet Xonotic Server Browser REST API & Web UI.

Provides uniform discovery, live DarkPlaces UDP `getstatus` probing, registration,
and remote launch across local loopback (MacBook Pro userspace) and all 4 Apple M5 Ultra
Mac Studios (and Mac Mini) over Tailscale / LAN.

Usage:
    python3 xonotic/solver/server_browser.py --host 0.0.0.0 --port 8765
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shlex
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

SSH_KEY = os.environ.get("MESH_SSH_KEY", "/Users/mdot/.ssh/primeintellect_ed25519")
TAILSCALE_BIN = "/Applications/Tailscale.app/Contents/MacOS/Tailscale"
DEFAULT_STUDIO_NODES = [
    {"id": "studio-1", "name": "mdots-Mac-Studio.local", "host": "100.103.69.53", "role": "studio", "arch": "Apple M5 Ultra"},
    {"id": "studio-2", "name": "mdots-Mac-Studio-2.local", "host": "100.65.151.124", "role": "studio", "arch": "Apple M5 Ultra"},
    {"id": "studio-3", "name": "mdots-Mac-Studio-3.local", "host": "100.107.135.46", "role": "studio", "arch": "Apple M5 Ultra"},
    {"id": "studio-4", "name": "m5ult2.local", "host": "100.86.148.87", "role": "studio", "arch": "Apple M5 Ultra"},
    {"id": "mini", "name": "Ms-Mac-mini", "host": "100.77.188.102", "role": "mini", "arch": "Apple M4 Pro"},
]
DEFAULT_SCAN_PORTS = (26000, 26001, 26002, 26003, 27000, 27001, 27002, 27003)
REPO_ROOT = Path(__file__).resolve().parents[2]
CLIENT_LAUNCHER = str(REPO_ROOT / "bin" / "mesh-xonotic-client")
REGISTRY_STATE_PATH = Path(os.environ.get("MESH_SERVER_BROWSER_STATE", "/tmp/mesh_server_browser_registry.json"))


class ReuseThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def discover_fleet_nodes() -> list[dict]:
    """Return local loopback + all online macOS mesh nodes discovered via Tailscale."""
    nodes = [{
        "id": "local",
        "name": socket.gethostname(),
        "host": "127.0.0.1",
        "role": "userspace",
        "online": True,
        "arch": "macOS local",
    }]
    known_by_ip = {n["host"]: dict(n) for n in DEFAULT_STUDIO_NODES}
    ts_found = {}
    if Path(TAILSCALE_BIN).exists():
        try:
            out = subprocess.check_output([TAILSCALE_BIN, "status", "--json"], timeout=3.0, stderr=subprocess.DEVNULL)
            data = json.loads(out.decode(errors="replace"))
            for peer in (data.get("Peer") or {}).values():
                if peer.get("OS") != "macOS":
                    continue
                ips = peer.get("TailscaleIPs") or []
                if not ips:
                    continue
                ip = str(ips[0])
                base = known_by_ip.get(ip, {
                    "id": str(peer.get("HostName", ip)),
                    "name": str(peer.get("DNSName") or peer.get("HostName") or ip).rstrip("."),
                    "host": ip,
                    "role": "studio",
                    "arch": "macOS",
                })
                base["online"] = bool(peer.get("Online", False))
                ts_found[ip] = base
        except Exception:
            pass
    for n in DEFAULT_STUDIO_NODES:
        rec = ts_found.pop(n["host"], {**n, "online": True})
        nodes.append(rec)
    for rec in ts_found.values():
        nodes.append(rec)
    return nodes


def probe_darkplaces_server(host: str, port: int, timeout: float = 0.35) -> dict:
    """Query a live DarkPlaces/Xonotic UDP server with Quake3/DP `getstatus`."""
    t0 = time.perf_counter()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout)
    try:
        sock.sendto(b"\xff\xff\xff\xffgetstatus\n", (host, int(port)))
        raw, _ = sock.recvfrom(8192)
        ping_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    except OSError:
        return {"online": False, "host": host, "port": int(port)}
    finally:
        sock.close()

    if not raw.startswith(b"\xff\xff\xff\xffstatusResponse"):
        return {"online": False, "host": host, "port": int(port)}
    lines = raw.decode("latin-1", errors="replace").splitlines()
    kv_line = lines[1] if len(lines) > 1 else ""
    parts = kv_line.lstrip("\\").split("\\")
    info = {parts[i]: parts[i + 1] for i in range(0, len(parts) - 1, 2)}
    players = []
    for pl in lines[2:]:
        pl = pl.strip()
        if not pl:
            continue
        toks = pl.split(None, 2)
        score = int(toks[0]) if len(toks) > 0 and toks[0].lstrip("-").isdigit() else 0
        ping = int(toks[1]) if len(toks) > 1 and toks[1].isdigit() else 0
        name = toks[2].strip('"') if len(toks) > 2 else ""
        players.append({"score": score, "ping": ping, "name": name})

    return {
        "online": True,
        "host": host,
        "port": int(port),
        "ping_ms": ping_ms,
        "hostname": info.get("hostname", ""),
        "mapname": info.get("mapname", ""),
        "gamename": info.get("gamename", ""),
        "clients": int(info.get("clients", len(players))),
        "maxclients": int(info.get("sv_maxclients", 16)),
        "bots": int(info.get("bots", 0)),
        "qcstatus": info.get("qcstatus", ""),
        "players": players,
        "connect_cmd": f"+connect {host}:{int(port)}",
        "scope": "loopback" if host in ("127.0.0.1", "localhost") else "remote_studio",
    }


class ServerRegistry:
    def __init__(self, state_path: Path = REGISTRY_STATE_PATH):
        self._lock = threading.Lock()
        self._entries: dict[str, dict] = {}
        self._procs: dict[str, subprocess.Popen] = {}
        self._state_path = state_path
        self._load_state()

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text())
            now = time.time()
            for sid, rec in (data.items() if isinstance(data, dict) else []):
                if isinstance(rec, dict) and now - float(rec.get("updated_at", 0.0)) <= 300.0:
                    self._entries[str(sid)] = rec
        except Exception:
            pass

    def _save_state_locked(self) -> None:
        try:
            tmp = self._state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._entries))
            tmp.replace(self._state_path)
        except Exception:
            pass

    def register(self, entry: dict) -> dict:
        sid = str(entry.get("id") or f"{entry['host']}:{entry['port']}")
        rec = {
            "id": sid,
            "node": str(entry.get("node", "local")),
            "host": str(entry.get("host", "127.0.0.1")),
            "port": int(entry.get("port", 26000)),
            "map": str(entry.get("map", "boil")),
            "policy_arms": entry.get("policy_arms", []),
            "updated_at": time.time(),
        }
        with self._lock:
            self._entries[sid] = rec
            self._save_state_locked()
        return rec

    def unregister(self, sid: str) -> bool:
        with self._lock:
            p = self._procs.pop(sid, None)
            if p is not None and p.poll() is None:
                try:
                    p.terminate()
                except OSError:
                    pass
            removed = self._entries.pop(sid, None) is not None
            if removed:
                self._save_state_locked()
            return removed

    def launch_client(self, req: dict) -> dict:
        host = str(req.get("host", "")).strip()
        port = int(req.get("port", 0))
        if not host or port <= 0 or port > 65535:
            raise ValueError("valid host and port required")
        endpoint = f"{host}:{port}"
        cmd = [CLIENT_LAUNCHER, "-window", "+connect", endpoint]
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return {
            "ok": True,
            "pid": proc.pid,
            "connect_cmd": f"+connect {endpoint}",
            "cli_cmd": f"{CLIENT_LAUNCHER} -window +connect {endpoint}",
        }

    def list_probed(self, scan_defaults: bool = True) -> list[dict]:
        now = time.time()
        with self._lock:
            stale = [k for k, v in self._entries.items() if now - v["updated_at"] > 300.0 and k not in self._procs]
            for k in stale:
                self._entries.pop(k, None)
            targets = {f"{v['host']}:{v['port']}": dict(v) for v in self._entries.values()}

        if scan_defaults:
            nodes = discover_fleet_nodes()
            for n in nodes:
                if not n.get("online", True):
                    continue
                for port in DEFAULT_SCAN_PORTS:
                    key = f"{n['host']}:{port}"
                    if key not in targets:
                        targets[key] = {
                            "id": key,
                            "node": n["name"],
                            "host": n["host"],
                            "port": port,
                            "map": "",
                            "policy_arms": [],
                            "scanned": True,
                        }

        def _probe_one(meta: dict) -> dict | None:
            st = probe_darkplaces_server(meta["host"], meta["port"])
            if not st["online"] and meta.get("scanned"):
                return None
            return {**meta, **st}

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, max(1, len(targets)))) as pool:
            for res in pool.map(_probe_one, targets.values()):
                if res is not None:
                    results.append(res)
        results.sort(key=lambda r: (0 if r.get("online") else 1, r.get("scope", ""), r.get("host", ""), r.get("port", 0)))
        return results

    def launch(self, req: dict) -> dict:
        host = str(req.get("host", "127.0.0.1"))
        port = int(req.get("port", 26000))
        mapname = str(req.get("map", "boil"))
        bots = int(req.get("bots", 9))
        dp_bin = str(req.get("dp_bin", "/tmp/xonotic-live-cgt/darkplaces-dedicated"))
        basedir = str(req.get("basedir", "/Users/mdot/dox/xonotic/Xonotic"))
        userdir = str(req.get("userdir", f"/tmp/xonotic-browser-sv-{port}"))
        src_data = Path(req.get("src_data", "/tmp/xonotic-live-cgt/userdir/data"))
        sid = f"{host}:{port}"

        dp_args = [
            dp_bin, "-xonotic", "-basedir", basedir, "-userdir", userdir,
            "+exec", "gamemodes-payload.cfg", "+sv_public", "0",
            "+hostname", f"Mesh Payload [{mapname} @ {host}]",
            "+port", str(port), "+maxplayers", "16",
            "+g_payload", "1", "+g_payload_teams_override", "3",
            "+g_payload_plan_only", "1", "+g_warmup", "0", "+g_start_delay", "0",
            "+bot_join_empty", "1", "+bot_number", str(bots), "+skill", "5",
            "+map", mapname,
        ]

        if host in ("127.0.0.1", "localhost"):
            ud = Path(userdir) / "data"
            ud.mkdir(parents=True, exist_ok=True)
            if src_data.exists():
                for item in src_data.iterdir():
                    dst = ud / item.name
                    if not dst.exists():
                        os.symlink(item, dst)
            logf = open(Path(userdir) / "server.log", "w")
            proc = subprocess.Popen(dp_args, cwd=basedir, stdout=logf, stderr=subprocess.STDOUT)
            with self._lock:
                self._procs[sid] = proc
        else:
            r_dp = str(req.get("dp_bin", "$HOME/xonotic-stage/darkplaces-dedicated"))
            r_base = str(req.get("basedir", "$HOME/xonotic-stage/Xonotic"))
            r_src = str(req.get("src_data", "$HOME/xonotic-stage/userdir/data"))
            r_args = [
                r_dp, "-xonotic", "-basedir", r_base, "-userdir", userdir,
                "+exec", "gamemodes-payload.cfg", "+sv_public", "0",
                "+hostname", f"Mesh Payload [{mapname} @ {host}]",
                "+port", str(port), "+maxplayers", "16",
                "+g_payload", "1", "+g_payload_teams_override", "3",
                "+g_payload_plan_only", "1", "+g_warmup", "0", "+g_start_delay", "0",
                "+bot_join_empty", "1", "+bot_number", str(bots), "+skill", "5",
                "+map", mapname,
            ]
            remote_cmd = (
                f"mkdir -p {shlex.quote(userdir)}/data && "
                f"for f in {r_src}/*; do ln -sfn \"$f\" {shlex.quote(userdir)}/data/; done && "
                f"nohup {' '.join(a if a.startswith('$HOME') else shlex.quote(a) for a in r_args)} > {shlex.quote(userdir)}/server.log 2>&1 < /dev/null &"
            )
            subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=5",
                 "-i", SSH_KEY, f"mdot@{host}", remote_cmd],
                check=True, timeout=10,
            )

        rec = self.register({"id": sid, "node": host, "host": host, "port": port, "map": mapname, "policy_arms": ["default"]})
        rec["connect_cmd"] = f"+connect {host}:{port}"
        return rec


INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Mesh Xonotic Fleet Server Browser</title>
<style>
body{font-family:system-ui,-apple-system,sans-serif;background:#0d1117;color:#e6edf3;margin:24px;}
h1{font-size:20px;margin:0 0 12px;} .bar{display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap;align-items:center;}
#toast{min-height:22px;margin-bottom:10px;font-size:13px;color:#3fb950;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;}
#toast.err{color:#f85149;}
table{width:100%;border-collapse:collapse;background:#161b22;border:1px solid #30363d;border-radius:8px;overflow:hidden;}
th,td{padding:8px 12px;border-bottom:1px solid #21262d;text-align:left;font-size:13px;}
th{background:#21262d;color:#8b949e;} .ok{color:#3fb950;font-weight:600;} .off{color:#f85149;}
.actions{display:flex;gap:6px;flex-wrap:wrap;}
button,input,select{background:#21262d;color:#e6edf3;border:1px solid #30363d;border-radius:6px;padding:5px 9px;font-size:12px;}
button:hover{background:#30363d;cursor:pointer;}
button.copied{border-color:#3fb950;color:#3fb950;}
button.launch{background:#1f6feb;border-color:#388bfd;color:#fff;}
button.launch:hover{background:#388bfd;}
code{background:#1f242c;padding:2px 6px;border-radius:4px;white-space:nowrap;}
</style></head><body>
<h1>Mesh Xonotic Fleet Server Browser (Loopback + 4 Mac Studios)</h1>
<div class="bar">
  <select id="nodeSel"></select>
  <select id="mapSel"><option>boil</option><option>stormkeep</option><option>catharsis</option><option>afterslime</option><option>dance</option><option>vorix</option><option>glowplant</option></select>
  <input id="portIn" type="number" value="26000" style="width:90px">
  <button onclick="launchSv()">Launch Server</button>
  <button onclick="refresh(true)">Refresh</button>
</div>
<div id="toast"></div>
<table><thead><tr><th>Status</th><th>Scope</th><th>Node</th><th>Endpoint</th><th>Map</th><th>Players (Bots)</th><th>Ping</th><th>Connect</th></tr></thead>
<tbody id="rows"></tbody></table>
<script>
const CLIENT_LAUNCHER = __CLIENT_LAUNCHER_JSON__;
let holdRenderUntil = 0;

function setToast(msg, isErr=false){
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = isErr ? 'err' : '';
}

function fallbackCopy(text){
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.setAttribute('readonly', '');
  ta.style.position = 'fixed';
  ta.style.top = '-9999px';
  ta.style.left = '-9999px';
  document.body.appendChild(ta);
  ta.focus();
  ta.select();
  ta.setSelectionRange(0, text.length);
  let ok = false;
  try { ok = document.execCommand('copy'); } catch (_) { ok = false; }
  document.body.removeChild(ta);
  return ok;
}

async function copyText(raw, btn){
  const clean = String(raw).replace(/[\\r\\n]+/g, ' ').replace(/\\s+/g, ' ').trim();
  let ok = false;
  if (window.isSecureContext && navigator.clipboard && navigator.clipboard.writeText) {
    try { await navigator.clipboard.writeText(clean); ok = true; } catch (_) { ok = false; }
  }
  if (!ok) {
    ok = fallbackCopy(clean);
  }
  holdRenderUntil = Date.now() + 1800;
  if (ok) {
    const prev = btn.textContent;
    btn.textContent = 'Copied!';
    btn.classList.add('copied');
    setToast('Copied to clipboard: ' + clean);
    setTimeout(() => { btn.textContent = prev; btn.classList.remove('copied'); }, 1400);
  } else {
    setToast('Copy blocked by browser — select manually: ' + clean, true);
    window.prompt('Copy command:', clean);
  }
}

async function launchClient(host, port, btn){
  holdRenderUntil = Date.now() + 2000;
  const prev = btn.textContent;
  btn.textContent = 'Launching...';
  try {
    const res = await fetch('/api/client/launch', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({host, port}),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'launch failed');
    btn.textContent = 'Launched!';
    setToast('Launched local Xonotic client (pid ' + data.pid + '): ' + data.connect_cmd);
    setTimeout(() => { btn.textContent = prev; }, 1600);
  } catch (err) {
    btn.textContent = prev;
    setToast('Client launch error: ' + err.message, true);
  }
}

async function refresh(force=false){
  if (!force && Date.now() < holdRenderUntil) return;
  const [nodes, svs] = await Promise.all([fetch('/api/nodes').then(r=>r.json()), fetch('/api/servers').then(r=>r.json())]);
  const sel = document.getElementById('nodeSel');
  if(!sel.options.length){ nodes.forEach(n=>{ const o=document.createElement('option'); o.value=n.host; o.textContent=`${n.name} (${n.host}) [${n.role}]`; sel.appendChild(o); }); }
  const tb = document.getElementById('rows');
  tb.textContent = '';
  if(!svs.length){
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.colSpan = 8;
    td.style.color = '#8b949e';
    td.textContent = 'No active Xonotic servers detected on loopback or 4 Studios.';
    tr.appendChild(td);
    tb.appendChild(tr);
    return;
  }
  svs.forEach(s=>{
    const endpoint = `${String(s.host).trim()}:${Number(s.port)}`;
    const consoleCmd = `connect ${endpoint}`;
    const cliCmd = `${CLIENT_LAUNCHER} -window +connect ${endpoint}`;
    const tr = document.createElement('tr');

    const cells = [
      {cls: s.online ? 'ok' : 'off', text: s.online ? 'ONLINE' : 'OFFLINE'},
      {text: s.scope || ''},
      {text: s.node || ''},
      {code: endpoint},
      {text: s.mapname || s.map || '-'},
      {text: `${s.clients ?? '-'}/${s.maxclients ?? 16} (${s.bots ?? 0})`},
      {text: s.ping_ms != null ? `${s.ping_ms} ms` : '-'},
    ];
    cells.forEach(c => {
      const td = document.createElement('td');
      if (c.cls) td.className = c.cls;
      if (c.code) {
        const codeEl = document.createElement('code');
        codeEl.textContent = c.code;
        td.appendChild(codeEl);
      } else {
        td.textContent = c.text;
      }
      tr.appendChild(td);
    });

    const actTd = document.createElement('td');
    const wrap = document.createElement('div');
    wrap.className = 'actions';

    const bAddr = document.createElement('button');
    bAddr.textContent = `Copy ${endpoint}`;
    bAddr.title = 'Paste into Xonotic Multiplayer -> Address box';
    bAddr.addEventListener('click', () => copyText(endpoint, bAddr));

    const bCon = document.createElement('button');
    bCon.textContent = 'Copy console';
    bCon.title = consoleCmd + ' (for in-game ~ console)';
    bCon.addEventListener('click', () => copyText(consoleCmd, bCon));

    const bCli = document.createElement('button');
    bCli.textContent = 'Copy CLI';
    bCli.title = cliCmd;
    bCli.addEventListener('click', () => copyText(cliCmd, bCli));

    const bLaunch = document.createElement('button');
    bLaunch.textContent = 'Launch Client';
    bLaunch.className = 'launch';
    bLaunch.addEventListener('click', () => launchClient(String(s.host).trim(), Number(s.port), bLaunch));

    wrap.appendChild(bAddr);
    wrap.appendChild(bCon);
    wrap.appendChild(bCli);
    wrap.appendChild(bLaunch);
    actTd.appendChild(wrap);
    tr.appendChild(actTd);
    tb.appendChild(tr);
  });
}
async function launchSv(){
  const body = {host: document.getElementById('nodeSel').value, port: Number(document.getElementById('portIn').value), map: document.getElementById('mapSel').value};
  await fetch('/api/servers/launch', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  setTimeout(() => refresh(true), 1200);
}
refresh(true); setInterval(() => refresh(false), 4000);
</script></body></html>
""".replace("__CLIENT_LAUNCHER_JSON__", json.dumps(CLIENT_LAUNCHER))


def make_handler(registry: ServerRegistry):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send_json(self, code: int, obj):
            body = json.dumps(obj, indent=2).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            p = urlparse(self.path).path
            if p == "/":
                body = INDEX_HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif p == "/api/health":
                self._send_json(200, {"ok": True, "time": time.time()})
            elif p == "/api/nodes":
                self._send_json(200, discover_fleet_nodes())
            elif p == "/api/servers":
                self._send_json(200, registry.list_probed(scan_defaults=True))
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self):
            p = urlparse(self.path).path
            n = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(n) if n > 0 else b"{}"
            try:
                body = json.loads(raw.decode())
            except ValueError:
                return self._send_json(400, {"error": "invalid json"})
            if p == "/api/servers/register":
                return self._send_json(200, registry.register(body))
            if p == "/api/servers/launch":
                try:
                    return self._send_json(200, registry.launch(body))
                except Exception as exc:
                    return self._send_json(500, {"error": str(exc)})
            if p == "/api/client/launch":
                try:
                    return self._send_json(200, registry.launch_client(body))
                except Exception as exc:
                    return self._send_json(500, {"error": str(exc)})
            return self._send_json(404, {"error": "not found"})

        def do_DELETE(self):
            p = urlparse(self.path).path
            if p.startswith("/api/servers/"):
                sid = p[len("/api/servers/"):]
                return self._send_json(200, {"deleted": registry.unregister(sid)})
            return self._send_json(404, {"error": "not found"})

    return Handler


def main():
    ap = argparse.ArgumentParser(description="Mesh Xonotic Fleet Server Browser REST API")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    reg = ServerRegistry()
    srv = ReuseThreadingHTTPServer((args.host, args.port), make_handler(reg))
    print(f"Mesh Server Browser listening on http://{args.host}:{args.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
