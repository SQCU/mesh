#!/usr/bin/env python3
"""A report of the fabric as it is, member by member, read only: what each Thunderbolt port's cable reaches, what the
bridge paired it with, and the IP paths beside them. It prints; it changes nothing on any member.

    python3 extras/fabric_report.py [--json]

For every member (etc/mesh-nodes.json, reached over its management (tailnet) address):
  - bridge0: its running members and the stored automatic Thunderbolt Bridge definition. Either one lets link-local
    HELLOs cross between cables, so a port can pair with a node not on its cable (m5ult3, 2026-11-01:
    bin/mesh-fabric-init.sh removes the definition).
  - each fabric port: its interface's link status, whether an RDMA peer is present (bin/mesh-rdma.sh), the node the
    bridge paired it with (mesh-stat; a link's index named by its last line in the bridge's log), and the node its cable
    reaches. The cable's node is the one whose Thunderbolt controller holds the Domain UUID that the port's XDomain link
    names (IOThunderboltXDomainLink over the port's AppleThunderboltRDMAPeerInterface, matched by RID). A port whose
    pairing is not its cable's node is marked MISPAIRED.
  - tailnet IPv6 routes that do not go to utun: a babel /128 host route over a Thunderbolt port takes a member's tailnet
    IPv6 address into the fabric.
  - reachability of every other member's ssh port over its fabric address and over its tailnet IPv4 address.
"""
import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rdma'))
import mesh_nodes  # noqa: E402

# Run on each member by its own /usr/bin/python3 (3.9): one JSON object on stdout.
PROBE = r'''
import json, plistlib, re, socket, subprocess, sys, threading
others = json.loads(sys.argv[1])
def run(*argv, timeout=8):
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout).stdout
    except Exception as e:
        return ''
def plist(*argv):
    try:
        return plistlib.loads(subprocess.run(argv, capture_output=True, timeout=8).stdout or b'<plist><array/></plist>')
    except Exception:
        return []
out = {}
running = run('ifconfig', 'bridge0')
out['bridge0_members'] = re.findall(r'member: (\S+)', running)
stored = run('/usr/libexec/PlistBuddy', '-c', 'Print :VirtualNetworkInterfaces:Bridge:bridge0:Interfaces',
             '/Library/Preferences/SystemConfiguration/preferences.plist')
out['bridge0_stored'] = re.findall(r'^\s+(\S+)$', stored, re.M) if 'Array' in stored else []
ports = {}
import os
for line in run(os.path.expanduser('~/mesh/bin/mesh-rdma.sh')).splitlines():
    f = line.split()
    if len(f) < 4:
        continue
    fields = dict(x.split('=', 1) for x in f[1:] if '=' in x)
    status = re.search(r'status: (\w+)', run('ifconfig', f[0]))
    ports[f[0]] = dict(device=fields.get('device'), rid=int(fields.get('rid', -1)), rdma_peer=fields.get('rdma_peer'),
                       link=status.group(1) if status else None)
# the port's cable: its XDomain link's remote Domain UUID, found over the RDMA peer interface the link carries (by RID)
def descendants(entry):
    for child in entry.get('IORegistryEntryChildren', []):
        yield child
        yield from descendants(child)
cable = {}
for link in plist('ioreg', '-a', '-r', '-c', 'IOThunderboltXDomainLink', '-l'):
    for child in descendants(link):
        if child.get('IOClass') == 'AppleThunderboltRDMAPeerInterface' and 'RID' in child:
            cable[int(child['RID'])] = link.get('Domain UUID')
for port in ports.values():
    port['cable_uuid'] = cable.get(port['rid'])
out['ports'] = ports
# this node's own Domain UUIDs (its controllers' local nodes), the ones its neighbours' XDomain links name
own = set()
for cls in ('IOThunderboltController', 'IOThunderboltLocalNode'):
    for entry in plist('ioreg', '-a', '-r', '-c', cls, '-l'):
        for e in (entry, *descendants(entry)):
            if e.get('IOClass') in ('IOThunderboltLocalNode',) or e is entry:
                if e.get('Domain UUID'):
                    own.add(e['Domain UUID'])
out['own_uuids'] = sorted(own - set(cable.values()))
# the bridge's pairings, and each link index's last interface name in its log
stat = run(os.path.expanduser('~/mesh/rdma/mesh-stat')).splitlines()
try:
    peers = json.loads(stat[0]).get('peers', [])
except Exception:
    peers = []
names = {}
try:
    for m in re.finditer(r'link (\d+) \((en\d+)\)', open(os.path.expanduser('~/Library/Logs/io.mesh.bridge.log'), errors='replace').read()[-2000000:]):
        names[int(m.group(1))] = m.group(2)
except Exception:
    pass
out['pairings'] = [dict(port=p.get('port'), interface=names.get(p.get('port')), node=p.get('node'), phase=p.get('phase')) for p in peers]
out['tailnet6_off_utun'] = [line.split()[:4] for line in run('netstat', '-rn', '-f', 'inet6').splitlines()
                            if line.startswith('fd7a:115c:a1e0:') and 'utun' not in line and 'lo0' not in line]
reach = {}
def probe(name, kind, address):
    try:
        s = socket.create_connection((address, 22), timeout=1.5); s.close(); reach[(name, kind)] = 'ok'
    except Exception as e:
        reach[(name, kind)] = type(e).__name__ + (': ' + str(e.errno) if getattr(e, 'errno', None) else '')
threads = [threading.Thread(target=probe, args=(o['name'], kind, o[kind])) for o in others for kind in ('fabric', 'tailnet4')]
for t in threads: t.start()
for t in threads: t.join()
out['reach'] = {f'{n} {k}': v for (n, k), v in reach.items()}
print(json.dumps(out))
'''


def gather(cluster, member, members):
    others = [dict(name=m.name, fabric=m.fabric, tailnet4=m.management) for m in members if m is not member]
    try:
        r = subprocess.run(['ssh', *cluster.ssh, '-o', 'ServerAliveInterval=2', '-o', 'ConnectTimeout=5',
                            f'{cluster.user}@{member.management}', '/usr/bin/python3 - ' + json.dumps(json.dumps(others))],
                           input=PROBE, capture_output=True, text=True, timeout=30)
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return dict(error=f'{type(e).__name__}: {e}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    cluster = mesh_nodes.load()
    members = sorted((m for m in cluster.members.values() if 'workstation' not in m.roles), key=lambda m: int(m.number))
    with concurrent.futures.ThreadPoolExecutor(len(members)) as pool:
        reports = dict(zip((m.name for m in members), pool.map(lambda m: gather(cluster, m, members), members)))
    if args.json:
        print(json.dumps(reports, indent=1))
        return
    owner = {uuid: name for name, r in reports.items() for uuid in r.get('own_uuids', [])}
    by_number = {int(m.number): m.name for m in members}
    for name, r in reports.items():
        print(f'== {name}')
        if 'error' in r:
            print('   unreachable:', r['error'])
            continue
        print(f"   bridge0 running members: {r['bridge0_members'] or 'none'}; stored definition: {r['bridge0_stored'] or 'none'}")
        paired = {p['interface']: p for p in r['pairings'] if p['interface']}
        for interface, port in sorted(r['ports'].items(), key=lambda item: item[1]['rid']):
            cable_node = owner.get(port['cable_uuid'], port['cable_uuid'] or '-')
            p = paired.get(interface, {})
            bridge_node = by_number.get(p.get('node'), p.get('node'))
            mark = ''
            if p.get('phase') == 'PAIRED' and port['cable_uuid'] and bridge_node != cable_node:
                mark = '  MISPAIRED'
            elif port['rdma_peer'] == 'present' and p.get('phase') != 'PAIRED':
                mark = '  UNPAIRED (an RDMA peer is present)'
            print(f"   {interface:5} rid {port['rid']} link {port['link'] or '-':8} rdma_peer {port['rdma_peer'] or '-':7} "
                  f"cable -> {cable_node:12} bridge -> {str(bridge_node or '-'):12} {p.get('phase', '-')}{mark}")
        if r['tailnet6_off_utun']:
            print('   tailnet IPv6 routed off utun:', [' '.join(x) for x in r['tailnet6_off_utun']])
        bad = {k: v for k, v in r['reach'].items() if v != 'ok'}
        print(f"   reach: {len(r['reach']) - len(bad)} of {len(r['reach'])} ok" + (f'; failing {bad}' if bad else ''))


if __name__ == '__main__':
    main()
