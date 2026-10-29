#!/bin/bash
# A Mac cabled to this node that runs nothing of the mesh yet is served bootstrap.sh over the cable, so the one line
# typed on it is `curl m.local|sh` (README, Bringing up a new node): sh asks for its password once and runs
# bootstrap.sh as root.  m.local is this node's mDNS proxy record for the cable port's link-local address, and the
# server answers on that address alone.  Both last while this runs; nothing starts it, and Ctrl-C ends both.
#   mesh-serve-bootstrap.sh [PORT]
# PORT is the fabric port the Mac is cabled to.  Without it, the port is the one whose neighbour holds a link-local
# address that mesh-fabric-init did not give it (a stock Mac's, which is not EUI-64).
set -u
here=$(cd "$(dirname "$0")/.." && pwd -P)
port=${1:-}
if [ -z "$port" ]; then
  for p in $(bash "$here/bin/mesh-rdma.sh" | awk '/rdma_peer=present/{print $1}'); do
    ping6 -c2 -i0.3 "ff02::1%$p" 2>/dev/null | grep -o 'from fe80::[0-9a-f:]*' | cut -d' ' -f2 | grep -qv ':[0-9a-f]*ff:fe[0-9a-f]*:' \
      && port="$port $p"
  done
  set -- $port
  [ $# -eq 1 ] || { echo "mesh-serve-bootstrap.sh PORT: ${#}${port:+ (}${port# }${port:+)} ports have a stock Mac on them; name one" >&2; exit 64; }
  port=$1
fi
address=$(ifconfig "$port" 2>/dev/null | awk '/inet6 fe80::/{sub(/%.*/, "", $2); print $2; exit}')
[ -n "$address" ] || { echo "$port has no link-local address (mesh-fabric-init gives one)" >&2; exit 1; }
trap 'kill $(jobs -p) 2>/dev/null' EXIT INT TERM
dns-sd -P m _http._tcp local 80 m.local "$address" &
sudo -n /usr/bin/python3 - "$port" "$address" "$here/bootstrap.sh" <<'PY' &
import http.server, socket, sys
port, address, bootstrap = sys.argv[1:]
# the page `curl m.local|sh` runs: bootstrap.sh as root through sudo, which asks on the terminal, not on the pipe
page = b"sudo bash <<'MESH_BOOTSTRAP'\n" + open(bootstrap, 'rb').read() + b"\nMESH_BOOTSTRAP\n"
class Page(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.send_header('Content-Length', str(len(page)))
        self.end_headers()
        self.wfile.write(page)
class Server(http.server.HTTPServer):
    address_family = socket.AF_INET6
print(f'serving bootstrap.sh as http://m.local/ on {port} ({address})', file=sys.stderr, flush=True)
Server((address, 80, 0, socket.if_nametoindex(port)), Page).serve_forever()
PY
wait
