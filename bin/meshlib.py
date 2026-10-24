# What the node's scripts share (README.md, bin/): the host's identity, its fabric ports, the bridge's own reading,
# the name table, launchd's loading of a job. Python 3.9, the Command Line Tools' /usr/bin/python3: nothing here needs
# the mesh runtime, so a broken runtime takes no node's services with it.
import glob, hashlib, json, locale, os, plistlib, re, subprocess, time

PREFIX = "fd6d:6573:68"
MESH = "/usr/local/mesh"
UUID = re.compile(r"^[0-9A-Fa-f]{8}(-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}$")


def collated(items):
    """ITEMS in the order the shell's globs and sort give them: the environment's collation (LC_COLLATE)."""
    try:
        locale.setlocale(locale.LC_COLLATE, "")
    except locale.Error:
        pass
    return sorted(items, key=locale.strxfrm)


def files(pattern):
    return collated(glob.glob(pattern))


def system_path():
    os.environ["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"


def run(*argv, **kw):
    kw.setdefault("stdin", subprocess.DEVNULL)
    try:
        return subprocess.run([str(a) for a in argv], capture_output=True, text=True, **kw)
    except OSError as error:
        return subprocess.CompletedProcess(argv, 127, "", str(error))


def out(*argv):
    return run(*argv).stdout


def here(script):
    path = script
    if os.path.islink(path):
        target = os.readlink(path)
        path = target if target.startswith("/") else os.path.join(os.path.dirname(path), target)
    return os.path.dirname(os.path.abspath(path))


def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def platform(key):
    """IOPlatformExpertDevice's KEY (IOPlatformUUID, IOPlatformSerialNumber), as `plutil -extract 0.KEY raw` gives it."""
    try:
        return str(plistlib.loads(run("ioreg", "-a", "-rd1", "-c", "IOPlatformExpertDevice").stdout.encode())[0][key])
    except (ValueError, IndexError, KeyError, TypeError):
        return ""


def fnv(uuid):
    """The node number of a host UUID: FNV-1 over its 16 bytes, 31 bits, never 0 (rdma's and the beacon's)."""
    h, n = uuid.replace("-", ""), 0
    for i in range(0, 32, 2):
        n = ((n * 0x01000193) & 0xFFFFFFFF) ^ int(h[i:i + 2] or "0", 16)
    n &= 0x7FFFFFFF
    return n or 1


def rdma():
    """Each fabric port: (port, device, rid, rdma_peer), from the IORegistry (bin/mesh-rdma.sh's lines)."""
    def dicts(cls):
        text = run("ioreg", "-a", "-r", "-c", cls, "-d", "1").stdout.encode()
        try:
            found = plistlib.loads(text) if text.strip() else []
        except Exception:
            return []
        return [d for d in (found if isinstance(found, list) else [found]) if isinstance(d, dict)]
    peers = {str(d["RID"]) for d in dicts("AppleThunderboltRDMAPeerInterface") if str(d.get("RID", ""))}
    seen, ports = set(), []
    for d in dicts("AppleThunderboltRDMAInterface"):
        rid, name = str(d.get("RID", "")), str(d.get("IORegistryEntryName", ""))
        if d.get("IORegistryEntryID") in seen:
            continue
        seen.add(d.get("IORegistryEntryID"))
        ports.append((name[5:] if name.startswith("rdma_") else name, name, rid, "present" if rid in peers else "absent"))
    return ports


def stat(directory):
    """The bridge's own reading (bin/mesh-stat beside this script): its last line, parsed; {} where it says nothing."""
    lines = run(os.path.join(directory, "mesh-stat")).stdout.strip().split("\n")
    try:
        return json.loads(lines[-1])
    except ValueError:
        return {}


def names(directory):
    """The name table (etc/mesh-nodes.json): each named node's number, its name."""
    try:
        nodes = json.load(open(os.path.join(directory, "..", "etc", "mesh-nodes.json"))).get("nodes") or {}
    except (OSError, ValueError):
        return {}
    return {fnv(k): v["name"] for k, v in nodes.items() if UUID.match(k) and isinstance(v, dict) and v.get("name")}


def raw(value):
    """A value as `plutil -extract ... raw` prints it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return "" if value is None else str(value)


def launchd(new, domain=None, plist=None):
    """bin/mesh-launchd-agent.sh: NEW put in place as PLIST and the job loaded in DOMAIN where it is not loaded; a loaded
    job runs on, its new definition taking effect at its next load; an operator's off (launchctl disable) stands
    (design/rubrics.md M3, 10-20). Returns (status, what it says)."""
    domain = domain or "user/%d" % os.getuid()
    plist = plist or (new[:-4] if new.endswith(".new") else new)
    label = os.path.basename(plist)
    label = label[:-6] if label.endswith(".plist") else label
    uid = domain.split("/", 1)[-1]

    def loaded():
        if run("launchctl", "print", "%s/%s" % (domain, label)).returncode == 0:
            return True
        if domain == "system":
            return False
        argv = ["launchctl", "list", label] if str(os.getuid()) == uid else ["launchctl", "asuser", uid, "launchctl", "list", label]
        return run(*argv).returncode == 0
    if run("plutil", "-lint", "-s", new).returncode:
        if new != plist and os.path.exists(new):
            os.unlink(new)
        return 1, "%s: its new definition does not parse; the one in place kept" % label
    changed = False
    if new != plist:
        changed = not (os.path.exists(plist) and open(new, "rb").read() == open(plist, "rb").read())
        os.replace(new, plist)
    if loaded():
        return 0, ("%s: its new definition is in place and takes effect at its next load (login or boot); the loaded job runs on"
                   % label if changed else "")
    if re.search(r'"%s" => (disabled|true)' % re.escape(label), run("launchctl", "print-disabled", domain).stdout):
        return 0, ""
    said = run("launchctl", "bootstrap", domain, plist)
    if said.returncode == 0:
        return 0, "%s loaded in %s" % (label, domain)
    return 1, "%s did not load in %s (launchctl bootstrap: %s); its definition is in place at %s" % (
        label, domain, (said.stdout + said.stderr).rstrip("\n"), plist)


def sha1(data):
    """`shasum` of DATA as the shell kept it: the digest and its file name, standard input."""
    return hashlib.sha1(data).hexdigest() + "  -"
