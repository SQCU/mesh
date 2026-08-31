import pathlib
import struct
import sys
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "payload" / "tools"))
import mkentfile


def mounted(root):
    files = {}
    for archive in sorted(pathlib.Path(root).glob("*.pk3")):
        try:
            with zipfile.ZipFile(archive) as zf:
                for info in zf.infolist():
                    if info.filename.startswith("maps/"):
                        files[info.filename] = (archive, info)
        except zipfile.BadZipFile:
            pass
    return files


def read(files, name, limit=None):
    archive, info = files[name]
    with zipfile.ZipFile(archive) as zf, zf.open(info) as src:
        return src.read() if limit is None else src.read(limit)


def bsp_valid(files, name):
    key = f"maps/{name}.bsp"
    if key not in files:
        return False
    head = read(files, key, 144)
    size = files[key][1].file_size
    if len(head) < 144 or head[:4] != b"IBSP":
        return False
    return all(offset >= 0 and length >= 0 and offset + length <= size
               for offset, length in struct.iter_unpack("<ii", head[8:144]))


def graph(files, name):
    key = f"maps/{name}.waypoints.cache"
    if key not in files:
        return [], []
    text = read(files, key).decode("latin-1").strip()
    if "\n" not in text and "*" not in text and text.endswith(".waypoints.cache"):
        key = f"maps/{text}"
        if key not in files:
            return [], []
        text = read(files, key).decode("latin-1")
    return mkentfile.parse_cache(text)


def graph_valid(files, name):
    nodes, edges = graph(files, name)
    if not nodes:
        return False
    seen = set()
    stack = [0]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(set(edges[node]) - seen)
    return len(seen) >= 2


def stage(files, names, out):
    root = pathlib.Path(out)
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        data = read(files, f"maps/{name}.bsp")
        offset, length = struct.unpack_from("<ii", data, 8)
        entities = data[offset:offset + length].split(b"\0")[0].decode("latin-1").rstrip()
        nodes, edges = graph(files, name)
        spawns = []
        for index, point in enumerate(nodes):
            if not edges[index]:
                continue
            origin = " ".join(f"{value:.1f}" for value in point)
            spawns.append("\n".join(("{", '"classname" "info_player_deathmatch"',
                                      f'"origin" "{origin}"', "}")))
        (root / f"{name}.ent").write_text(entities + "\n" + "\n".join(spawns) + "\n",
                                          encoding="latin-1")


def main():
    files = mounted(sys.argv[1])
    names = sorted(pathlib.PurePosixPath(path).stem for path in files
                   if path.endswith(".mapinfo") and "/" not in path[5:])
    selected = [name for name in names if bsp_valid(files, name) and graph_valid(files, name)]
    if len(sys.argv) > 2:
        stage(files, selected, sys.argv[2])
    print(" ".join(selected))


main()
