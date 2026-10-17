import json
import pathlib
import sys
import tempfile
import zipfile
from contextlib import redirect_stdout

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
        except zipfile.BadZipFile as error:
            print(json.dumps({"event":"archive_read_error","archive":str(archive),"error":f"{type(error).__name__}: {error}"}), file=sys.stderr)
    return files

def read(files, name, limit=None):
    archive, info = files[name]
    with zipfile.ZipFile(archive) as zf, zf.open(info) as src:
        return src.read() if limit is None else src.read(limit)

def stage(files, names, out, teams, carts):
    root = pathlib.Path(out)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mesh-training-maps-") as temporary:
        workspace = pathlib.Path(temporary)
        for path in files:
            if path.endswith((".waypoints", ".waypoints.cache")):
                (workspace / pathlib.PurePosixPath(path).name).write_bytes(read(files, path))
        for name in names:
            key = f"maps/{name}.bsp"
            bsp = workspace / f"{name}.bsp"
            bsp.write_bytes(read(files, key))
            archive = str(files[key][0])
            with redirect_stdout(sys.stderr):
                mkentfile.emit(str(bsp), str(root / f"{name}.ent"), teams, carts, archive)

def main():
    files = mounted(sys.argv[1])
    names = sorted(pathlib.PurePosixPath(path).stem for path in files
                   if path.endswith(".bsp") and "/" not in path[5:])
    if len(sys.argv) > 2:
        requested = sys.argv[5:]
        if requested:
            names = [name for name in names if name in set(requested)]
        stage(files, names, sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    print(" ".join(names))

main()
