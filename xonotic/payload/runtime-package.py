#!/usr/bin/env mesh-python
import argparse
import os
import pathlib
import tempfile
import zipfile

def info(name):
    item = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
    item.compress_type = zipfile.ZIP_DEFLATED
    item.external_attr = 0o100644 << 16
    return item

def build(source, config, output):
    files = {
        "progs.dat": source / "progs.dat",
        "csprogs.dat": source / "csprogs.dat",
        "menu.dat": source / "menu.dat",
        "effectinfo.txt": source / "effectinfo.txt",
        "gamemodes-payload.cfg": config,
        "runtime/BUILD_MANIFEST": source / "BUILD_MANIFEST",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=output.name + ".", delete=False) as handle:
        temporary = pathlib.Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w") as package:
            for name, path in files.items():
                package.writestr(info(name), path.read_bytes())
        os.chmod(temporary, 0o644)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"runtime package={output} files={len(files)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("config", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    build(args.source, args.config, args.output)

if __name__ == "__main__":
    main()
