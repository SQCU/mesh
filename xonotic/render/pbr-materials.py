#!/usr/bin/env mesh-python
import argparse
import binascii
import os
import pathlib
import re
import struct
import sys
import tempfile
import zipfile
import zlib

IMAGE = re.compile(r"\.(?:dds|jpg|jpeg|png|tga)$", re.I)
AUX = re.compile(r"_(?:bump|gloss|glow|norm|normal|pants|pbr|reflect|shirt)$", re.I)
RULES = (
    (("sky", "cloud", "fog"), 1.00, 0.00, "atmosphere"),
    (("water", "liquid", "slime", "lava"), 0.08, 0.00, "liquid"),
    (("glass", "window", "mirror"), 0.12, 0.00, "glass"),
    (("chrome", "gold", "silver", "copper", "brass"), 0.18, 1.00, "polished-metal"),
    (("metal", "steel", "iron", "alum", "grate", "pipe"), 0.34, 1.00, "metal"),
    (("plastic", "rubber", "trim", "panel", "tech"), 0.48, 0.00, "manufactured"),
    (("wood", "board", "crate"), 0.72, 0.00, "wood"),
    (("rock", "stone", "concrete", "brick", "sand", "dirt", "terrain"), 0.88, 0.00, "mineral"),
)

def chunk(kind, payload):
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)

def png(red, green, blue):
    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    pixels = zlib.compress(bytes((0, red, green, blue)), 9)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", pixels) + chunk(b"IEND", b"")

def material(name):
    lower = name.lower()
    for words, roughness, metallic, family in RULES:
        if any(word in lower for word in words):
            return roughness, metallic, family
    return 0.62, 0.00, "dielectric"

def base_name(name):
    name = name.replace("\\", "/")
    if name.lower().startswith("dds/"):
        name = name[4:]
    if not name.lower().startswith("textures/") or not IMAGE.search(name):
        return None, False
    stem = IMAGE.sub("", name)
    gloss = stem.lower().endswith("_gloss")
    return AUX.sub("", stem), gloss

def zip_info(name):
    info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info

def build(assetroot, output):
    materials = {}
    gloss = set()
    output_identity = output.resolve()
    archives = [
        archive for archive in sorted((assetroot / "data").glob("*.pk3"))
        if archive.resolve() != output_identity
    ]
    failures = []
    for archive in archives:
        try:
            with zipfile.ZipFile(archive) as package:
                for entry in package.namelist():
                    base, is_gloss = base_name(entry)
                    if base:
                        materials[base.lower()] = base
                        if is_gloss:
                            gloss.add(base.lower())
        except (OSError, zipfile.BadZipFile) as exc:
            failures.append(f"{archive}: {exc}")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = ["material\troughness\tmetallic\tfamily\tgloss"]
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=output.name + ".", delete=False) as handle:
        temporary = pathlib.Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w") as package:
            for key in sorted(materials):
                base = materials[key]
                roughness, metallic, family = material(base)
                pbr = png(255, round(roughness * 255), round(metallic * 255))
                package.writestr(zip_info(f"{base}_pbr.png"), pbr)
                source = "stock" if key in gloss else "derived-default"
                if key not in gloss:
                    package.writestr(zip_info(f"{base}_gloss.png"), png(13, 13, 13))
                rows.append(f"{base}\t{roughness:.2f}\t{metallic:.2f}\t{family}\t{source}")
            package.writestr(zip_info("materials/mesh-pbr.tsv"), ("\n".join(rows) + "\n").encode())
            package.writestr(zip_info("materials/mesh-pbr-sources.txt"), ("\n".join(path.name for path in archives) + "\n").encode())
            package.writestr(zip_info("materials/mesh-pbr-read-failures.txt"), ("\n".join(failures) + ("\n" if failures else "")).encode())
        os.chmod(temporary, 0o644)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    for failure in failures:
        print(f"pbr asset read failure: {failure}", file=sys.stderr)
    print(f"pbr materials={len(materials)} stock_gloss={len(gloss)} derived_gloss={len(materials) - len(gloss)} output={output}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("assetroot", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    build(args.assetroot, args.output)

if __name__ == "__main__":
    main()
