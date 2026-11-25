#!/usr/bin/env mesh-python
import argparse
import binascii
import io
import os
import pathlib
import re
import struct
import sys
import tempfile
import zipfile
import zlib

try:
    from PIL import Image
except ImportError:
    Image = None

IMAGE = re.compile(r"\.(?:dds|jpg|jpeg|png|tga)$", re.I)
DECODABLE = re.compile(r"\.(?:jpg|jpeg|png|tga)$", re.I)
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

def png_rgb_grid(width, height, rgb_bytes):
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    stride = width * 3
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        rows.extend(rgb_bytes[y * stride:(y + 1) * stride])
    pixels = zlib.compress(bytes(rows), 6)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", pixels) + chunk(b"IEND", b"")

def _decode_tga_rgb(raw):
    if len(raw) < 18:
        return None
    id_len, cmap_type, img_type = raw[0], raw[1], raw[2]
    if cmap_type != 0 or img_type != 2:
        return None
    w, h, bpp, desc = struct.unpack_from("<HHBB", raw, 12)
    if w <= 0 or h <= 0 or bpp not in (24, 32):
        return None
    stride = bpp // 8
    off = 18 + id_len
    if off + w * h * stride > len(raw):
        return None
    top_origin = bool(desc & 0x20)
    rgb = bytearray(w * h * 3)
    for y in range(h):
        sy = y if top_origin else (h - 1 - y)
        row_off = off + sy * w * stride
        dst_off = y * w * 3
        for x in range(w):
            p = row_off + x * stride
            rgb[dst_off + 3 * x] = raw[p + 2]
            rgb[dst_off + 3 * x + 1] = raw[p + 1]
            rgb[dst_off + 3 * x + 2] = raw[p]
    return w, h, bytes(rgb)

def _decode_png_rgb(raw):
    if len(raw) < 33 or raw[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    pos, w, h, bit_depth, color_type = 8, 0, 0, 0, 0
    idat = bytearray()
    while pos + 8 <= len(raw):
        ln, kind = struct.unpack_from(">I4s", raw, pos)
        data = raw[pos + 8:pos + 8 + ln]
        pos += 12 + ln
        if kind == b"IHDR" and ln >= 13:
            w, h, bit_depth, color_type, comp, flt, interlace = struct.unpack_from(">IIBBBBB", data, 0)
            if bit_depth != 8 or comp != 0 or flt != 0 or interlace != 0 or color_type not in (0, 2, 6):
                return None
        elif kind == b"IDAT":
            idat.extend(data)
        elif kind == b"IEND":
            break
    if w <= 0 or h <= 0 or not idat:
        return None
    bpp = {0: 1, 2: 3, 6: 4}[color_type]
    stride = w * bpp
    try:
        plain = zlib.decompress(bytes(idat))
    except zlib.error:
        return None
    if len(plain) < h * (stride + 1):
        return None
    prev = bytearray(stride)
    rgb = bytearray(w * h * 3)
    for y in range(h):
        base = y * (stride + 1)
        f = plain[base]
        cur = bytearray(plain[base + 1:base + 1 + stride])
        for i in range(stride):
            a = cur[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            if f == 1:
                cur[i] = (cur[i] + a) & 0xFF
            elif f == 2:
                cur[i] = (cur[i] + b) & 0xFF
            elif f == 3:
                cur[i] = (cur[i] + ((a + b) >> 1)) & 0xFF
            elif f == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                cur[i] = (cur[i] + pr) & 0xFF
        prev = cur
        dst = y * w * 3
        if bpp == 3:
            rgb[dst:dst + w * 3] = cur
        elif bpp == 4:
            for x in range(w):
                rgb[dst + 3 * x:dst + 3 * x + 3] = cur[4 * x:4 * x + 3]
        else:
            for x in range(w):
                v = cur[x]
                rgb[dst + 3 * x] = rgb[dst + 3 * x + 1] = rgb[dst + 3 * x + 2] = v
    return w, h, bytes(rgb)

def _decode_image_rgb(raw_bytes, max_dim=64):
    if Image is not None:
        try:
            img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            w, h = img.size
            if w > max_dim or h > max_dim:
                scale = max_dim / max(w, h)
                w, h = max(4, int(round(w * scale))), max(4, int(round(h * scale)))
                img = img.resize((w, h), Image.BILINEAR)
            return w, h, img.tobytes()
        except Exception:
            pass
    decoded = _decode_png_rgb(raw_bytes) or _decode_tga_rgb(raw_bytes)
    if decoded is None:
        return None
    w, h, data = decoded
    if w <= max_dim and h <= max_dim:
        return w, h, data
    scale = max_dim / max(w, h)
    nw, nh = max(4, int(round(w * scale))), max(4, int(round(h * scale)))
    out = bytearray(nw * nh * 3)
    for y in range(nh):
        sy = min(h - 1, int(y * h / nh))
        for x in range(nw):
            sx = min(w - 1, int(x * w / nw))
            src = (sy * w + sx) * 3
            dst = (y * nw + x) * 3
            out[dst:dst + 3] = data[src:src + 3]
    return nw, nh, bytes(out)

def derive_spatial_maps(raw_bytes, base_rough, base_metal, max_dim=64):
    if not raw_bytes:
        return None
    decoded = _decode_image_rgb(raw_bytes, max_dim=max_dim)
    if decoded is None:
        return None
    w, h, data = decoded
    lum = [0.0] * (w * h)
    sat = [0.0] * (w * h)
    for i in range(w * h):
        r, g, b = data[3 * i] / 255.0, data[3 * i + 1] / 255.0, data[3 * i + 2] / 255.0
        lum[i] = 0.2126 * r + 0.7152 * g + 0.0722 * b
        mx, mn = max(r, g, b), min(r, g, b)
        sat[i] = (mx - mn) / mx if mx > 1e-4 else 0.0
    pbr_rgb = bytearray(w * h * 3)
    gloss_rgb = bytearray(w * h * 3)
    for y in range(h):
        ym, yp = max(0, y - 1), min(h - 1, y + 1)
        for x in range(w):
            xm, xp = max(0, x - 1), min(w - 1, x + 1)
            idx = y * w + x
            l = lum[idx]
            avg = 0.25 * (lum[y * w + xm] + lum[y * w + xp] + lum[ym * w + x] + lum[yp * w + x])
            detail = min(1.0, abs(l - avg) * 4.0)
            ao = max(0.35, min(1.0, 0.72 + 0.28 * l - 0.15 * detail))
            rough = max(0.05, min(0.98, base_rough + 0.22 * detail - 0.14 * (l - 0.5)))
            metal = max(0.0, min(1.0, base_metal * (1.0 - 0.45 * sat[idx]) * (0.75 + 0.25 * l)))
            pbr_rgb[3 * idx] = round(ao * 255)
            pbr_rgb[3 * idx + 1] = round(rough * 255)
            pbr_rgb[3 * idx + 2] = round(metal * 255)
            gval = round(max(0.04, min(0.95, 1.0 - rough)) * 255)
            gloss_rgb[3 * idx] = gloss_rgb[3 * idx + 1] = gloss_rgb[3 * idx + 2] = gval
    return png_rgb_grid(w, h, pbr_rgb), png_rgb_grid(w, h, gloss_rgb)

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
    diffuse_src = {}
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
                        key = base.lower()
                        materials[key] = base
                        if is_gloss:
                            gloss.add(key)
                        elif DECODABLE.search(entry) and not AUX.search(IMAGE.sub("", entry)):
                            diffuse_src[key] = (archive, entry)
        except (OSError, zipfile.BadZipFile) as exc:
            failures.append(f"{archive}: {exc}")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = ["material\troughness\tmetallic\tfamily\tgloss\tsource"]
    spatial_count = 0
    open_zips = {}
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=output.name + ".", delete=False) as handle:
        temporary = pathlib.Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w") as package:
            for key in sorted(materials):
                base = materials[key]
                roughness, metallic, family = material(base)
                spatial = None
                if key in diffuse_src:
                    arch, entry = diffuse_src[key]
                    try:
                        zf = open_zips.get(arch)
                        if zf is None:
                            zf = open_zips[arch] = zipfile.ZipFile(arch)
                        spatial = derive_spatial_maps(zf.read(entry), roughness, metallic)
                    except Exception:
                        spatial = None
                if spatial is not None:
                    pbr, gloss_png = spatial
                    spatial_count += 1
                    map_src = "spatial-albedo"
                else:
                    pbr = png(255, round(roughness * 255), round(metallic * 255))
                    gloss_png = png(13, 13, 13)
                    map_src = "rule-fallback"
                package.writestr(zip_info(f"{base}_pbr.png"), pbr)
                source = "stock" if key in gloss else "derived-default"
                if key not in gloss:
                    package.writestr(zip_info(f"{base}_gloss.png"), gloss_png)
                rows.append(f"{base}\t{roughness:.2f}\t{metallic:.2f}\t{family}\t{source}\t{map_src}")
            package.writestr(zip_info("materials/mesh-pbr.tsv"), ("\n".join(rows) + "\n").encode())
            package.writestr(zip_info("materials/mesh-pbr-sources.txt"), ("\n".join(path.name for path in archives) + "\n").encode())
            package.writestr(zip_info("materials/mesh-pbr-read-failures.txt"), ("\n".join(failures) + ("\n" if failures else "")).encode())
        os.chmod(temporary, 0o644)
        os.replace(temporary, output)
    finally:
        for zf in open_zips.values():
            zf.close()
        temporary.unlink(missing_ok=True)
    for failure in failures:
        print(f"pbr asset read failure: {failure}", file=sys.stderr)
    print(f"pbr materials={len(materials)} spatial={spatial_count} fallback={len(materials) - spatial_count} stock_gloss={len(gloss)} output={output}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("assetroot", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    build(args.assetroot, args.output)

if __name__ == "__main__":
    main()
