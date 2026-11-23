#!/usr/bin/env python3
"""Kuroshio logo: a pixel-art dark current. This file is the source of truth.

Edit GRID (one character per pixel, legend in PALETTE) and regenerate with

    python3 -I assets/make_logo.py

which rewrites, next to this script:

    logo.svg     crisp vector, one <rect> per horizontal run of pixels
    logo.png     512x512, each grid pixel scaled x16 (nearest neighbour)
    logo-64.png  64x64,   each grid pixel scaled x2

Standard library only.
"""
import os
import struct
import zlib

PALETTE = {
    ".": None,       # transparent, only outside the rounded tile
    "n": "#151c3d",  # navy: the banks and the tile itself
    "b": "#27376b",  # ink blue: the lit edge of the near bank
    "d": "#0f1329",  # deep navy: shallower water along the far bank
    "k": "#090a19",  # near-black indigo: the deepest water
    "t": "#3f7c8e",  # muted teal: flow lines
    "p": "#cfe6e4",  # pale: the glint, use sparingly
}

GRID = [
    "....nnnnnnnnnnnnnnnnnnnnnnnn....",
    "..nnnnnnnnnnnnnbkkkkkkkkkkddnn..",
    ".nnnnnnnnnnnnnnnbkkkkkkkkkkddnn.",
    ".nnnnnnnnnnnnnnnbkkkkkkkktkddnn.",
    "nnnnnnnnnnnnnnnnnbkkkkkkktkddnnn",
    "nnnnnnnnnnnnnnnnnbkkkkkkktkddnnn",
    "nnnnnnnnnnnnnnnnnbkkkkkkktkddnnn",
    "nnnnnnnnnnnnnnnnbkkkkkkkktkddnnn",
    "nnnnnnnnnnnnnnnnbkkkkkkkktkddnnn",
    "nnnnnnnnnnnnnnnbkkkkkkkktkkddnnn",
    "nnnnnnnnnnnnnnbkkkkkkkkkpkkddnnn",
    "nnnnnnnnnnnnnbkkkkkkkkkpkkddnnnn",
    "nnnnnnnnnnnnbkkkkkkkkkktkkddnnnn",
    "nnnnnnnnnnbbkkkkkkkkkktkkddnnnnn",
    "nnnnnnnnnbkkkkkkkkkkktkkddnnnnnn",
    "nnnnnnnnbkkkktkkkkkktkkddnnnnnnn",
    "nnnnnnnbkkkktkkkkkktkkddnnnnnnnn",
    "nnnnnnbkkkktkkkkkkkkkddnnnnnnnnn",
    "nnnnnbkkkktkkkkkkkkddnnnnnnnnnnn",
    "nnnnbkkkktkkkkkkkkddnnnnnnnnnnnn",
    "nnnnbkkkktkkkkkkkddnnnnnnnnnnnnn",
    "nnnbkkkktkkkkkkkddnnnnnnnnnnnnnn",
    "nnnbkkkktkkkkkkddnnnnnnnnnnnnnnn",
    "nnnbkkktkkkkkkddnnnnnnnnnnnnnnnn",
    "nnnbkkktkkkkkkddnnnnnnnnnnnnnnnn",
    "nnnbkkktkkkkkddnnnnnnnnnnnnnnnnn",
    "nnnbkkktkkkkkddnnnnnnnnnnnnnnnnn",
    "nnnbkkktkkkkkddnnnnnnnnnnnnnnnnn",
    ".nnbkkktkkkkkkddnnnnnnnnnnnnnnn.",
    ".nnbkkkkkkkkkkddnnnnnnnnnnnnnnn.",
    "..nnkkkkkkkkkkkddnnnnnnnnnnnnn..",
    "....nnnnnnnnnnnnnnnnnnnnnnnn....",
]

HERE = os.path.dirname(os.path.abspath(__file__))


def check(grid):
    width = len(grid[0])
    for y, row in enumerate(grid):
        if len(row) != width:
            raise SystemExit(f"row {y} is {len(row)} wide, expected {width}")
        for ch in row:
            if ch not in PALETTE:
                raise SystemExit(f"row {y}: unknown palette key {ch!r}")
    return width, len(grid)


def rgba(key):
    colour = PALETTE[key]
    if colour is None:
        return (0, 0, 0, 0)
    return (int(colour[1:3], 16), int(colour[3:5], 16), int(colour[5:7], 16), 255)


def png_chunk(tag, data):
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data)))


def write_png(path, grid, scale):
    """8-bit RGBA PNG, every grid pixel becomes a scale x scale block."""
    width, height = check(grid)
    raw = bytearray()
    for row in grid:
        line = b"\x00" + b"".join(bytes(rgba(ch)) * scale for ch in row)
        raw += line * scale
    header = struct.pack(">IIBBBBB", width * scale, height * scale, 8, 6, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n"
                + png_chunk(b"IHDR", header)
                + png_chunk(b"IDAT", zlib.compress(bytes(raw), 9))
                + png_chunk(b"IEND", b""))


def write_svg(path, grid, size=512):
    """One <rect> per horizontal run of same-coloured pixels."""
    width, height = check(grid)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{size}" height="{size}" shape-rendering="crispEdges">',
        "<title>Kuroshio</title>",
    ]
    for y, row in enumerate(grid):
        x = 0
        while x < width:
            end = x
            while end < width and row[end] == row[x]:
                end += 1
            if PALETTE[row[x]] is not None:
                out.append(f'<rect x="{x}" y="{y}" width="{end - x}" height="1" '
                           f'fill="{PALETTE[row[x]]}"/>')
            x = end
    out.append("</svg>")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + "\n")


def main():
    width, height = check(GRID)
    write_svg(os.path.join(HERE, "logo.svg"), GRID)
    write_png(os.path.join(HERE, "logo.png"), GRID, 512 // width)
    write_png(os.path.join(HERE, "logo-64.png"), GRID, 64 // width)
    used = sorted({ch for row in GRID for ch in row if PALETTE[ch] is not None})
    print(f"{width}x{height} grid, {len(used)} colours ({''.join(used)}) -> "
          "logo.svg, logo.png, logo-64.png")


if __name__ == "__main__":
    main()
