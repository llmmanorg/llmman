#!/usr/bin/env python3
"""Makes llmman-desktop's icons from the manatee mark build.rs pins (needs Pillow and curl).

    python3 desktop/generate-assets.py [OUT_DIR]      # default ./assets-out

The PNGs are not kept in git. Upload them to the docs-assets release, then copy
the printed sha256s into the pins in desktop/build.rs (tray-*),
packaging/desktop-dmg.sh (icon-512) and packaging/desktop-msix.ps1 (logos).
"""

import hashlib
import pathlib
import subprocess
import sys
import tempfile

from PIL import Image

URL = "https://github.com/llmmanorg/llmman/releases/download/docs-assets/llmman-mark.png"
SHA256 = "c6036711a349dc8f8b2b4405be90b342b477dbcf4a7d95eb7d05b42702a241d5"

OUTPUTS = {
    "llmman-desktop-tray-32.png": 32,  # in the binary
    "llmman-desktop-tray-64.png": 64,  # in the binary
    "llmman-desktop-icon-512.png": 512,  # the .icns source
    "llmman-desktop-Square44x44Logo.png": 44,  # MSIX logos
    "llmman-desktop-Square150x150Logo.png": 150,
    "llmman-desktop-StoreLogo.png": 50,
}


def main() -> None:
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "assets-out")
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        mark = pathlib.Path(tmp) / "mark.png"
        subprocess.run(["curl", "-fsSL", "--retry", "3", "-o", mark, URL], check=True)
        if hashlib.sha256(mark.read_bytes()).hexdigest() != SHA256:
            raise SystemExit("llmman-mark.png does not match the pinned sha256")
        src = Image.open(mark).convert("RGBA")

    side = max(src.size)
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    square.paste(src, ((side - src.width) // 2, (side - src.height) // 2))
    for name, size in OUTPUTS.items():
        img = square.resize((size, size), Image.LANCZOS)
        if size > 100:  # 512 px: 176 KiB as RGBA, 34 KiB as a palette image
            img = img.quantize(256, method=Image.Quantize.FASTOCTREE)
        img.save(out / name, optimize=True)
        print(f"{hashlib.sha256((out / name).read_bytes()).hexdigest()}  {name}")

    files = " ".join(str(out / name) for name in OUTPUTS)
    print(f"\ngh release upload docs-assets {files} --repo llmmanorg/llmman --clobber")


main()
