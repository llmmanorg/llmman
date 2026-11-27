#!/usr/bin/env bash
# Builds llmman-desktop's macOS .dmg (needs macOS: sips, iconutil, codesign,
# hdiutil):
#
#   packaging/desktop-dmg.sh <llmman-desktop> <llmman> <version> <out.dmg>
#
# llmman.app (both binaries, no Dock icon) plus an /Applications link,
# signed ad hoc and not notarized.

set -euo pipefail

die() {
	printf 'desktop-dmg.sh: %s\n' "$*" >&2
	exit 1
}

[ $# -eq 4 ] || die "usage: desktop-dmg.sh <llmman-desktop> <llmman> <version> <out.dmg>"
desktop=$1 llmman=$2 version=$3 out=$4
[ -f "$desktop" ] || die "no such file: $desktop"
[ -f "$llmman" ] || die "no such file: $llmman"
case "$version" in
*[!0-9.]* | '') die "version \"$version\" is not MAJOR.MINOR.PATCH" ;;
esac

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
app="$work/dmg/llmman.app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources" "$work/llmman.iconset"
install -m 755 "$desktop" "$app/Contents/MacOS/llmman-desktop"
install -m 755 "$llmman" "$app/Contents/MacOS/llmman"

# Pinned download from the docs-assets release; desktop/generate-assets.py
# makes it and prints the sha256.
icon="llmman-desktop-icon-512.png:97c10798327876bfe2937ee8c963632e5f777e39016f7fdcedceb3668acf271c"
src=$work/${icon%:*}
curl -fsSL --retry 5 -o "$src" "https://github.com/llmmanorg/llmman/releases/download/docs-assets/${icon%:*}" ||
	die "could not download ${icon%:*}"
printf '%s  %s\n' "${icon#*:}" "$src" | shasum -a 256 -c - >/dev/null ||
	die "${icon%:*} does not match the sha256 pinned in this script"

# The iconset wants 16..512 and their @2x.
for size in 16 32 128 256; do
	sips -z "$size" "$size" "$src" --out "$work/llmman.iconset/icon_${size}x${size}.png" >/dev/null
	sips -z $((size * 2)) $((size * 2)) "$src" --out "$work/llmman.iconset/icon_${size}x${size}@2x.png" >/dev/null
done
cp "$src" "$work/llmman.iconset/icon_512x512.png"
iconutil -c icns -o "$app/Contents/Resources/llmman.icns" "$work/llmman.iconset"

cat >"$app/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleExecutable</key><string>llmman-desktop</string>
	<key>CFBundleIdentifier</key><string>org.llmman.desktop</string>
	<key>CFBundleName</key><string>llmman</string>
	<key>CFBundleIconFile</key><string>llmman</string>
	<key>CFBundlePackageType</key><string>APPL</string>
	<key>CFBundleShortVersionString</key><string>$version</string>
	<key>CFBundleVersion</key><string>$version</string>
	<key>LSMinimumSystemVersion</key><string>11.0</string>
	<key>LSUIElement</key><true/>
	<key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
EOF
plutil -lint "$app/Contents/Info.plist" >/dev/null

codesign --force --deep --sign - "$app"
ln -s /Applications "$work/dmg/Applications"
rm -f "$out"
# hdiutil fails now and then on busy runners.
for try in 1 2 3; do
	hdiutil create -volname llmman -srcfolder "$work/dmg" -ov -format UDZO "$out" >/dev/null && break
	[ "$try" -lt 3 ] || die "hdiutil create failed"
	sleep 5
done
printf '%s\n' "$out"
