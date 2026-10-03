#!/usr/bin/env python3
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "ci/e2e/environment/manifest.json"
LOCK = ROOT / "ci/e2e/environment/lock.json"


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def release_value(relative_path: str) -> str:
    path = (MANIFEST.parent / relative_path).resolve()
    return path.read_text(encoding="utf-8").strip()


def mise_tools(relative_path: str) -> dict[str, str]:
    path = (MANIFEST.parent / relative_path).resolve()
    tools: dict[str, str] = {}
    in_tools = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            in_tools = line == "[tools]"
            continue
        if not in_tools or "=" not in line:
            continue
        key, value = line.split("=", 1)
        tools[key.strip()] = value.strip()
    return tools


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def require_pinned(name: str, version: str) -> None:
    floating = {"latest", "stable", "unversioned", "pending-versioned-release"}
    require(version not in floating, f"{name} must be pinned, not {version}")
    require(not version.startswith("["), f"{name} must have exactly one version")
    require(
        bool(re.match(r"^v?\d+(\.\d+){1,3}([-.][0-9A-Za-z]+)*$", version)),
        f"{name} must be an exact version, got {version}",
    )


def require_digest(name: str, digest: str) -> None:
    require(
        bool(re.match(r"^sha256:[0-9a-f]{64}$", digest)),
        f"{name} must be a sha256 digest",
    )


def main() -> None:
    manifest = read_json(MANIFEST)
    lock = read_json(LOCK)

    require(manifest.get("schema") == 1, "manifest schema must be 1")
    require(lock.get("schema") == 1, "lock schema must be 1")
    require(manifest["base"]["linux"]["platforms"] == ["linux/amd64", "linux/arm64"], "linux OCI platforms changed unexpectedly")

    tools = mise_tools(manifest["toolchains"]["mise"])
    for name in ("go", "rust", "node", "bun", "python"):
        require(name in tools, f".mise.toml is missing {name}")
        require(tools[name], f".mise.toml has an empty {name} version")
        require_pinned(f".mise.toml {name}", tools[name].strip('"'))

    npm_bins = [entry["binary"] for entry in manifest["npm"]]
    require(len(npm_bins) == len(set(npm_bins)), "npm binary names must be unique")
    for entry in manifest["npm"]:
        require(entry["version"], f"{entry['name']} is missing a version")
        require_pinned(entry["name"], entry["version"])
        require(entry["installer"] in {"npm", "bun"}, f"{entry['name']} has an unsupported installer")

    for name, entry in manifest["native_tools"].items():
        require_pinned(name, entry["version"])

    for name, entry in manifest["engines"].items():
        if "version" in entry:
            require_pinned(name, entry["version"])

    for key, relative_path in manifest["release_files"].items():
        value = release_value(relative_path)
        require(value, f"{key} release file is empty")

    locked_platforms = [entry["platform"] for entry in lock["linux_oci_manifest_list"]["platforms"]]
    require(locked_platforms == manifest["base"]["linux"]["platforms"], "lock platforms must match manifest platforms")
    require_digest("linux manifest list digest", lock["linux_oci_manifest_list"]["digest"])
    for platform in lock["linux_oci_manifest_list"]["platforms"]:
        require_digest(f"{platform['platform']} digest", platform["digest"])

    native = {(entry["os"], entry["arch"]) for entry in lock["native_bundles"]}
    require(("macos", "arm64") in native, "lock must include the macOS arm64 bundle")
    require(("windows", "x64") in native, "lock must include the Windows x64 bundle")
    require(("windows", "arm64") in native, "lock must include the Windows arm64 bundle")

    print("e2e environment manifest is valid")


if __name__ == "__main__":
    main()
