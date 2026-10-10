#!/usr/bin/env python3
import json
import re
import tomllib
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


def mise_tools(relative_path: str) -> dict:
    path = (MANIFEST.parent / relative_path).resolve()
    with path.open("rb") as handle:
        return tomllib.load(handle)["tools"]


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


SHA256_HEX = re.compile(r"[0-9a-f]{64}")
ZERO_SHA256 = "0" * 64


def require_sha256(name: str, value: object) -> None:
    require(isinstance(value, str), f"{name} must be a string or null")
    require(value != ZERO_SHA256, f"{name} is the all-zero placeholder; use null until the artifact is published")
    require(SHA256_HEX.fullmatch(value) is not None, f"{name} must be 64 lowercase hex characters")


def require_digest(name: str, digest: object) -> None:
    require(isinstance(digest, str) and digest.startswith("sha256:"), f"{name} must be a sha256: digest")
    require_sha256(name, digest.removeprefix("sha256:"))


def main() -> None:
    manifest = read_json(MANIFEST)
    lock = read_json(LOCK)

    require(manifest.get("schema") == 1, "manifest schema must be 1")
    require(lock.get("schema") == 1, "lock schema must be 1")
    require(manifest["base"]["linux"]["platforms"] == ["linux/amd64", "linux/arm64"], "linux OCI platforms changed unexpectedly")

    tools = mise_tools(manifest["toolchains"]["mise"])
    for name in ("go", "rust", "node", "bun", "python"):
        require(name in tools, f".mise.toml is missing {name}")
    for name, entry in tools.items():
        version = entry.get("version") if isinstance(entry, dict) else entry
        require(isinstance(version, str) and bool(version), f".mise.toml {name} must have exactly one version")
        require_pinned(f".mise.toml {name}", version)

    require("native_tools" not in manifest and "engines" not in manifest,
            "native tools and engines must be classified as deferred")
    deferred = manifest.get("deferred")
    require(isinstance(deferred, dict), "manifest must classify deferred tools and engines")
    require(deferred.get("status") == "not-installed", "deferred status must be not-installed")
    reason = deferred.get("reason")
    require(isinstance(reason, str) and bool(reason.strip()), "deferred reason must be nonempty")
    for category in ("native_tools", "engines"):
        entries = deferred.get(category)
        require(isinstance(entries, dict) and bool(entries), f"deferred {category} must be nonempty")
        for name, entry in entries.items():
            require(isinstance(entry, dict), f"deferred {name} must describe its pin")
            if "version" in entry:
                version = entry["version"]
                require(isinstance(version, str), f"deferred {name} version must be a string")
                require_pinned(name, version)
            else:
                relative_path = entry.get("version_file")
                require(category == "engines" and isinstance(relative_path, str),
                        f"deferred {name} must have a version or engine release file")
                require(relative_path in manifest["release_files"].values(),
                        f"deferred {name} must use a declared release file")
                require(bool(release_value(relative_path)), f"deferred {name} release file is empty")

    for key, relative_path in manifest["release_files"].items():
        value = release_value(relative_path)
        require(value, f"{key} release file is empty")

    linux = lock["linux_oci_manifest_list"]
    locked_platforms = [entry["platform"] for entry in linux["platforms"]]
    expected_platforms = manifest["base"]["linux"]["platforms"]
    duplicates = sorted({name for name in locked_platforms if locked_platforms.count(name) > 1})
    require(not duplicates, f"lock platforms are duplicated: {', '.join(duplicates)}")
    missing = sorted(set(expected_platforms) - set(locked_platforms))
    unexpected = sorted(set(locked_platforms) - set(expected_platforms))
    require(
        not missing and not unexpected,
        "lock platforms must match manifest platforms "
        f"(missing: {', '.join(missing) or 'none'}; unexpected: {', '.join(unexpected) or 'none'})",
    )

    unpublished = []
    recorded = [linux["tag"], linux["digest"], *(entry["digest"] for entry in linux["platforms"])]
    if all(value is None for value in recorded):
        unpublished.append("linux OCI manifest list")
    else:
        require(
            all(value is not None for value in recorded),
            "linux OCI manifest list tag and digests must all be recorded, or all be null before publication",
        )
        require(isinstance(linux["tag"], str) and bool(linux["tag"].strip()), "linux OCI manifest list tag must be nonempty")
        require_digest("linux manifest list digest", linux["digest"])
        for platform in linux["platforms"]:
            require_digest(f"{platform['platform']} digest", platform["digest"])

    native = {(entry["os"], entry["arch"]) for entry in lock["native_bundles"]}
    require(("macos", "arm64") in native, "lock must include the macOS arm64 bundle")
    require(("windows", "x64") in native, "lock must include the Windows x64 bundle")
    require(("windows", "arm64") in native, "lock must include the Windows arm64 bundle")
    for entry in lock["native_bundles"]:
        bundle = f"{entry['os']}/{entry['arch']} bundle"
        if entry["sha256"] is None:
            unpublished.append(bundle)
        else:
            require_sha256(f"{bundle} sha256", entry["sha256"])

    if unpublished:
        print(f"e2e environment manifest is valid (unpublished: {', '.join(unpublished)})")
    else:
        print("e2e environment manifest is valid")


if __name__ == "__main__":
    main()
