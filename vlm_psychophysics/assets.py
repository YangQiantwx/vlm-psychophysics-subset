"""Resolve and verify the pinned checkpoint assets without importing model runtimes."""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
import csv
import hashlib
from importlib import metadata
from http.client import HTTPException
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class AssetError(RuntimeError):
    """A checkpoint or runtime prerequisite could not be established."""


_ROOT_ENV = "VLM_PSYCHOPHYSICS_MODEL_ROOT"
_CUDA_ENV = "VLM_PSYCHOPHYSICS_CUDA_HOME"
_LOCK = Path("configs/runtime-lock.json")
_PROJECT = Path(__file__).resolve().parents[1]
_GIB = 1024**3


def _read_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def _lock(config):
    configured = Path(config.get("runtime_lock", _LOCK)).expanduser()
    path = configured if configured.is_absolute() else _PROJECT / configured
    try:
        data = _read_json(path)
    except (OSError, ValueError) as exc:
        raise AssetError(
            f"Could not read runtime lock {path}. Use the release configs directory."
        ) from exc
    if data.get("schema") != "vlm-psychophysics-runtime-lock-v1":
        raise AssetError(f"Unsupported runtime lock: {path}")
    return data


def _keys(models):
    return (models,) if isinstance(models, str) else tuple(dict.fromkeys(models))


def _existing_parent(path):
    path = path.expanduser().resolve()
    while not path.exists() and path.parent != path:
        path = path.parent
    return path


def storage_info(path):
    """Identify the physical block device behind a path, including Linux symlinks."""
    path = Path(path).expanduser().resolve()
    parent = _existing_parent(path)
    try:
        device = parent.stat().st_dev
        sysfs = Path(f"/sys/dev/block/{os.major(device)}:{os.minor(device)}")
        devices = []

        def inspect_device(entry):
            entry = entry.resolve(strict=True)
            slaves = tuple((entry / "slaves").glob("*"))
            if slaves:
                for slave in slaves:
                    inspect_device(slave)
                return
            if (entry / "partition").exists():
                entry = entry.parent
            if entry.name.startswith(("loop", "ram", "zram")):
                return
            rotational = (entry / "queue/rotational").read_text().strip()
            if rotational in {"0", "1"}:
                devices.append({"name": entry.name, "rotational": rotational == "1"})

        inspect_device(sysfs)
    except OSError:
        devices = []
    kind = (
        ("ssd" if not any(x["rotational"] for x in devices) else "rotational")
        if devices
        else "unknown"
    )
    free_bytes = shutil.disk_usage(parent).free
    return {
        "path": str(path),
        "kind": kind,
        "devices": devices,
        "free_bytes": free_bytes,
    }


def _ssd(path):
    info = storage_info(path)
    if info["kind"] != "ssd":
        raise AssetError(
            f"{info['path']} resolves to {info['kind']} storage; use a verified SSD with --model-root or {_ROOT_ENV}."
        )
    return info


def _ssd_root(config, model_root, required_bytes=0):
    explicit = model_root or os.environ.get(_ROOT_ENV) or config.get("model_root")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        info = _ssd(path)
        if info["free_bytes"] < required_bytes:
            raise AssetError(
                f"SSD {path} needs {required_bytes / _GIB:.1f} GiB available."
            )
        return path
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    candidates = [
        cache / "vlm-psychophysics" / "models",
        Path.home() / ".cache/vlm-psychophysics/models",
    ]
    # Mounted scratch disks are useful when the home disk is slow or too small.
    for mount in (
        *Path("/mnt").glob("*"),
        *Path("/scratch").glob("*"),
        *Path("/media").glob("*/*"),
    ):
        if mount.is_mount() and os.access(mount, os.W_OK):
            candidates.append(mount / f"vlm-psychophysics-{os.getuid()}" / "models")
    for candidate in dict.fromkeys(candidates):
        info = storage_info(candidate)
        if (
            info["kind"] == "ssd"
            and info["free_bytes"] >= required_bytes
            and os.access(_existing_parent(candidate), os.W_OK)
        ):
            return candidate.expanduser().resolve()
    raise AssetError(
        f"No writable SSD with {required_bytes / _GIB:.1f} GiB available; set --model-root or {_ROOT_ENV}."
    )


def _file_sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _safe_relative(name):
    p = Path(name)
    if not p.parts or p.is_absolute() or ".." in p.parts:
        raise AssetError(f"Unsafe relative asset name: {name}")
    return p


def _check_files(path, material):
    """Check metadata bytes, shard sizes and the real device of every weight."""
    path = Path(path).expanduser().resolve()
    devices = set()
    for name, entry in material["files"].items():
        p = path / _safe_relative(name)
        if not p.is_file():
            raise AssetError(f"Missing pinned asset: {p}")
        if p.stat().st_size != entry["bytes"]:
            raise AssetError(f"Pinned asset has the wrong byte size: {p}")
        if entry.get("weight"):
            st_dev = p.stat().st_dev
            if st_dev not in devices:
                _ssd(p)
                devices.add(st_dev)
        if not entry.get("weight") and _file_sha(p) != entry["sha256"]:
            raise AssetError(f"Pinned asset has the wrong SHA-256: {p}")
    unexpected_code = {p.name for p in path.glob("*.py")} - material["files"].keys()
    if unexpected_code:
        raise AssetError(
            f"Unpinned Python files in {path}: {', '.join(sorted(unexpected_code))}"
        )
    return path


def _weight_stamp(path, sha256):
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "sha256": sha256,
    }


def _weight_receipt(root, path):
    key = hashlib.sha256(os.fsencode(path.resolve())).hexdigest()
    return root / ".verified" / f"{key}.json"


def _cached_weights(root, path):
    if root is None:
        return {}
    try:
        receipt = _read_json(_weight_receipt(root, path))
        if receipt.get("schema") == 1 and receipt.get("path") == str(path.resolve()):
            return receipt["weights"]
    except (OSError, ValueError, KeyError, AttributeError):
        pass
    return {}


def _verify_weight_hashes(root, path, material, *, force=False, preverified=None):
    """Hash uncached shards and bind each result to the actual file stat."""
    weights = {
        name: entry for name, entry in material["files"].items() if entry.get("weight")
    }
    if not weights:
        return
    cached = {} if force else _cached_weights(root, path)
    if preverified:
        cached = {**cached, **preverified}

    def verify(item):
        name, entry = item
        asset = path / _safe_relative(name)
        sha = entry["sha256"]
        before = _weight_stamp(asset, sha)
        if before != cached.get(name):
            if _file_sha(asset) != sha or before != _weight_stamp(asset, sha):
                raise AssetError(
                    f"Pinned weight has the wrong SHA-256 or changed during verification: {asset}"
                )
        return name, before

    with ThreadPoolExecutor(max_workers=2) as pool:
        verified = dict(pool.map(verify, weights.items()))
    if verified == cached and not preverified:
        return
    receipt = _weight_receipt(root, path)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=receipt.parent, delete=False
    ) as f:
        temp = Path(f.name)
        json.dump({"schema": 1, "path": str(path.resolve()), "weights": verified}, f)
    try:
        temp.replace(receipt)
    finally:
        temp.unlink(missing_ok=True)


def _full_weight_proof(root, path, material):
    cached = _cached_weights(root, path)
    try:
        return all(
            _weight_stamp(path / name, entry["sha256"]) == cached.get(name)
            for name, entry in material["files"].items()
            if entry.get("weight")
        )
    except OSError:
        return False


def _target(root, material):
    return (
        root / _safe_relative(material["repo"]) / _safe_relative(material["revision"])
    )


def _hub_snapshot(material):
    if material["provider"] != "huggingface":
        return None
    xdg_cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    hf_home = Path(os.environ.get("HF_HOME", xdg_cache / "huggingface"))
    hub = Path(
        os.environ.get(
            "HF_HUB_CACHE", os.environ.get("HUGGINGFACE_HUB_CACHE", hf_home / "hub")
        )
    )
    repo = "models--" + material["repo"].replace("/", "--")
    return (
        hub.expanduser()
        / _safe_relative(repo)
        / "snapshots"
        / _safe_relative(material["revision"])
    )


def _model_plan(config, models, lock):
    selected = {}
    explicit = {}
    for model in _keys(models):
        try:
            selected[model] = lock["bindings"][model]
        except KeyError:
            raise AssetError(
                f"Unknown model: {model}. Choose from {', '.join(lock['bindings'])}."
            ) from None
        model_config = config.get("models", {}).get(model, {})
        for role, key in selected[model].items():
            value = model_config.get(role)
            if value:
                path = Path(value).expanduser().resolve()
                if key in explicit and explicit[key] != path:
                    raise AssetError(
                        f"Conflicting local paths for {key}: {explicit[key]} and {path}"
                    )
                explicit[key] = path
    return selected, explicit


def _missing_bytes(root, plan, explicit, materials, *, reuse_hub=False):
    total = 0
    considered = set()

    def add(key):
        nonlocal total
        if key in considered or key in explicit:
            return
        considered.add(key)
        material = materials[key]
        target = _target(root, material)
        snapshot = _hub_snapshot(material) if reuse_hub else None
        if snapshot and _path_ready(snapshot, material):
            return
        if material["provider"] == "composition":
            if not (target / "config.json").is_file():
                add(material["weights"])
                add(material["processor"])
            return
        for name, entry in material["files"].items():
            path = target / _safe_relative(name)
            if not path.is_file() or path.stat().st_size != entry["bytes"]:
                partial = path.with_name(path.name + ".part")
                resumable = (
                    material["provider"] == "ngc"
                    and partial.is_file()
                    and not partial.is_symlink()
                )
                total += (
                    entry["bytes"] - min(entry["bytes"], partial.stat().st_size)
                    if resumable
                    else entry["bytes"]
                )

    for bindings in plan.values():
        for key in bindings.values():
            add(key)
    return total


@contextmanager
def _acquisition_lock(root):
    import fcntl

    root.mkdir(parents=True, exist_ok=True)
    with (root / ".acquire.lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def _download_hf(root, target, material):
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError:
        raise AssetError(
            "Hugging Face downloads need huggingface-hub and hf-xet; install the release download dependencies."
        ) from None
    try:
        snapshot_download(
            repo_id=material["repo"],
            revision=material["revision"],
            allow_patterns=list(material["files"]),
            local_dir=target,
            cache_dir=root / ".hub",
            max_workers=4,
            endpoint="https://huggingface.co",
        )
    except HfHubHTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        if status in {401, 403}:
            raise AssetError(
                f"Access to {material['repo']} was denied (HTTP {status}). Sign in with 'hf auth login' and accept the repository terms if requested."
            ) from None
        raise AssetError(
            f"Download failed for {material['repo']} at {material['revision']} (HTTP {status})."
        ) from None
    except (OSError, RuntimeError, ValueError) as exc:
        raise AssetError(
            f"Download failed for {material['repo']} at {material['revision']} ({type(exc).__name__}). Check the connection, SSD space and Hugging Face offline setting."
        ) from None


class _SafeRedirect(HTTPRedirectHandler):
    """Do not forward the NGC credential to a different redirect host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme != "https":
            raise AssetError("NGC redirected to a non-HTTPS URL.")
        next_request = super().redirect_request(req, fp, code, msg, headers, newurl)
        if next_request and urlsplit(newurl).netloc != urlsplit(req.full_url).netloc:
            next_request.remove_header("Authorization")
        return next_request


def _download_ngc(target, material):
    key = os.environ.get("NGC_API_KEY") or os.environ.get("NGC_CLI_API_KEY")
    if not key:
        raise AssetError(
            "Cosmos weights are distributed through NVIDIA NGC. Set NGC_API_KEY to a personal/service key with access to the pinned model, or provide an existing checkpoint."
        )
    if urlsplit(material["url"]).netloc != "api.ngc.nvidia.com" or not material[
        "url"
    ].startswith("https://"):
        raise AssetError(
            "Refusing to send an NGC credential to an unrecognized endpoint."
        )
    target.mkdir(parents=True, exist_ok=True)

    def download(item):
        name, entry = item
        path = target / _safe_relative(name)
        if path.is_file() and path.stat().st_size == entry["bytes"]:
            stamp = _weight_stamp(path, entry["sha256"])
            if _file_sha(path) == entry["sha256"] and stamp == _weight_stamp(
                path, entry["sha256"]
            ):
                return name, stamp
        if path.exists() or path.is_symlink():
            raise AssetError(
                f"Existing NGC asset does not match the lock: {path}. Use a fresh model root."
            )
        temporary = path.with_name(path.name + ".part")
        start = temporary.stat().st_size if temporary.is_file() else 0
        if temporary.is_symlink() or start > entry["bytes"]:
            raise AssetError(f"Unexpected NGC partial file: {temporary}")
        if start == entry["bytes"] and _file_sha(temporary) == entry["sha256"]:
            temporary.replace(path)
            return name, _weight_stamp(path, entry["sha256"])
        if start == entry["bytes"]:
            start = 0
        headers = {"Authorization": "Bearer " + key}
        if start:
            headers["Range"] = f"bytes={start}-"
        url = material["url"] + quote(name, safe="/")
        try:
            opener = build_opener(_SafeRedirect())
            with opener.open(Request(url, headers=headers), timeout=90) as response:
                ranged = response.status == 206 and response.headers.get(
                    "Content-Range", ""
                ).startswith(f"bytes {start}-")
                if response.status == 206 and not ranged:
                    raise AssetError(
                        f"NGC returned an unexpected byte range for {name}."
                    )
                resuming = bool(start and ranged)
                digest = hashlib.sha256()
                received = start if resuming else 0
                if resuming:
                    with temporary.open("rb") as previous:
                        digest = hashlib.file_digest(previous, lambda: digest)
                with temporary.open("ab" if resuming else "wb") as stream:
                    while block := response.read(8 * 1024 * 1024):
                        received += len(block)
                        if received > entry["bytes"]:
                            raise AssetError(
                                f"NGC sent more bytes than the lock permits: {name}."
                            )
                        stream.write(block)
                        digest.update(block)
        except HTTPError as exc:
            if exc.code in {401, 402, 403}:
                raise AssetError(
                    f"NGC denied access to the pinned Cosmos release (HTTP {exc.code}). Check your key scope and NVIDIA account entitlement, or provide the original checkpoint."
                ) from None
            raise AssetError(
                f"NGC download failed for {name} (HTTP {exc.code})."
            ) from None
        except (URLError, OSError, HTTPException) as exc:
            raise AssetError(
                f"NGC download interrupted for {name} ({type(exc).__name__}); rerun to resume."
            ) from None
        if received != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
            raise AssetError(f"NGC file failed the pinned size/SHA-256 check: {name}.")
        temporary.replace(path)
        return name, _weight_stamp(path, entry["sha256"])

    # Three workers matches the bounded official acquisition of this release.
    with ThreadPoolExecutor(max_workers=3) as pool:
        return dict(pool.map(download, material["files"].items()))


def _compose_gemma(target, material, weights, processor):
    """Recreate the effective model config from the original processor and official NVFP4."""
    if target.exists() or target.is_symlink():
        raise AssetError(
            f"Incomplete Gemma composition exists: {target}. Use a fresh model root."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    config = _read_json(processor / "config.json")
    config["quantization_config"] = _read_json(weights / "config.json")[
        "quantization_config"
    ]
    with TemporaryDirectory(prefix=".gemma-stage-", dir=target.parent) as temporary:
        stage = Path(temporary) / target.name
        stage.mkdir()
        # Matches the frozen compact, sorted JSON serialization including its newline.
        (stage / "config.json").write_text(
            json.dumps(
                config, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
            + "\n",
            encoding="utf-8",
        )
        for name in material["files"]:
            if name == "config.json":
                continue
            src = (
                weights
                if name.startswith("model") or name == "hf_quant_config.json"
                else processor
            )
            original = (src / name).resolve()
            if name.startswith("model"):
                if original.stat().st_dev != stage.stat().st_dev:
                    raise AssetError(
                        "Gemma weights and the effective checkpoint must use the same SSD filesystem to avoid duplicating weights."
                    )
                os.link(original, stage / name)
            else:
                (stage / name).symlink_to(original)
        _check_files(stage, material)
        stage.rename(target)


def _path_ready(path, material):
    try:
        return _check_files(path, material)
    except AssetError:
        return None


def ensure_models(
    config, models, model_root=None, *, download=True, verify_weights=False
):
    """Return a resolved config; verified local assets are reused before fetching anything.

    Each weight is fully verified on first use. Later runs reuse an exact SHA
    only while the resolved path, inode, size, mtime and ctime still match.
    Pass verify_weights=True to disregard cached results and rehash shards.
    """
    lock = _lock(config)
    plan, explicit = _model_plan(config, models, lock)
    materials = lock["materials"]
    result = deepcopy(config)
    configured_root = (
        model_root or os.environ.get(_ROOT_ENV) or config.get("model_root")
    )
    root = _ssd_root(config, model_root)
    reuse_hub = not configured_root
    if download and reuse_hub:
        required = _missing_bytes(root, plan, explicit, materials, reuse_hub=True)
        root = _ssd_root(config, model_root, required)
    resolved = {}
    pending = set()

    def resolve(key, online=False):
        if key in resolved:
            return resolved[key]
        material = materials[key]
        preverified = None
        if key in explicit:
            path = _check_files(explicit[key], material)
        else:
            target = _target(root, material)
            path = _path_ready(target, material)
            if path is None and reuse_hub and (snapshot := _hub_snapshot(material)):
                path = _path_ready(snapshot, material)
            if path is None:
                if not online:
                    pending.add(key)
                    return None
                _ssd(target)
                provider = material["provider"]
                if provider == "composition":
                    weights = resolve(material["weights"], online=True)
                    processor = resolve(material["processor"], online=True)
                    _compose_gemma(target, material, weights, processor)
                elif provider == "huggingface":
                    _download_hf(root, target, material)
                elif provider == "ngc":
                    preverified = _download_ngc(target, material)
                else:
                    raise AssetError(f"Unknown model source provider: {provider}")
                path = _check_files(target, material)
        _verify_weight_hashes(
            root, path, material, force=verify_weights, preverified=preverified
        )
        resolved[key] = path
        return path

    for key in dict.fromkeys(
        key for bindings in plan.values() for key in bindings.values()
    ):
        resolve(key)
    if pending:
        if not download:
            detail = ", ".join(
                f"{k} ({_target(root, materials[k])})" for k in sorted(pending)
            )
            raise AssetError(
                f"Pinned assets are missing or invalid: {detail}. Disable offline mode or provide verified checkpoint paths."
            )
        _ssd_root(
            config,
            root,
            _missing_bytes(
                root, plan, {**explicit, **resolved}, materials, reuse_hub=reuse_hub
            ),
        )
        with _acquisition_lock(root):
            for key in sorted(pending):
                resolve(key, online=True)
    for model, bindings in plan.items():
        row = result.setdefault("models", {}).setdefault(model, {})
        for role, key in bindings.items():
            row[role] = str(resolved[key])
    return result


def _command(args, *, failure=""):
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, check=True, timeout=10
        )
        return proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return failure


def _torch_build(dist):
    """Read Torch's generated version file without importing it or touching CUDA."""
    if dist is None:
        return {}
    try:
        tree = ast.parse(dist.locate_file("torch/version.py").read_text())
        result = {}
        for statement in tree.body:
            names = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target] if isinstance(statement, ast.AnnAssign) else []
            )
            for name in names:
                if (
                    isinstance(name, ast.Name)
                    and name.id in {"__version__", "cuda"}
                    and isinstance(statement.value, ast.Constant)
                ):
                    result[name.id] = statement.value.value
        return result
    except (OSError, SyntaxError):
        return {}


def _cuda_toolchain():
    root = os.environ.get(_CUDA_ENV) or os.environ.get("CUDA_HOME")
    bin_dirs = [str(Path(sys.executable).parent)]
    if root:
        bin_dirs.insert(0, str(Path(root).expanduser() / "bin"))
    search = os.pathsep.join([*bin_dirs, os.environ.get("PATH", "")])
    nvcc = (
        str(Path(root).expanduser() / "bin/nvcc")
        if root
        else os.environ.get("FLASHINFER_NVCC") or shutil.which("nvcc", path=search)
    )
    output = _command([nvcc, "--version"]) if nvcc else ""
    match = re.search(r"\bV(\d+\.\d+\.\d+)\b", output)
    return {
        "nvcc": nvcc,
        "version": match.group(1) if match else None,
        "ninja": shutil.which("ninja", path=search),
    }


def _available_memory_gib():
    try:
        line = next(
            x
            for x in Path("/proc/meminfo").read_text().splitlines()
            if x.startswith("MemAvailable:")
        )
        return round(int(line.split()[1]) * 1024 / _GIB, 2)
    except (OSError, StopIteration, ValueError):
        return None


def _runtime_report(runtime, models):
    package_report = {}
    distributions = {}
    for name, expected in runtime["packages"].items():
        try:
            dist = metadata.distribution(name)
            distributions[name] = dist
            version = dist.version
        except metadata.PackageNotFoundError:
            version = None
        package_report[name] = {
            "expected": expected,
            "found": version,
            "ok": version == expected,
        }
        if name in runtime["package_sources"]:
            package_report[name]["source"] = runtime["package_sources"][name]
    build = _torch_build(distributions.get("torch"))
    torch_report = {
        "expected_build": runtime["torch_build"],
        "found_build": build.get("__version__"),
        "expected_cuda": runtime["torch_cuda"],
        "found_cuda": build.get("cuda"),
        "ok": build.get("__version__") == runtime["torch_build"]
        and build.get("cuda") == runtime["torch_cuda"],
    }
    file_report = {}
    for name, digest in runtime["files"].items():
        dist = distributions.get(name.split("/", 1)[0])
        path = dist.locate_file(name) if dist else None
        found = _file_sha(path) if path and path.is_file() else None
        file_report[name] = {
            "sha256": found,
            "expected_sha256": digest,
            "ok": found == digest,
        }

    compiler = _cuda_toolchain()
    compiler["required"] = bool(set(models) & set(runtime["compiler_models"]))
    compiler["expected"] = runtime["compiler"]
    compiler["source"] = runtime.get("compiler_source")
    compiler["ok"] = not compiler["required"] or bool(
        compiler["version"] == compiler["expected"] and compiler["ninja"]
    )

    host = {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
    }
    host["ok"] = all(
        host[name] == runtime[name] for name in ("system", "machine", "python")
    )
    return {
        "ok": bool(
            host["ok"]
            and all(package["ok"] for package in package_report.values())
            and torch_report["ok"]
            and file_report
            and all(file["ok"] for file in file_report.values())
            and compiler["ok"]
        ),
        "host": host,
        "packages": package_report,
        "torch_build": torch_report,
        "runtime_files": file_report,
        "cuda_compiler": compiler,
    }


def assert_runtime_environment(config, models):
    """Reject an unqualified software stack before starting inference."""
    runtime = _lock(config)["runtime"]
    report = _runtime_report(runtime, _keys(models))
    if report["ok"]:
        return

    failures = []
    host = report["host"]
    for name in ("system", "machine", "python"):
        if host[name] != runtime[name]:
            failures.append(f"{name}: expected {runtime[name]}, found {host[name]}")
    for name, package in report["packages"].items():
        if not package["ok"]:
            failures.append(
                f"{name}: expected {package['expected']}, found {package['found'] or 'not installed'}"
            )
    build = report["torch_build"]
    if not build["ok"]:
        failures.append(
            f"Torch build: expected {build['expected_build']} with CUDA {build['expected_cuda']}, "
            f"found {build['found_build']} with CUDA {build['found_cuda']}"
        )
    bad_files = [
        name for name, file in report["runtime_files"].items() if not file["ok"]
    ]
    if not report["runtime_files"]:
        failures.append("The runtime lock lists no SDK files")
    if bad_files:
        failures.append("Missing or changed SDK files: " + ", ".join(bad_files))
    compiler = report["cuda_compiler"]
    if not compiler["ok"]:
        if compiler["version"] != compiler["expected"]:
            failures.append(
                f"nvcc: expected {compiler['expected']}, found {compiler['version'] or 'unavailable'} "
                f"(set {_CUDA_ENV} to the pinned CUDA toolkit)"
            )
        if not compiler["ninja"]:
            failures.append("Ninja is unavailable")
    raise AssetError("Inference runtime check failed.\n  " + "\n  ".join(failures))


def doctor_environment(config, models, model_root=None):
    """Report package, device and local asset readiness without CUDA initialization."""
    lock = _lock(config)
    runtime = lock["runtime"]
    plan, explicit = _model_plan(config, models, lock)
    report = _runtime_report(runtime, plan)
    host = report["host"]
    host["available_ram_gib"] = _available_memory_gib()
    driver = []
    smi = _command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,compute_cap,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    for row in csv.reader(smi.splitlines(), skipinitialspace=True):
        if len(row) != 7:
            continue
        index, uuid, name, version, capability, total, free = row
        try:
            int(version.split(".")[0])
            driver.append(
                {
                    "index": int(index),
                    "uuid": uuid,
                    "name": name,
                    "driver_version": version,
                    "compute_capability": capability,
                    "total_vram_mib": int(total),
                    "free_vram_mib": int(free),
                }
            )
        except ValueError:
            continue
    jobs = _command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        failure=None,
    )
    running = {}
    if jobs is not None:
        for row in csv.reader(jobs.splitlines(), skipinitialspace=True):
            if len(row) == 2 and row[0].startswith("GPU-") and row[1].isdigit():
                running[row[0]] = running.get(row[0], 0) + 1
    for gpu in driver:
        gpu["active_compute_processes"] = (
            running.get(gpu["uuid"], 0) if jobs is not None else None
        )
    gpu_requirement = runtime["gpu_compute_capability"]
    driver_requirement = runtime.get("minimum_driver_major", 580)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    visibility = visible.split(",") if visible is not None else None
    eligible = [
        gpu
        for gpu in driver
        if gpu["compute_capability"] == gpu_requirement
        and int(gpu["driver_version"].split(".")[0]) >= driver_requirement
        and (
            visibility is None
            or str(gpu["index"]) in visibility
            or any(
                item.startswith("GPU-") and gpu["uuid"].startswith(item)
                for item in visibility
            )
        )
    ]
    model_report = {}
    checked = {}
    reuse_hub = not (
        model_root or os.environ.get(_ROOT_ENV) or config.get("model_root")
    )
    try:
        root = _ssd_root(config, model_root)
        storage = storage_info(root)
        storage["download_bytes_missing"] = _missing_bytes(
            root, plan, explicit, lock["materials"], reuse_hub=reuse_hub
        )
        storage["download_fits"] = (
            storage["free_bytes"] >= storage["download_bytes_missing"]
        )
    except AssetError as exc:
        root = None
        storage = {"error": str(exc)}
    for model, bindings in plan.items():
        assets = {}
        for role, key in bindings.items():
            if key not in checked:
                material = lock["materials"][key]
                path = explicit.get(key) or (_target(root, material) if root else None)
                asset = {
                    "provider": material["provider"],
                    "repo": material["repo"],
                    "revision": material["revision"],
                    "path": str(path) if path else None,
                    "ready": False,
                }
                if path is not None:
                    try:
                        _check_files(path, material)
                        asset["ready"] = True
                    except AssetError as exc:
                        asset["detail"] = str(exc)
                if (
                    key not in explicit
                    and reuse_hub
                    and not asset["ready"]
                    and (snapshot := _hub_snapshot(material))
                ):
                    if _path_ready(snapshot, material):
                        asset.pop("detail", None)
                        asset.update(path=str(snapshot), ready=True)
                if asset["ready"]:
                    proof = _full_weight_proof(root, Path(asset["path"]), material)
                    asset["verification"] = (
                        "full_sha256"
                        if proof
                        else "metadata_sha256_and_shard_sizes; full_sha256_on_first_run"
                    )
                checked[key] = asset
            assets[role] = checked[key]
        model_config = config.get("models", {}).get(model, {})
        required_vram = model_config.get("minimum_free_vram_mib", 0)
        required_ram = model_config.get("minimum_host_ram_gib", 0)
        cards = [
            gpu["index"]
            for gpu in eligible
            if gpu["free_vram_mib"] >= required_vram
            and gpu["active_compute_processes"] == 0
        ]
        ram_ok = not required_ram or (
            host["available_ram_gib"] is not None
            and host["available_ram_gib"] >= required_ram
        )
        model_report[model] = {
            "assets": assets,
            "minimum_free_vram_mib": required_vram,
            "available_gpu_indices": cards,
            "minimum_host_ram_gib": required_ram,
            "host_ram_ok": ram_ok,
            "ok": bool(
                cards and ram_ok and all(asset["ready"] for asset in assets.values())
            ),
        }
    return {
        **report,
        "ok": bool(
            report["ok"]
            and model_report
            and all(m["ok"] for m in model_report.values())
        ),
        "gpu": {
            "qualified": runtime["qualified_gpu"],
            "required_compute_capability": gpu_requirement,
            "minimum_driver_major": driver_requirement,
            "detected": driver,
        },
        "storage": storage,
        "models": model_report,
    }
