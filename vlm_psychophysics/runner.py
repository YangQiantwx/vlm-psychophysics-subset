"""Schedule independent benchmark calls and persist their execution state."""

from __future__ import annotations

from collections import defaultdict
from contextlib import closing, contextmanager
import hashlib
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import time

from . import data, results


PROJECT = Path(__file__).resolve().parent.parent
GPU_PATTERN = r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"
GIB = 1024**3


def existing_parent(path):
    path = Path(path).expanduser().resolve()
    while not path.exists():
        path = path.parent
    return path


def mount_for(path):
    output = subprocess.check_output(
        [
            "findmnt",
            "--json",
            "--target",
            str(path),
            "--output",
            "TARGET,SOURCE,FSTYPE,OPTIONS",
        ],
        text=True,
    )
    return json.loads(output)["filesystems"][0]


def check_storage(path, *, minimum_free=0, writable=False):
    from .assets import storage_info

    path = existing_parent(path)
    info = storage_info(path)
    if info["kind"] != "ssd":
        raise ValueError(f"{path} has no verified SSD backing.")
    if writable:
        mount = mount_for(path)
        if "rw" not in mount["options"].split(",") or not os.access(path, os.W_OK):
            raise ValueError(f"{path} is not writable.")
        info["mount"] = mount["target"]
    if info["free_bytes"] < minimum_free:
        raise ValueError(f"{path} needs {minimum_free / GIB:.1f} GiB free on SSD.")
    return info


def data_directory(explicit=None, *, create=True, require_ssd=True):
    explicit = explicit or os.environ.get("VLM_PSYCHOPHYSICS_DATA_ROOT")
    if not require_ssd:
        root = Path(explicit or "dataset").expanduser().resolve()
        if create:
            root.mkdir(parents=True, exist_ok=True)
        return root
    if explicit:
        candidates = [Path(explicit)]
    else:
        xdg = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
        candidates = []
        local = Path.cwd() / "dataset"
        index = local / "dataset-index.json"
        if index.is_file() and data.file_sha(index) == data.dataset_catalog.INDEX_SHA:
            candidates.append(local)
        candidates.append(xdg / "vlm-psychophysics" / "data" / data.REVISION)
        mounts = json.loads(
            subprocess.check_output(
                ["findmnt", "--list", "--json", "--output", "TARGET,SOURCE"], text=True
            )
        )["filesystems"]
        candidates.extend(
            Path(mount["target"])
            / f"vlm-psychophysics-{os.getuid()}"
            / "data"
            / data.REVISION
            for mount in mounts
            if mount["source"].startswith("/dev/")
        )
    reasons = []
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        try:
            check_storage(candidate, minimum_free=GIB, writable=True)
            if create:
                candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            reasons.append(str(error))
    raise ValueError(
        "No writable SSD data directory found. Set --data-root. " + " ".join(reasons)
    )


def available_ram():
    with Path("/proc/meminfo").open() as stream:
        for line in stream:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise ValueError("Linux MemAvailable is unavailable")


def gpu_list():
    properties = "uuid,index,name,compute_cap,memory.free,memory.used,utilization.gpu"
    output = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={properties}", "--format=csv,noheader,nounits"],
        text=True,
    )
    devices = {}
    for line in output.splitlines():
        uuid, index, name, capability, free, used, util = (
            field.strip() for field in line.split(",")
        )
        if not re.fullmatch(GPU_PATTERN, uuid):
            raise ValueError(f"Unexpected GPU UUID {uuid!r}")
        devices[uuid] = dict(
            uuid=uuid,
            index=int(index),
            name=name,
            compute_capability=capability,
            free_mib=int(free),
            used_mib=int(used),
            utilization=int(util),
            compute_pids=[],
        )
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    for line in output.splitlines():
        uuid, pid = (field.strip() for field in line.split(","))
        if uuid in devices:
            devices[uuid]["compute_pids"].append(int(pid))
    return devices


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    return {"pid": pid, "parent_pid": int(fields[1]), "start_ticks": int(fields[19])}


def is_descendant(pid, ancestor):
    identity = process_identity(pid)
    while identity:
        if identity["pid"] == ancestor["pid"]:
            return identity["start_ticks"] == ancestor["start_ticks"]
        if identity["parent_pid"] <= 1:
            return False
        identity = process_identity(identity["parent_pid"])
    return False


def idle_for_model(gpu, requirement, owner=None):
    pids = gpu["compute_pids"]
    if pids and (owner is None or not all(is_descendant(pid, owner) for pid in pids)):
        return False
    return (
        gpu["free_mib"] >= requirement["minimum_free_vram_mib"]
        and gpu["used_mib"] <= 1024
        and gpu["utilization"] <= 5
    )


def allowed_gpus(selection, observed, capability):
    visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    selectors = visibility.split(",") if visibility is not None else None
    eligible = {
        uuid
        for uuid, gpu in observed.items()
        if gpu["compute_capability"] == capability
        and (
            selectors is None
            or str(gpu["index"]) in selectors
            or any(
                item.startswith("GPU-") and uuid.startswith(item) for item in selectors
            )
        )
    }
    if selection == "auto":
        return sorted(eligible)
    devices = [item.strip() for item in selection.split(",")]
    if len(set(devices)) != len(devices) or any(gpu not in eligible for gpu in devices):
        raise ValueError(
            "Choose visible, compatible GPU UUIDs with --gpus or use auto."
        )
    return sorted(devices)


def runtime_lock_path(config):
    path = Path(config["runtime_lock"]).expanduser()
    return path if path.is_absolute() else PROJECT / path


def required_capability(config):
    return json.loads(runtime_lock_path(config).read_bytes())["runtime"][
        "gpu_compute_capability"
    ]


def new_schedule(config, models, selection):
    observed = gpu_list()
    allowed = allowed_gpus(selection, observed, required_capability(config))
    ram_per_worker = (
        max(config["models"][model]["minimum_host_ram_gib"] for model in models) * GIB
    )
    worker_limit = max(0, (available_ram() - 8 * GIB) // ram_per_worker)
    schedules = {}
    for model in models:
        candidates = [
            gpu
            for gpu in allowed
            if idle_for_model(observed[gpu], config["models"][model])
        ]
        schedules[model] = candidates[:worker_limit]
    unavailable = [model for model in models if not schedules[model]]
    if unavailable:
        state = ", ".join(
            f"{gpu} ({observed[gpu]['used_mib']} MiB used, PIDs {observed[gpu]['compute_pids']})"
            for gpu in allowed
        )
        raise ValueError(
            f"No eligible idle GPU/RAM for {', '.join(unavailable)}. {state}"
        )
    active = sorted({uuid for candidates in schedules.values() for uuid in candidates})
    if len(active) > worker_limit:
        allowed = sorted(active, key=lambda gpu: (-observed[gpu]["free_mib"], gpu))[
            :worker_limit
        ]
        for model in models:
            schedules[model] = sorted(set(schedules[model]).intersection(allowed))
    if any(not candidates for candidates in schedules.values()):
        raise ValueError(
            "Available host RAM cannot support the selected set of GPU workers."
        )
    return {
        "model_gpus": schedules,
        "devices": {gpu: observed[gpu] for gpu in allowed},
        "shard_policy": "sorted cases per model, round-robin on sorted frozen GPU UUIDs",
    }


@contextmanager
def exclusive_lock(path, *, shared_host=False):
    flags = os.O_RDWR | os.O_NOFOLLOW
    try:
        try:
            fd = os.open(path, flags)
        except FileNotFoundError:
            try:
                fd = os.open(
                    path,
                    flags | os.O_CREAT | os.O_EXCL,
                    0o666 if shared_host else 0o600,
                )
            except FileExistsError:
                fd = os.open(path, flags)
    except PermissionError as error:
        raise ValueError(f"Unable to acquire lock {path}") from error
    with os.fdopen(fd, "r+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            lock.seek(0)
            raise ValueError(f"Busy lock {path}: {lock.read().strip()}") from error
        if shared_host and os.fstat(lock.fileno()).st_uid == os.getuid():
            os.fchmod(lock.fileno(), 0o666)
        lock.seek(0)
        lock.truncate()
        lock.write(json.dumps(process_identity(os.getpid())) + "\n")
        lock.flush()
        yield lock.fileno()


def gpu_lock(uuid):
    if not re.fullmatch(GPU_PATTERN, uuid):
        raise ValueError(f"Invalid GPU UUID {uuid}")
    return exclusive_lock(
        Path("/tmp") / f"vlm-psychophysics-{uuid}.lock", shared_host=True
    )


def init_ledger(out):
    ledger = sqlite3.connect(out / "evidence/execution.sqlite3", timeout=30)
    ledger.row_factory = sqlite3.Row
    ledger.execute("PRAGMA journal_mode = WAL")
    ledger.execute("PRAGMA synchronous = FULL")
    ledger.execute(
        """
        CREATE TABLE IF NOT EXISTS attempts (
            job_id TEXT PRIMARY KEY,
            gpu_uuid TEXT NOT NULL,
            state TEXT NOT NULL CHECK (state IN ('INTENT', 'RETURNED')),
            started_ns INTEGER NOT NULL,
            finished_ns INTEGER,
            answer_json TEXT,
            error TEXT
        )
    """
    )
    return ledger


def record_intent(ledger, job, uuid):
    with ledger:
        inserted = ledger.execute(
            "INSERT OR IGNORE INTO attempts(job_id,gpu_uuid,state,started_ns) VALUES(?,?,'INTENT',?)",
            (job, uuid, time.time_ns()),
        )
    return inserted.rowcount == 1


def record_answer(ledger, job, answer):
    payload = data.json_bytes(answer).decode()
    with ledger:
        update = ledger.execute(
            "UPDATE attempts SET state='RETURNED', finished_ns=?, answer_json=? "
            "WHERE job_id=? AND state='INTENT'",
            (time.time_ns(), payload, job),
        )
        if update.rowcount != 1:
            raise ValueError(f"Missing request intent for {job}")


def record_error(ledger, job, error):
    with ledger:
        ledger.execute(
            "UPDATE attempts SET error=? WHERE job_id=? AND state='INTENT'",
            (f"{type(error).__name__}: {error}", job),
        )


def source_fingerprint():
    names = {"run.py", "generate.py", "pyproject.toml"}
    names.update(
        path.relative_to(PROJECT).as_posix()
        for path in (PROJECT / "vlm_psychophysics").rglob("*.py")
    )
    return {name: data.file_sha(PROJECT / name) for name in sorted(names)}


def save_plan(out, config_path, config, models, manifest, selection):
    (out / "evidence").mkdir(parents=True, exist_ok=True)
    config_bytes = config_path.read_bytes()
    if json.loads(config_bytes) != config:
        raise ValueError("Config changed while creating the plan")
    input_bytes = results.jsonl_bytes(manifest["cases"])
    source_bytes = data.json_bytes(source_fingerprint()) + b"\n"
    runtime_lock = runtime_lock_path(config).read_bytes()
    run = {
        "schema": "vlm-psychophysics-run-v1",
        "repo_id": manifest["repo_id"],
        "revision": manifest["revision"],
        "index_sha256": manifest["index_sha256"],
        "manifests": manifest["manifests"],
        "selection": selection,
        "models": list(models),
        "canonical_cases": len(manifest["cases"]),
        "logical_rows": sum(len(case["aliases"]) for case in manifest["cases"]),
        "planned_calls": len(models) * len(manifest["cases"]),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "input_manifest_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "source_code_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "runtime_lock_sha256": hashlib.sha256(runtime_lock).hexdigest(),
    }
    run_path = out / "evidence/run.json"
    payloads = {
        "evidence/config.json": config_bytes,
        "evidence/runtime-lock.json": runtime_lock,
        "evidence/input-manifest.jsonl": input_bytes,
        "evidence/source-code.json": source_bytes,
    }
    if run_path.exists():
        if json.loads(run_path.read_bytes()) != run:
            raise ValueError(
                "Output already has a different frozen plan. Use a new --out directory."
            )
        for name, payload in payloads.items():
            if data.file_sha(out / name) != hashlib.sha256(payload).hexdigest():
                raise ValueError(f"Existing evidence changed: {name}")
    else:
        for name, payload in payloads.items():
            results.write_bytes(out / name, payload)
        results.write_json(run_path, run)
    return run


def load_plan(out):
    run = json.loads((out / "evidence/run.json").read_bytes())
    for name, field in (
        ("config.json", "config_sha256"),
        ("input-manifest.jsonl", "input_manifest_sha256"),
        ("source-code.json", "source_code_sha256"),
        ("runtime-lock.json", "runtime_lock_sha256"),
    ):
        if data.file_sha(out / "evidence" / name) != run[field]:
            raise ValueError(f"Frozen evidence SHA-256 differs: {name}")
    config = json.loads((out / "evidence/config.json").read_bytes())
    with (out / "evidence/input-manifest.jsonl").open(encoding="utf-8") as stream:
        cases = data.read_jsonl(stream)
    return run, config, cases


def job_assignments(cases, models, schedule):
    for model in models:
        devices = schedule["model_gpus"][model]
        for index, case in enumerate(cases):
            yield devices[index % len(devices)], model, case, results.job_id(
                case, model
            )


def answer_receipt(answer):
    if not isinstance(answer, dict):
        raise ValueError("Model adapter must return an answer and an input receipt.")
    normalized = {
        key: answer.get(key)
        for key in (
            "raw_answer",
            "output_token_ids",
            "finish_reason",
            "stop_reason",
            "technical_pass",
            "technical_error",
            "input_receipt",
            "profile_sha256",
        )
    }
    ids = answer.get("prompt_token_ids", answer.get("unexpected_prompt_token_ids"))
    if ids is None:
        normalized["prompt_token_count"] = answer.get("prompt_token_count")
        normalized["prompt_token_ids_sha256"] = answer.get("prompt_ids_sha256")
    else:
        normalized["prompt_token_count"] = len(ids)
        normalized["prompt_token_ids_sha256"] = data.digest(ids)
        receipt = normalized["input_receipt"] or {}
        if receipt.get("deep_observation") or "unexpected_prompt_token_ids" in answer:
            normalized["prompt_token_ids"] = ids
    return normalized


def worker_jobs(out, uuid, active_models):
    run, config, cases = load_plan(out)
    schedule = json.loads((out / "evidence/schedule.json").read_bytes())
    with closing(init_ledger(out)) as ledger:
        attempted = {
            row["job_id"] for row in ledger.execute("SELECT job_id FROM attempts")
        }
    grouped = defaultdict(list)
    for assigned, model, case, job in job_assignments(cases, run["models"], schedule):
        if assigned == uuid and model in active_models and job not in attempted:
            grouped[(model, results.recipe_for(config, case, model))].append(
                (case, job)
            )
    return config, grouped


def supervise_gpu(
    out, data_root, runtime_config_path, uuid, active_models, run_lock_fd
):
    """Hold one GPU across its queue; each recipe gets an isolated Python runtime."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != uuid:
        raise ValueError("Worker CUDA visibility does not match its reserved GPU.")
    with gpu_lock(uuid) as gpu_lock_fd:
        _, grouped = worker_jobs(out, uuid, active_models)
        for model, recipe in grouped:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "vlm_psychophysics.runner",
                    "recipe",
                    str(out),
                    str(data_root),
                    str(runtime_config_path),
                    uuid,
                    model,
                    recipe,
                ],
                cwd=PROJECT,
                pass_fds=(run_lock_fd, gpu_lock_fd),
            )
            identity = process_identity(child.pid)
            exit_code = child.wait()
            if identity:
                session_dir = session_directory(out, uuid, recipe, identity)
                results.write_json(
                    session_dir / "process-exit.json",
                    {"process": identity, "exit_code": exit_code},
                )
            if exit_code:
                raise RuntimeError(f"{uuid} {recipe} exited with status {exit_code}")


def session_directory(out, uuid, recipe, owner):
    session_id = f'{owner["pid"]}-{owner["start_ticks"]}'
    return out / "evidence" / "sessions" / uuid / recipe / session_id


def run_recipe(out, data_root, runtime_config_path, uuid, model, recipe):
    config, grouped = worker_jobs(out, uuid, (model,))
    pending = grouped.get((model, recipe))
    if not pending:
        return
    runtime_config = json.loads(runtime_config_path.read_bytes())
    owner = process_identity(os.getpid())
    gpu = gpu_list().get(uuid)
    if gpu is None or not idle_for_model(gpu, config["models"][model], owner):
        raise ValueError(
            f"{uuid} is busy or has insufficient free VRAM before {recipe}."
        )
    if available_ram() < config["models"][model]["minimum_host_ram_gib"] * GIB:
        raise ValueError(f"Available host RAM is below the requirement for {recipe}.")
    from .models import Session

    first = pending[0][0]
    print(f"{uuid} {recipe} starting {len(pending)} independent calls", flush=True)
    session = Session(
        model,
        first["task"],
        first["modality"],
        runtime_config,
        data_root,
        session_directory(out, uuid, recipe, owner),
        uuid,
        scope=data.case_scope(first),
        cohort=first.get("cohort"),
    )
    try:
        with closing(init_ledger(out)) as ledger:
            for count, (case, job) in enumerate(pending, 1):
                supplied = data.runtime_case(case, data_root)
                if not record_intent(ledger, job, uuid):
                    continue
                try:
                    answer = session.predict(supplied)
                    record_answer(ledger, job, answer_receipt(answer))
                except BaseException as error:
                    record_error(ledger, job, error)
                    raise
                if answer.get("technical_pass") is not True:
                    raise RuntimeError(
                        f'{uuid} {recipe} failed validation: {answer.get("technical_error")}'
                    )
                if count % 100 == 0 or count == len(pending):
                    print(
                        f"{uuid} {recipe} {count}/{len(pending)} returned", flush=True
                    )
    finally:
        session.close()


def stop_workers(children):
    for child, identity in children:
        if child.poll() is None and identity == process_identity(child.pid):
            os.killpg(child.pid, signal.SIGTERM)
    for child, identity in children:
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            if identity == process_identity(child.pid):
                os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def execute(
    out,
    config,
    cases,
    models,
    data_root,
    overrides,
    gpus,
    model_root,
    *,
    offline,
    lock_fd,
):
    schedule_path = out / "evidence/schedule.json"
    if schedule_path.exists():
        schedule = json.loads(schedule_path.read_bytes())
        if gpus != "auto" and set(item.strip() for item in gpus.split(",")) != set(
            schedule["devices"]
        ):
            raise ValueError(
                "GPU allocation is frozen for this output. Resume with --gpus auto."
            )
    else:
        schedule = new_schedule(config, models, gpus)
    with closing(init_ledger(out)) as ledger:
        attempted = {
            row["job_id"] for row in ledger.execute("SELECT job_id FROM attempts")
        }
    by_gpu = defaultdict(set)
    observed = gpu_list()
    visible = allowed_gpus("auto", observed, required_capability(config))
    pending = []
    for uuid, model, case, job in job_assignments(cases, models, schedule):
        if (
            job not in attempted
            and uuid in visible
            and idle_for_model(observed[uuid], config["models"][model])
        ):
            by_gpu[uuid].add(model)
            pending.append((uuid, case))
    remaining_ram = max(0, available_ram() - 8 * GIB)
    for uuid, assigned_models in sorted(by_gpu.items()):
        required_ram = (
            max(
                config["models"][model]["minimum_host_ram_gib"]
                for model in assigned_models
            )
            * GIB
        )
        if required_ram > remaining_ram:
            del by_gpu[uuid]
        else:
            remaining_ram -= required_ram
    pending_cases = {case["case_id"]: case for uuid, case in pending if uuid in by_gpu}
    if not by_gpu:
        if len(attempted) == len(cases) * len(models):
            return
        raise ValueError(
            "The frozen allocation has no eligible idle GPU for the remaining calls."
        )
    if not schedule_path.exists():
        results.write_json(schedule_path, schedule)
    from .assets import ensure_models

    required_models = sorted(set().union(*by_gpu.values()))
    runtime_config = ensure_models(
        config, required_models, model_root, download=not offline
    )
    prepare = data.prepare_media(
        data_root,
        pending_cases.values(),
        overrides,
        offline=offline,
        check_storage=check_storage,
    )
    print(json.dumps(prepare), flush=True)
    runtime_path = out / "evidence/.runtime-config.json"
    results.write_json(runtime_path, runtime_config)
    children = []
    status = []
    try:
        for uuid in sorted(by_gpu):
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid)
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "vlm_psychophysics.runner",
                    "gpu",
                    str(out),
                    str(data_root),
                    str(runtime_path),
                    uuid,
                    ",".join(sorted(by_gpu[uuid])),
                    str(lock_fd),
                ],
                cwd=PROJECT,
                env=environment,
                start_new_session=True,
                pass_fds=(lock_fd,),
            )
            children.append((child, process_identity(child.pid)))
        for child, identity in children:
            status.append({"process": identity, "exit_code": child.wait()})
    except BaseException:
        stop_workers(children)
        status = [
            {"process": identity, "exit_code": child.returncode}
            for child, identity in children
        ]
        raise
    finally:
        runtime_path.unlink(missing_ok=True)
        path = out / "evidence/worker-exits.json"
        previous = json.loads(path.read_bytes()) if path.exists() else []
        results.write_json(path, previous + status)
    return status


def recover_returned(out, attempts, cases, config):
    """Keep a durable engine return if the worker exited before post-call validation."""
    case_by_id = {case["case_id"]: case for case in cases}
    for path in results.session_files(out, "returned.jsonl"):
        _, _, gpu, recipe, _, _ = path.relative_to(out).parts
        model = recipe.split("-", 1)[0]
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.endswith("\n"):
                    break
                returned = json.loads(line)
                case = case_by_id.get(returned["case_id"])
                if case is None or results.recipe_for(config, case, model) != recipe:
                    raise ValueError(f"Unexpected durable model return in {path}")
                job = results.job_id(case, model)
                attempt = attempts.get(job)
                if not attempt or attempt["gpu_uuid"] != gpu:
                    raise ValueError(
                        f"Durable model return has no matching GPU intent for {job}"
                    )
                if attempt["state"] == "RETURNED":
                    if attempt["answer"]["raw_answer"] != returned["raw_answer"]:
                        raise ValueError(
                            f"Durable raw answer differs from the ledger for {job}"
                        )
                    continue
                receipt = answer_receipt(returned)
                receipt["technical_pass"] = False
                receipt["technical_error"] = (
                    "Engine returned before the worker completed validation"
                )
                attempt.update(state="RETURNED", answer=receipt)


def report(out):
    run, config, cases = load_plan(out)
    with closing(init_ledger(out)) as ledger:
        attempts = {
            row["job_id"]: {
                "state": row["state"],
                "gpu_uuid": row["gpu_uuid"],
                "error": row["error"],
                "started_ns": row["started_ns"],
                "finished_ns": row["finished_ns"],
                "answer": (
                    json.loads(row["answer_json"]) if row["answer_json"] else None
                ),
            }
            for row in ledger.execute("SELECT * FROM attempts")
        }
    recover_returned(out, attempts, cases, config)
    rows = []
    for model in run["models"]:
        for case in cases:
            job = results.job_id(case, model)
            rows.append(
                results.result_row(
                    case, model, config, run["config_sha256"], attempts.pop(job, {})
                )
            )
    if attempts:
        raise ValueError(f"Execution ledger has {len(attempts)} unknown request IDs")
    evidence = {}
    for name in ("schedule", "worker-exits"):
        path = out / f"evidence/{name}.json"
        evidence[name] = json.loads(path.read_bytes()) if path.exists() else None
    summary = results.write_reports(out, rows, evidence)
    archive = results.make_archive(out)
    return {"archive": str(archive), **summary["overall"]}


def worker_main():
    def terminate(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    kind, out, data_root, runtime_config, uuid, *options = sys.argv[1:]
    arguments = (Path(out), Path(data_root), Path(runtime_config), uuid)
    if kind == "gpu":
        active_models, lock_fd = options
        supervise_gpu(*arguments, active_models.split(","), int(lock_fd))
    elif kind == "recipe":
        run_recipe(*arguments, *options)
    else:
        raise ValueError(f"Unknown worker type {kind!r}")


if __name__ == "__main__":
    worker_main()
