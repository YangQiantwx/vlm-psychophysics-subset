"""Read the release manifest and prepare exactly the selected input media."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import shutil
from subprocess import check_output
from tempfile import NamedTemporaryFile

from . import release as current_release, catalog as dataset_catalog
from . import paired_inputs


REPO_ID = current_release.REPO_ID
REVISION = current_release.REVISION
DEFAULT_SCOPES = current_release.SCOPES
PAIRED_SCOPES = current_release.PAIRED_SCOPES
SCOPES = DEFAULT_SCOPES
TASK_OPTIONS = {
    **{f"{scope[-2:]}-{scope.split('/')[0]}": scope for scope in DEFAULT_SCOPES},
    **{
        f"t5-{scope.split('/')[0]}-equal-canvas": scope
        for scope in paired_inputs.SCOPES
    },
}
MODELS = ("qwen", "cosmos", "nemo", "gemma")
MODALITIES = ("video", "static")
CHOICES = ["LEFT", "RIGHT"]
T5_CHOICES = ["YES", "NO"]
FIXED_STATIC_FRAMES = {"T1": (0, 1, 2, 3), "T4": (0, 179)}


def json_bytes(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"Invalid SHA-256: {value!r}")
    return value


def read_jsonl(stream):
    return [json.loads(line) for line in stream if line.strip()]


def fetch_release_file(
    root, filename, expected_sha=None, *, offline=False, revision=REVISION
):
    path = root / filename
    if not path.is_file():
        if offline:
            raise ValueError(f"Missing cached release file {path}. Run download first.")
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            revision=revision,
            filename=filename,
            local_dir=root,
            token=False,
        )
    if expected_sha is not None and file_sha(path) != expected_sha:
        raise ValueError(
            f"SHA-256 mismatch for {path}. Choose an intact data directory."
        )
    return path


def read_manifest(root, scope, entry, mode, *, offline):
    count, _, video_sha, static_sha = dataset_catalog.manifest_scope(scope)
    key = "metadata" if mode == "video" else "static_metadata"
    count_key = "media_count" if mode == "video" else "static_trial_count"
    sha = video_sha if mode == "video" else static_sha
    name = current_release.payload_path(scope, entry[key])
    path = fetch_release_file(root, name, sha, offline=offline)
    with path.open(encoding="utf-8") as stream:
        rows = read_jsonl(stream)
    by_id = {row["sample_id"]: row for row in rows}
    if len(rows) != count or len(by_id) != count or entry[count_key] != count:
        raise ValueError(f"Unexpected row count or duplicate sample IDs in {name}")
    domain, task, *_ = scope.split("/")
    if any(row["task"] != task.upper() or row["domain"] != domain for row in rows):
        raise ValueError(f"Unexpected task/domain in {name}")
    return by_id, {"path": name, "sha256": sha, "rows": count}


def parameters(row):
    return row.get("parameters") or row.get("source_parameters") or {}


def source_video(scope, row):
    return {
        "path": current_release.payload_path(
            paired_inputs.source_scope(scope), row.get("video_path") or row["path"]
        ),
        "sha256": require_sha(row["video_sha256"]),
        "frames": row["frames"],
        "fps": row["fps"],
        "width": row["width"],
        "height": row["height"],
    }


def source_reference(scope, row):
    descriptor = {
        "path": current_release.payload_path(scope, row["reference_path"]),
        "sha256": require_sha(row["reference_sha256"]),
        "pixel_sha256": require_sha(row["reference_pixel_sha256"]),
        "frame_index": None,
    }
    if paired_inputs.is_current_reference(scope):
        descriptor["size_wh"] = [768, 512]
    return descriptor


def image_size(image):
    if image["frame_index"] is not None:
        return (768, 512)
    size = tuple(image.get("size_wh", (160, 160)))
    if size not in ((160, 160), (768, 512)):
        raise ValueError("Unexpected independent reference geometry")
    return size


def source_alias(row, video):
    return {
        "sample_id": row["sample_id"],
        "truth": row["truth"],
        "parameters": parameters(row),
        "condition_key": row.get("condition_key"),
        "background_id": row.get("background_id"),
        "object_id": row.get("object_id"),
        "source_record_sha256": row.get("source_record_sha256"),
        "source_video": video,
    }


def checked_labels(row, choices):
    truth = row["truth"]
    expected = T5_CHOICES if row["task"] == "T5" else CHOICES
    if choices != expected or truth not in (*expected, None):
        raise ValueError(f"Unexpected choices or ground truth for {row['sample_id']}")
    eligible = row.get("accuracy_eligible", truth is not None)
    if type(eligible) is not bool or eligible != (truth is not None):
        raise ValueError(f"Invalid accuracy eligibility for {row['sample_id']}")
    return truth, eligible


def make_case(row, scope, modality, canonical_id, input_sha, video, images, aliases):
    choices = row.get("choices", CHOICES)
    truth, eligible = checked_labels(row, choices)
    prompt = row["prompt"]
    if not isinstance(prompt, str) or not prompt or prompt.isspace():
        raise ValueError(f"Empty prompt for {canonical_id}")
    case = {
        "case_id": digest([scope, modality, canonical_id, input_sha]),
        "sample_id": canonical_id,
        "task": row["task"],
        "domain": row["domain"],
        "modality": modality,
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "choices": choices,
        "truth": truth,
        "accuracy_eligible": eligible,
        "video": video,
        "video_sha256": video["sha256"],
        "images": images,
        "frame_indices": [image["frame_index"] for image in images],
        "model_input_sha256": input_sha,
        "aliases": aliases,
    }
    if paired_inputs.is_current_reference(scope):
        case["reference_version"] = paired_inputs.VERSION
        case["cohort"] = (
            "single" if len(aliases[0]["parameters"]["events"]) == 1 else "dual"
        )
    return case


def case_scope(case):
    task = case["task"].lower()
    scope = f'{case["domain"]}/{task}'
    if task == "t5" and any(
        "/paired-v1/" in image["path"] for image in case.get("images", [])
    ):
        scope += "/paired-v1"
        version = case.get("reference_version")
        if version is not None:
            scope += f"/{version}"
    return scope


def profile_key(modality, scope):
    task = scope.split("/")[1].upper()
    key = f"{modality}/{task}"
    return (
        f"{key}/{paired_inputs.VERSION}"
        if paired_inputs.is_current_reference(scope)
        else key
    )


def recipe_name(name, cohort):
    return f"{name}-{cohort}" if cohort is not None else name


def video_case(scope, row):
    video = source_video(scope, row)
    images = [source_reference(scope, row)] if row["task"] == "T5" else []
    identity = video_input_identity(video["sha256"], images)
    identity = digest([identity, row["prompt"], row.get("choices", CHOICES)])
    return make_case(
        row,
        scope,
        "video",
        row.get("canonical_video_id", row["sample_id"]),
        identity,
        video,
        images,
        [source_alias(row, video)],
    )


def video_input_identity(video_sha, images):
    return [images[0]["sha256"], video_sha] if images else video_sha


def static_frame_indices(parent):
    task = parent["task"]
    if task in FIXED_STATIC_FRAMES:
        return list(FIXED_STATIC_FRAMES[task])
    source = parameters(parent)
    if task == "T5":
        return sorted(event["onset_frame"] + 15 for event in source["events"])
    onset = source.get("onset_frame", source.get("onset"))
    if not isinstance(onset, int):
        raise ValueError(
            f'Static midpoint needs a source onset for {parent["sample_id"]}'
        )
    return [onset + 15]


def static_identity(scope, task, images, frames, prompt, choices, sample_id):
    """Check ordered stills and hash the exact model-visible input."""
    if any(type(index) is not int or not 0 <= index < 180 for index in frames):
        raise ValueError(f"Wrong ordered images for {sample_id}")
    indices = [image["frame_index"] for image in images]
    if task == "T5":
        roles = [image.get("role") for image in images]
        valid = (
            len(frames) in (1, 2)
            and roles == ["reference", *(["event_midpoint"] * len(frames))]
            and indices == [None, *frames]
            and frames == sorted(set(frames))
        )
    else:
        expected = FIXED_STATIC_FRAMES.get(task)
        valid = indices == frames and (
            frames == list(expected) if expected is not None else len(frames) == 1
        )
    if not valid:
        raise ValueError(f"Wrong ordered images for {sample_id}")
    for image in images:
        reference = image["frame_index"] is None
        image_scope = scope if reference else paired_inputs.source_scope(scope)
        current_release.payload_path(image_scope, image["path"])
        if reference:
            expected_size = (
                (768, 512) if paired_inputs.is_current_reference(scope) else (160, 160)
            )
            if image_size(image) != expected_size:
                raise ValueError(f"Wrong reference geometry for {sample_id}")
        require_sha(image["sha256"])
    pixels = [require_sha(image["pixel_sha256"]) for image in images]
    return digest([pixels, prompt, choices])


def static_cases(
    scope, parents, static_rows, *, expected_count=None, canonical_sources=None
):
    if parents.keys() != static_rows.keys():
        raise ValueError(f"Video and static sample IDs differ in {scope}")
    grouped = defaultdict(list)
    id_to_input = {}
    for sample_id, static in static_rows.items():
        parent = parents[sample_id]
        video = source_video(scope, parent)
        if (
            static["original_video_ID"] != sample_id
            or static["video_path"] != video["path"]
            or static["video_sha256"] != video["sha256"]
            or static["truth"] != parent["truth"]
        ):
            raise ValueError(f"Wrong static parent or truth for {sample_id}")
        images = [image.copy() for image in static["images"]]
        frames = static["frame_indices"]
        if frames != static_frame_indices(parent):
            raise ValueError(f"Wrong ordered images for {sample_id}")
        identity = static_identity(
            scope,
            parent["task"],
            images,
            frames,
            static["prompt"],
            static["choices"],
            sample_id,
        )
        if parent["task"] == "T5":
            reference = source_reference(scope, parent)
            if any(images[0][key] != value for key, value in reference.items()):
                raise ValueError(f"Wrong static reference for {sample_id}")
        case_name = static["static_case_id"]
        if (
            identity != static["model_input_sha256"]
            or id_to_input.setdefault(case_name, identity) != identity
        ):
            raise ValueError(f"Wrong static input identity for {sample_id}")
        checked_labels(static, static["choices"])
        alias = source_alias(parent, video)
        alias["static_case_id"] = case_name
        alias["frame_indices"] = static["frame_indices"]
        alias["images"] = images
        grouped[identity].append((static, alias, video, images))
    cases = []
    for identity, members in grouped.items():
        if len({item[0]["truth"] for item in members}) != 1:
            raise ValueError(
                f"Conflicting truth for identical static input {scope}/{identity}"
            )
        members.sort(key=lambda item: (item[0]["static_case_id"], item[0]["sample_id"]))
        if canonical_sources is None:
            row, _, video, images = members[0]
        else:
            case_name = members[0][0]["static_case_id"]
            canonical_sample = canonical_sources[case_name]
            matches = [
                member
                for member in members
                if member[0]["sample_id"] == canonical_sample
                and member[0]["static_case_id"] == case_name
            ]
            if len(matches) != 1:
                raise ValueError(f"Wrong canonical static source for {case_name}")
            row, _, video, images = matches[0]
        cases.append(
            make_case(
                row,
                scope,
                "static",
                row["static_case_id"],
                identity,
                video,
                images,
                [item[1] for item in members],
            )
        )
    if expected_count is None:
        expected_count = dataset_catalog.manifest_scope(scope)[1]
    if len(cases) != expected_count:
        raise ValueError(f"Independent static input count differs in {scope}")
    return cases


def selected_static_cases(scope, parents, static_rows, selected_ids, **static_options):
    for case in static_cases(scope, parents, static_rows, **static_options):
        aliases = [
            alias for alias in case["aliases"] if alias["sample_id"] in selected_ids
        ]
        if aliases:
            yield {**case, "aliases": aliases}


def published_selection(matches, sample_id, video_name):
    if sample_id is not None:
        return {
            (scope, row["sample_id"])
            for scope, row in matches
            if sample_id
            in (
                row["sample_id"],
                row.get("legacy_sample_id"),
                row.get("canonical_video_id"),
                (row.get("static") or {}).get("static_case_id"),
            )
        }
    if video_name is None:
        return {(scope, row["sample_id"]) for scope, row in matches}
    selector = Path(video_name)
    names = set()
    for scope, row in matches:
        path = source_video(scope, row)["path"]
        paths = (path, *row.get("alias_paths", []))
        if row.get("master_video"):
            paths += (row["master_video"]["path"],)
        if any(
            video_name == name
            or (video_name == selector.name and video_name in (Path(name).name, Path(name).stem))
            for name in paths
        ):
            names.add((scope, row["sample_id"]))
    if selector.is_file():
        sha = file_sha(selector)
        names = {
            (scope, row["sample_id"])
            for scope, row in matches
            if row["video_sha256"] == sha
        }
    if len(names) > 1:
        raise ValueError(
            "The video matches multiple published trials. Select a scope and sample ID."
        )
    return names


def custom_case(path, scopes, modalities, parents):
    if not path.is_file():
        raise ValueError(f"Unknown published video or local file {path}")
    if len(scopes) != 1 or modalities != ("video",):
        raise ValueError(
            "An unlisted local video requires a specific --task and --input video."
        )
    scope = scopes[0]
    if scope.split("/")[1] == "t5":
        raise ValueError("Paired T5 needs a published trial with its reference image.")
    rows = parents[scope].values()
    prompts = {(row["prompt"], tuple(row.get("choices", CHOICES))) for row in rows}
    if len(prompts) != 1:
        raise ValueError(f"Cannot select one published video prompt for {scope}")
    prompt, choices = prompts.pop()
    probe = json.loads(
        check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=width,height,avg_frame_rate,nb_read_frames",
                "-of",
                "json",
                str(path.resolve()),
            ]
        )
    )["streams"][0]
    frame_count = int(probe["nb_read_frames"])
    fps = float(Fraction(probe["avg_frame_rate"]))
    if (probe["width"], probe["height"], frame_count, fps) != (768, 512, 180, 60):
        raise ValueError(
            "The pinned video recipes expect 768 × 512, 180 frames at 60 fps."
        )
    sha = file_sha(path)
    sample_id = f"custom-{sha[:24]}"
    domain, task = scope.split("/")
    row = dict(
        task=task.upper(),
        domain=domain,
        sample_id=sample_id,
        prompt=prompt,
        choices=list(choices),
        truth=None,
    )
    video = dict(
        path=f"custom/{sample_id}{path.suffix}",
        sha256=sha,
        width=probe["width"],
        height=probe["height"],
        frames=frame_count,
        fps=fps,
    )
    identity = digest([sha, prompt, row["choices"]])
    case = make_case(
        row, scope, "video", sample_id, identity, video, [], [source_alias(row, video)]
    )
    return case, (video["path"], path.resolve())


def build_selection(
    root, scopes, modalities, *, sample=None, sample_id=None, video_name=None,
    offline=False,
):
    """Return the portable evidence manifest and any local source file override."""
    index_path = fetch_release_file(
        root, "dataset-index.json", dataset_catalog.INDEX_SHA, offline=offline
    )
    entries = json.loads(index_path.read_bytes())["scopes"]

    def load(scope):
        original = paired_inputs.source_scope(scope)
        entry = entries[original]
        parents, manifest = read_manifest(
            root, original, entry, "video", offline=offline
        )
        provenance = [manifest]
        stills = None
        if "static" in modalities:
            if original == scope and (
                scope in PAIRED_SCOPES or entry["static_purpose"] != "model evaluation"
            ):
                raise ValueError(f"{scope} stills are for inspection")
            stills, manifest = read_manifest(
                root, original, entry, "static", offline=offline
            )
            provenance.append(manifest)
        return scope, parents, stills, provenance

    with ThreadPoolExecutor(max_workers=min(4, len(scopes))) as pool:
        inputs = list(pool.map(load, scopes))
    parents = {scope: rows for scope, rows, _, _ in inputs}
    matches = [(scope, row) for scope, rows in parents.items() for row in rows.values()]
    if sample is None:
        selected = published_selection(matches, sample_id, video_name)
    else:
        selected = current_release.select_samples(
            ((scope, row["sample_id"], source_video(scope, row)["path"])
             for scope, row in matches),
            (sample,),
        )
    files = {}
    cases = []
    if not selected and video_name:
        case, override = custom_case(Path(video_name), scopes, modalities, parents)
        cases = [case]
        files = dict([override])
    else:
        if sample_id is not None:
            # Static case IDs need not be repeated in the embedded video row.
            selected.update(
                (scope, row["sample_id"])
                for scope, _, rows, _ in inputs
                for row in (rows or {}).values()
                if sample_id == row["static_case_id"]
            )
        if not selected:
            raise ValueError(
                f"No published trial matches {sample_id!r} in the selected scopes"
            )
        for scope, rows, stills, _ in inputs:
            ids = {sample for source, sample in selected if source == scope}
            if "video" in modalities:
                cases.extend(video_case(scope, rows[sample]) for sample in sorted(ids))
            if stills is not None and ids:
                options = {}
                if paired_inputs.is_current_reference(scope):
                    canonical = {
                        row["static_case_id"]: row["canonical_source_id"]
                        for row in stills.values()
                    }
                    # Keep parent scientific IDs intact when an ID-filtered release
                    # omits the full dataset's canonical video. Any retained alias
                    # of this identical static input is a valid transport source.
                    for case_id, source_id in canonical.items():
                        if source_id not in stills:
                            canonical[case_id] = min(
                                row["sample_id"] for row in stills.values()
                                if row["static_case_id"] == case_id
                            )
                    options = {
                        "expected_count": len(canonical),
                        "canonical_sources": canonical,
                    }
                cases.extend(selected_static_cases(scope, rows, stills, ids, **options))
        if video_name and Path(video_name).is_file():
            scope, sample = next(iter(selected))
            files[source_video(scope, parents[scope][sample])["path"]] = Path(
                video_name
            ).resolve()
    cases.sort(
        key=lambda case: (
            case["task"],
            case["domain"],
            case["modality"],
            case["sample_id"],
        )
    )
    if not cases or len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Empty plan or duplicate canonical case ID")
    return {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "index_sha256": dataset_catalog.INDEX_SHA,
        "manifests": [item for _, _, _, provenance in inputs for item in provenance],
        "cases": cases,
    }, files


def prepare_media(
    root, cases, overrides, *, offline=False, workers=4, check_storage=None,
    include_alias_images=False,
):
    """Materialize and verify the required media, including alias images for downloads."""
    media = {}
    for case in cases:
        entries = case["images"]
        if case["modality"] == "video":
            entries = [*entries, case["video"]]
        elif include_alias_images:
            entries = [*entries, *(image for alias in case["aliases"] for image in alias["images"])]
        for entry in entries:
            path, sha = entry["path"], entry["sha256"]
            if media.setdefault(path, sha) != sha:
                raise ValueError(f"Conflicting media SHA-256 for {path}")

    def prepare(item):
        path, expected_sha = item
        destination = root / path
        if not destination.resolve().is_relative_to(root.resolve()):
            raise ValueError(
                f"Input resolves outside the approved data root: {destination}"
            )
        if not destination.is_file() and check_storage:
            check_storage(destination, minimum_free=1024**3)
        if path in overrides:
            source = overrides[path]
            if not destination.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                with NamedTemporaryFile(
                    dir=destination.parent, prefix=".media-", delete=False
                ) as temporary:
                    staged = Path(temporary.name)
                    try:
                        with source.open("rb") as stream:
                            shutil.copyfileobj(stream, temporary)
                        temporary.flush()
                        if file_sha(staged) != expected_sha:
                            raise ValueError(f"Source changed while copying {source}")
                        staged.replace(destination)
                    finally:
                        staged.unlink(missing_ok=True)
            if file_sha(destination) != expected_sha:
                raise ValueError(f"SHA-256 mismatch for {destination}")
        else:
            revision = REVISION
            destination = fetch_release_file(
                root, path, expected_sha, offline=offline, revision=revision
            )
        if check_storage:
            check_storage(destination.resolve())
        return destination.stat().st_size

    with ThreadPoolExecutor(max_workers=workers) as pool:
        size = sum(pool.map(prepare, sorted(media.items())))
    return {"files_sha256_verified": len(media), "verified_bytes": size}


def runtime_case(case, root):
    """Build the Session input without labels, scientific parameters or alias metadata."""
    keys = (
        "case_id",
        "sample_id",
        "task",
        "domain",
        "modality",
        "prompt",
        "choices",
        "video_sha256",
        "frame_indices",
        "model_input_sha256",
        "prompt_sha256",
    )
    result = {key: case[key] for key in keys}
    if "reference_version" in case:
        result["reference_version"] = case["reference_version"]
        result["cohort"] = case["cohort"]
    if case["modality"] == "video":
        result["video_path"] = str((root / case["video"]["path"]).resolve())
        result.update(
            {key: case["video"][key] for key in ("fps", "frames", "width", "height")}
        )
    if case["images"]:
        result["images"] = [
            {**image, "path": str((root / image["path"]).resolve())}
            for image in case["images"]
        ]
    return result
