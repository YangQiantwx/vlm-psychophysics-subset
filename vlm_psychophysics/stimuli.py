"""Rebuild released stimuli from parameters and source assets on CPU."""

from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from tempfile import TemporaryDirectory

import av
import numpy as np

from .generators import classic_events as classic_t23
from . import catalog as dataset_catalog
from .media import opencv_rgb8, write_rgb_mp4
from .release import REPO_ID, REVISION

SOURCE_PARENT_REPO = "DrQY/vlm-psychophysics"
SOURCE_PARENT_REVISION = "901ab8c3065f3d21acb870c74a4cce76f0908b20"
from .generators import events as t5_paired

from .generators import _classic as classic
from .generators import natural


SCOPES = {
    f"t{i}-{domain}": f"{domain}/t{i}"
    for i in range(1, 6)
    for domain in ("classic", "natural")
}
ALL_TASKS = tuple(SCOPES)
SOURCES = "sources"
BASIS = f"{SOURCES}/classic/t1"
T5 = f"{SOURCES}/t5"


@lru_cache(maxsize=1024)
def checked_file(root, revision, relative, expected):
    parts = PurePosixPath(relative)
    if not parts.parts or parts.is_absolute() or ".." in parts.parts:
        raise ValueError(f"Unsafe dataset path {relative!r}")
    local = root / parts
    if not local.is_file():
        from huggingface_hub import hf_hub_download

        from huggingface_hub.utils import EntryNotFoundError
        try:
            downloaded = hf_hub_download(
                REPO_ID,
                relative,
                repo_type="dataset",
                revision=revision,
                token=False,
                # Keep concurrent renders on the Hub's immutable snapshot cache.
                cache_dir=root / ".cache" / "huggingface" / "hub",
            )
        except EntryNotFoundError:
            # Rendering assets remain in the immutable parent release; evaluation
            # media and metadata are shipped in this self-contained subset.
            downloaded = hf_hub_download(
                SOURCE_PARENT_REPO, relative, repo_type="dataset",
                revision=SOURCE_PARENT_REVISION, token=False,
                cache_dir=root / ".cache" / "huggingface" / "hub",
            )
        local = Path(downloaded)
    if expected is not None and file_sha(local) != expected:
        raise ValueError(f"Source SHA-256 differs for {relative}")
    return local


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Dataset:
    """Use a local snapshot where available and fetch missing files at one revision."""

    def __init__(self, root, revision):
        self.root = root
        self.revision = revision

    def file(self, relative, expected=None):
        return checked_file(self.root, self.revision, relative, expected)

    def sha(self, relative):
        return file_sha(self.file(relative))

    def json(self, relative):
        return json_file(self.file(relative))

    def index(self, relative, make_index):
        return indexed_file(self.file(relative), make_index)


@lru_cache(maxsize=12)
def json_file(path):
    return json.loads(path.read_bytes())


@lru_cache(maxsize=8)
def indexed_file(path, make_index):
    return make_index(json_file(path))


def render_classic_t1(data, row, destination):
    p = row["parameters"]
    name = f"sf{classic.token(p['sf_cycles_per_aperture'])}.npz"
    expected = data.json(f"{BASIS}/basis-sha256.json")[name]
    path = data.file(f"{BASIS}/basis/{name}", expected)
    trial = {"phase_index": p["bank_start_index"], "params": p}
    classic.encode(destination, classic.states(trial, path))


def render_classic_t23(row, destination, work):
    trial = {**row, "source_parameters": row["parameters"]}
    base, event, _ = classic_t23.make_state(trial)
    path = work / "states.npy"
    np.save(path, np.stack((base, event)), allow_pickle=False)
    result = classic_t23._encode_one((trial, {"indices": [0, 1]}, path, work, 0))
    (work / result["video_relative"]).replace(destination)


def render_classic_t4(data, row, destination):
    p = row["parameters"]
    start = np.asarray(p["positions_xy_px"])
    velocity = np.asarray(p["velocity_xy_px_s"])
    track = (
        start[None]
        + (np.arange(row["frames"])[:, None, None] / row["fps"]) * velocity[None]
    )
    if p.get("appearance") == "mixed":
        plan = json_file(data.file(p["shape_definitions"], row["source_plan_sha256"]))
        classic.encode_strict(
            destination, track, shape_ids=p["shape_ids"], definitions=plan["shapes"]
        )
    else:
        classic.encode_strict(destination, track)


def render_paired_t5(data, row, destination, work):
    for folder in ("qa", "videos", "inspection"):
        (work / folder).mkdir()
    trial = {**row, "video_path": str(work / "videos" / destination.name)}
    trial["reference_path"] = str(
        data.file(row["reference_path"], row["reference_sha256"])
    )
    if row["scene"] is not None:
        scene = trial["scene"] = row["scene"].copy()
        for key in ("background_prepared", "object_prepared"):
            entry = scene[key]
            scene[key] = {
                **entry,
                "path": str(data.file(entry["path"], entry["sha256"])),
            }
    t5_paired.render_one((work, trial))
    Path(trial["video_path"]).replace(destination)


def decoded_video_sha(path, row):
    """Decode every frame with the published pixel format and sample timing."""
    rgb16 = path.suffix == ".mkv"
    codec, pix_fmt = ("ffv1", "gbrp16le") if rgb16 else ("h264", "gbrp")
    fmt = "rgb48le" if rgb16 else "rgb24"
    digest = hashlib.sha256()
    frames = 0
    expected_per_frame = row.get("decoded_frame_sha256")
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        observed = (
            stream.codec_context.name,
            stream.codec_context.pix_fmt,
            stream.width,
            stream.height,
            stream.average_rate,
        )
        expected = codec, pix_fmt, row["width"], row["height"], row["fps"]
        if len(container.streams) != 1 or observed != expected:
            raise ValueError(
                f"Video stream parameters differ for {path.name}: {observed}"
            )
        for i, frame in enumerate(container.decode(stream)):
            ideal = Fraction(i, row["fps"])
            if rgb16:
                ideal = Fraction(int(ideal * 1000 + Fraction(1, 2)), 1000)
            if frame.pts is None or frame.pts * frame.time_base != ideal:
                raise ValueError(f"Timestamp differs for {path.name} at frame {i}")
            rgb = frame.to_ndarray(format=fmt)
            if rgb16:
                rgb = rgb.astype("<u2", copy=False)
            digest.update(rgb)
            if expected_per_frame is not None and (
                i >= len(expected_per_frame)
                or hashlib.sha256(rgb).hexdigest() != expected_per_frame[i]
            ):
                raise ValueError(f"RGB frame differs for {path.name} at frame {i}")
            frames += 1
    if frames != row["frames"] or (
        expected_per_frame is not None and len(expected_per_frame) != frames
    ):
        raise ValueError(f"Decoded frame count differs for {path.name}")
    return digest.hexdigest()


def verify_video(data, path, row):
    stream_sha = decoded_video_sha(path, row)
    expected = row.get("decoded_stream_sha256") or row.get("RGB_stream_sha256")
    if expected is None and row.get("decoded_frame_sha256") is None:
        # Classic T1 m=0.5 records identify the original master by its container SHA.
        reference = row.get("master_video")
        if reference is None:
            master = data.file(row.get("video_path") or row["path"], row["video_sha256"])
        else:
            from huggingface_hub import hf_hub_download

            master = Path(hf_hub_download(
                reference["repo_id"], reference["path"], repo_type="dataset",
                revision=reference["revision"], token=False,
            ))
            if file_sha(master) != reference["sha256"]:
                raise ValueError(f"Archived master SHA-256 differs for {path.name}")
        expected = decoded_video_sha(master, row)
    if expected is not None and stream_sha != expected:
        raise ValueError(f"Decoded RGB SHA-256 differs for {path.name}")
    return stream_sha


def render(job):
    data, row, output = job
    scope = f"{row['domain']}/{row['task'].lower()}"
    paired = row.get("source_version") == t5_paired.PROTOCOL
    if paired and "/paired-v1/" in row["video_path"]:
        scope += "/paired-v1"
    name = Path(row.get("video_path") or row["path"]).name
    destination = output / scope / "videos" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Output already exists: {destination}")
    with TemporaryDirectory(prefix=".render-", dir=destination.parent) as stage:
        work = Path(stage)
        video = work / name
        master = row.get("master_video")
        source = video
        if master is not None:
            source = video.with_suffix(".mkv")
        route = {
            "classic/t1": lambda: render_classic_t1(data, row, source),
            "classic/t2": lambda: render_classic_t23(row, source, work),
            "classic/t3": lambda: render_classic_t23(row, source, work),
            "classic/t4": lambda: render_classic_t4(data, row, source),
            "natural/t1": lambda: natural.render_t1(data, row, video),
            "natural/t2": lambda: natural.render_t2(data, row, video),
            "natural/t3": lambda: natural.render_t3(data, row, video, work),
            "natural/t4": lambda: natural.render_t4(data, row, video, work),
            "classic/t5": lambda: render_paired_t5(data, row, video, work),
            "natural/t5": lambda: render_paired_t5(data, row, video, work),
            "classic/t5/paired-v1": lambda: render_paired_t5(data, row, video, work),
            "natural/t5/paired-v1": lambda: render_paired_t5(data, row, video, work),
        }
        route[scope]()
        if master is not None:
            master_row = {
                **row,
                "decoded_stream_sha256": master.get("decoded_stream_sha256"),
                "decoded_frame_sha256": master.get("decoded_frame_sha256"),
                "RGB_stream_sha256": master.get("RGB_stream_sha256"),
            }
            verify_video(data, source, master_row)
            write_rgb_mp4(video, opencv_rgb8(source))
        stream_sha = verify_video(data, video, row)
        container_sha = file_sha(video)
        # Link only verified output into place and refuse concurrent overwrites.
        os.link(video, destination)
    receipt = {
        "sample_id": row["sample_id"],
        "path": str(destination),
        "decoded_rgb_sha256": stream_sha,
        "video_sha256": container_sha,
        "container_matches": container_sha == row["video_sha256"],
        "frames": row["frames"],
    }
    if row["task"] == "T5":
        receipt["purpose"] = (
            "video and static evaluation"
            if row.get("input_protocol")
            else "stimulus inspection"
        )
        receipt["ordered_images"] = [image["path"] for image in row["static"]["images"]]
    return receipt


def generate(task, sample_ids, output, data_root, revision=REVISION, workers=1):
    data = Dataset(Path(data_root).resolve(), revision)
    output = Path(output).resolve()
    if output.is_relative_to(data.root):
        raise ValueError("Choose an output directory outside the input data root")
    tasks = ALL_TASKS if task == "all" else (task,)
    if task == "all" and revision != REVISION:
        available = data.json("dataset-index.json")["scopes"]
        tasks = tuple(name for name in tasks if SCOPES[name] in available)
    wanted = set(sample_ids)
    seen = set()
    jobs = []
    for task in tasks:
        scope = SCOPES[task]
        with data.file(
            f"{scope}/metadata.jsonl", dataset_catalog.manifest_scope(scope, revision)[2]
        ).open() as source:
            for row in map(json.loads, source):
                matching = {row["sample_id"], row.get("legacy_sample_id")} & wanted
                if not wanted or matching:
                    seen.update(matching)
                    jobs.append((data, row, output))
    if wanted - seen:
        raise ValueError("Unknown sample IDs: " + ", ".join(sorted(wanted - seen)))
    if not jobs:
        raise ValueError("No stimulus trials selected")
    if workers == 1:
        yield from map(render, jobs)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            yield from pool.map(render, jobs)
