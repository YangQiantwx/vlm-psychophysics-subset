"""Lossless MP4 transport for the pixels delivered by the evaluation reader."""

from fractions import Fraction
import hashlib
from pathlib import Path


def x_reader():
    from vllm.multimodal.media.image import ImageMediaIO
    from vllm.multimodal.media.video import VideoMediaIO

    return VideoMediaIO(ImageMediaIO(), video_backend="opencv", num_frames=180, fps=-1)


def x_pixels(reader, path):
    frames, metadata = reader.load_file(Path(path)).media
    if (
        frames.dtype.name != "uint8"
        or frames.shape != (180, 512, 768, 3)
        or metadata["frames_indices"] != list(range(180))
        or metadata["total_num_frames"] != 180
        or metadata["fps"] != 60
    ):
        raise ValueError(f"Unexpected evaluation input at {path}")
    return frames, metadata


def opencv_rgb8(path):
    """Decode the regenerated master with OpenCV and check its size and order."""
    import cv2
    import numpy as np

    source = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not source.isOpened() or source.get(cv2.CAP_PROP_FPS) != 60:
        source.release()
        raise ValueError(f"Cannot read a 60 fps video at {path}")
    frames = np.empty((180, 512, 768, 3), dtype=np.uint8)
    try:
        for i in range(180):
            ok, bgr = source.read()
            if not ok or bgr.shape != (512, 768, 3) or bgr.dtype != np.uint8:
                raise ValueError(f"Unexpected frame {i} at {path}")
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB, dst=frames[i])
        if source.grab():
            raise ValueError(f"More than 180 frames at {path}")
    finally:
        source.release()
    return frames


def write_rgb_mp4(path, frames):
    import av

    with av.open(
        str(path), "w", format="mp4",
        options={"movflags": "+faststart", "video_track_timescale": "60000"},
    ) as video:
        stream = video.add_stream("libx264rgb", rate=60)
        stream.width, stream.height = 768, 512
        stream.pix_fmt = "rgb24"
        stream.codec_context.thread_count = 1
        stream.options = {"crf": "0", "preset": "ultrafast", "bf": "0"}
        for i, rgb in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, 60)
            for packet in stream.encode(frame):
                video.mux(packet)
        for packet in stream.encode():
            video.mux(packet)


def verify_timeline(path, *, master):
    import av

    with av.open(str(path)) as video:
        (stream,) = video.streams
        fmt = ("ffv1", "gbrp16le") if master else ("h264", "gbrp")
        actual = (stream.codec_context.name, stream.codec_context.pix_fmt)
        if actual != fmt or (stream.width, stream.height, stream.average_rate) != (768, 512, 60):
            raise ValueError(f"Unexpected video format at {path}: {actual}")
        stamps = [p.pts * p.time_base for p in video.demux(stream) if p.size]
    expected = [Fraction(i, 60) for i in range(180)]
    if master:
        expected = [Fraction(int(t * 1000 + Fraction(1, 2)), 1000) for t in expected]
    if stamps != expected:
        raise ValueError(f"Unexpected frame timestamps at {path}")


def compare_x(reader, master, candidate, before=None):
    import numpy as np

    before, original_metadata = x_pixels(reader, master) if before is None else before
    after, metadata = x_pixels(reader, candidate)
    if original_metadata != metadata:
        raise ValueError(f"Evaluation video metadata differs for {candidate}")
    if not np.array_equal(before, after):
        for i, (a, b) in enumerate(zip(before, after, strict=True)):
            if not np.array_equal(a, b):
                changed = int(np.count_nonzero(a != b))
                raise ValueError(f"{candidate} frame {i} has {changed} changed RGB components")
    verify_timeline(master, master=True)
    verify_timeline(candidate, master=False)
    return {
        "processor_input_rgb_sha256": hashlib.sha256(before).hexdigest(),
        "frames": len(before),
        "shape": list(before.shape),
        "dtype": before.dtype.name,
        "fps": metadata["fps"],
        "max_abs_code_difference": 0,
        "reader_metadata_equal": True,
    }


def _verify_pair(job):
    from .release import payload_path
    from .data import file_sha

    master_root, mp4_root, row = job
    scope = "/".join(row["master_path"].split("/")[:2])
    master = master_root / payload_path(scope, row["master_path"])
    mp4 = mp4_root / payload_path(scope, row["mp4_path"])
    for path in (master, mp4):
        if not path.is_file():
            raise ValueError(f"Missing media for {row['sample_id']}: {path}")
    if file_sha(master) != row["master_sha256"] or file_sha(mp4) != row["mp4_sha256"]:
        raise ValueError(f"Media SHA-256 differs for {row['sample_id']}")
    proof = compare_x(x_reader(), master, mp4)
    if any(proof[key] != row[key] for key in proof):
        raise ValueError(f"Evaluation receipt differs for {row['sample_id']}")
    return proof["frames"]


def main(argv=None):
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    from .catalog import INDEX_SHA
    from .data import file_sha

    index_path = args.data_root / "dataset-index.json"
    if not index_path.is_file() or file_sha(index_path) != INDEX_SHA:
        parser.error("Use the pinned MP4 dataset index in --data-root")
    entry = json.loads(index_path.read_bytes())["media_equivalence"]
    manifest = args.manifest or args.data_root / entry["manifest"]
    if not manifest.is_file() or file_sha(manifest) != entry["manifest_sha256"]:
        parser.error("Provide media-equivalence.jsonl from the pinned MP4 release")
    with manifest.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    by_id = {row["sample_id"]: row for row in rows}
    if len(rows) != len(by_id) or not rows or set(args.sample_id) - by_id.keys():
        parser.error("Invalid manifest or unknown sample ID")
    jobs = [
        (args.master_root, args.data_root, by_id[sid])
        for sid in (dict.fromkeys(args.sample_id) if args.sample_id else by_id)
    ]
    if args.workers == 1:
        frames = sum(map(_verify_pair, jobs))
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
            frames = sum(pool.map(_verify_pair, jobs))
    print(json.dumps({
        "status": "EXACT_EVALUATION_RGB8",
        "verified_videos": len(jobs),
        "verified_frames": frames,
        "max_abs_code_difference": 0,
    }, indent=2))
    return 0
