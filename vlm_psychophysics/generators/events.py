"""T5 single and dual events from the frozen trial manifest."""

from fractions import Fraction
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time
import av
import numpy as np
from PIL import Image


PROTOCOL = "t5-paired-v1"


WIDTH, HEIGHT, FPS, FRAMES = 768, 512, 60, 180


CENTERS = ((160, 256), (608, 256), (384, 128), (384, 384))


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def value_sha(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def rgb8(linear):
    linear = np.clip(linear, 0, 1)
    encoded = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * linear ** (1 / 2.4) - 0.055,
    )
    return np.rint(255 * encoded).astype(np.uint8)


def linearize(encoded):
    return np.where(
        encoded <= 0.04045,
        encoded / 12.92,
        ((encoded + 0.055) / 1.055) ** 2.4,
    )


@lru_cache(maxsize=64)
def load_array(path):
    return np.load(path)


def frame_images(row):
    """Render the fixed scene and the exact visible state of each event."""
    scene = row["scene"]
    background = (
        np.full((HEIGHT, WIDTH, 3), 0.10)
        if scene is None
        else load_array(scene["background_prepared"]["path"])
    )
    base = rgb8(background)
    for x, y in CENTERS:
        base[y - 80 : y + 80, x - 80 : x - 78] = 115
        base[y - 80 : y + 80, x + 78 : x + 80] = 115
        base[y - 80 : y - 78, x - 80 : x + 80] = 115
        base[y + 78 : y + 80, x - 80 : x + 80] = 115
    base[255:257, 379:390] = 170
    base[251:262, 383:385] = 170
    if scene is None:
        canonical = np.full((128, 128, 3), int(rgb8(np.array(0.10))), np.uint8)
        canonical[:, :28] = int(rgb8(np.array(0.82)))
        canonical[-28:, :] = int(rgb8(np.array(0.82)))
    else:
        canonical = load_array(scene["object_prepared"]["path"])
    images = [base]
    schedule = np.zeros(FRAMES, dtype=np.int16)
    for index, event in enumerate(row["events"], start=1):
        start = event["onset_frame"]
        assert 0 <= start and start + 30 <= FRAMES
        assert not schedule[start : start + 30].any()
        sprite = np.rot90(canonical, -event["angle_deg"] // 90)
        frame = base.copy()
        x, y = CENTERS[event["position_index"]]
        roi = frame[y - 64 : y + 64, x - 64 : x + 64]
        if scene is None:
            roi[:] = sprite
        else:
            back = linearize(roi.astype(np.float64) / 255)
            roi[:] = rgb8(sprite[:, :, :3] + (1 - sprite[:, :, 3, None]) * back)
        images.append(frame)
        schedule[start : start + 30] = index
    return images, schedule


def render_one(payload):
    root, row = payload
    started = time.monotonic()
    video = Path(row["video_path"])
    qa_path = video.parent.parent / "qa" / (row["sample_id"] + ".json")
    if qa_path.exists():
        qa = json.loads(qa_path.read_text())
        assert qa["row_sha256"] == value_sha(row)
        assert file_sha(video) == qa["video_sha256"]
        return qa
    assert not video.exists(), f"Unverified existing video: {video}"
    images, schedule = frame_images(row)
    hashes = [hashlib.sha256(frame.tobytes()).hexdigest() for frame in images]
    temporary = video.with_suffix(".partial.mp4")
    assert not temporary.exists(), f"Preserved partial video: {temporary}"
    stream_hash = hashlib.sha256()
    with av.open(str(temporary), "w") as container:
        stream = container.add_stream("libx264rgb", rate=FPS)
        stream.width, stream.height = WIDTH, HEIGHT
        stream.pix_fmt = "rgb24"
        stream.codec_context.thread_count = 1
        stream.options = {"crf": "0", "preset": "ultrafast", "threads": "1"}
        for index, state in enumerate(schedule):
            rgb = images[state]
            stream_hash.update(rgb.tobytes())
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, FPS)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    pts = []
    with av.open(str(temporary)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        assert stream.codec_context.pix_fmt == "gbrp"
        assert stream.average_rate == FPS
        for index, frame in enumerate(container.decode(video=0)):
            assert index < FRAMES
            actual = frame.to_ndarray(format="rgb24")
            assert (
                hashlib.sha256(actual.tobytes()).hexdigest() == hashes[schedule[index]]
            )
            pts.append(float(frame.pts * frame.time_base))
            assert abs(pts[-1] - index / FPS) <= 1e-6
    assert len(pts) == FRAMES
    temporary.replace(video)
    inspection = video.parent.parent / "inspection" / row["sample_id"]
    inspection.mkdir(exist_ok=True)
    inspection_paths = [row["reference_path"]]
    for event in sorted(row["events"], key=lambda event: event["onset_frame"]):
        index = event["onset_frame"] + 15
        path = inspection / f"frame{index:03d}.png"
        Image.fromarray(images[schedule[index]]).save(path)
        inspection_paths.append(str(path))
    qa = {
        "status": "PASS_ENCODER_ALL180_RGB8_PTS",
        "sample_id": row["sample_id"],
        "row_sha256": value_sha(row),
        "video_sha256": file_sha(video),
        "stream_sha256": stream_hash.hexdigest(),
        "frame_sha256": [hashes[state] for state in schedule],
        "frames": FRAMES,
        "max_PTS_error_seconds": max(
            abs(value - index / FPS) for index, value in enumerate(pts)
        ),
        "inspection_paths": inspection_paths,
        "implementation_sha256": file_sha(__file__),
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(qa_path, qa)
    return qa
