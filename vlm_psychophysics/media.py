"""Lossless video fixture writer for API input-contract tests."""
from fractions import Fraction


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
