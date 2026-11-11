"""Native image preparation and exact input accounting."""

from __future__ import annotations

from contextlib import ExitStack, nullcontext
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import time

from . import data, observers as base


def require(condition, message):
    if not condition:
        raise ValueError(message)


def descriptor(path):
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def image_entries(item):
    return [entry for entry in item["content"] if entry["type"] == "image"]


def native_kwargs(profile):
    return profile.get("request_mm_processor_kwargs", {})


def sha(value):
    return hashlib.sha256(value).hexdigest()


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


def tensor_sha(value):
    import torch

    return sha(
        value.detach()
        .cpu()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )


def official_source(path, distribution):
    """Verify unmodified installed source against its own wheel RECORD."""
    import base64
    import importlib.metadata

    path = Path(path).resolve()
    dist = importlib.metadata.distribution(distribution)
    entry = next(
        (x for x in dist.files if Path(dist.locate_file(x)).resolve() == path), None
    )
    require(
        entry is not None and entry.hash is not None and entry.hash.mode == "sha256",
        "Installed official source lacks wheel RECORD: " + str(path),
    )
    encoded = (
        base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest())
        .decode()
        .rstrip("=")
    )
    require(
        encoded == entry.hash.value,
        "Installed official source differs from wheel RECORD",
    )
    return {
        **descriptor(path),
        "distribution": distribution,
        "version": dist.version,
        "wheel_record_sha256": entry.hash.value,
    }


def processor_sources(native, model):
    sources = [official_source(inspect.getsourcefile(type(native)), "vllm")]
    if model == "nemo":
        from vllm.transformers_utils.processors import nano_nemotron_vl

        sources.append(official_source(nano_nemotron_vl.__file__, "vllm"))
    if model == "gemma":
        from transformers.models.gemma4 import (
            image_processing_gemma4,
            image_processing_pil_gemma4,
            processing_gemma4,
        )

        for module in (
            image_processing_gemma4,
            image_processing_pil_gemma4,
            processing_gemma4,
        ):
            sources.append(official_source(module.__file__, "transformers"))
    return sources


class NoSpatialTransform:
    """Fail before processor resize/crop; guard is local to the CPU process."""

    def __enter__(self):
        self.saved = []
        from PIL import Image
        import torch.nn.functional as tf

        modules = [(Image.Image, ("resize", "crop")), (tf, ("interpolate",))]
        for name in (
            "torchvision.transforms.functional",
            "torchvision.transforms.v2.functional",
            "transformers.image_transforms",
        ):
            module = importlib.import_module(name)
            modules.append(
                (
                    module,
                    (
                        "resize",
                        "crop",
                        "center_crop",
                        "resized_crop",
                        "five_crop",
                        "ten_crop",
                    ),
                )
            )

        def forbidden(*args, **kwargs):
            raise ValueError(
                "Input resize/crop/interpolation was attempted; no model call may qualify"
            )

        for module, names in modules:
            for name in names:
                if hasattr(module, name):
                    self.saved.append((module, name, getattr(module, name)))
                    setattr(module, name, forbidden)
        return self

    def __exit__(self, *_):
        for module, name, original in reversed(self.saved):
            setattr(module, name, original)


def load_images(item, verify_video=False):
    import numpy as np
    from PIL import Image
    from vllm.multimodal.media.image import ImageMediaIO

    images, receipts = [], []
    frames = None
    if verify_video:
        from vllm.multimodal.media.video import VideoMediaIO

        source = item["original_source"]
        require(
            file_sha256(source["path"]) == source["sha256"],
            "Original video bytes changed",
        )
        frames, metadata = (
            VideoMediaIO(ImageMediaIO(), video_backend="opencv", num_frames=180, fps=-1)
            .load_file(Path(source["path"]))
            .media
        )
        require(
            frames.dtype == np.uint8
            and list(frames.shape) == [180, 512, 768, 3]
            and metadata["frames_indices"] == list(range(180)),
            "Video source decode/order differs",
        )
    for entry in image_entries(item):
        width, height = data.image_size(entry)
        index = entry["frame_index"]
        require(
            file_sha256(entry["path"]) == entry["sha256"], "Source PNG bytes changed"
        )
        with Image.open(entry["path"]) as png:
            require(
                png.mode == "RGB" and png.size == (width, height) and not png.info,
                "PNG must be untagged RGB8 at native resolution",
            )
        image = ImageMediaIO().load_file(Path(entry["path"])).media
        pixels = np.asarray(image)
        require(
            image.mode == "RGB"
            and pixels.dtype == np.uint8
            and list(pixels.shape) == [height, width, 3],
            "Official image reader changed mode/dtype/size",
        )
        checksum = sha(pixels.tobytes())
        require(checksum == entry["pixel_sha256"], "PNG reader pixels changed")
        matches_video = frames is not None and index is not None
        if matches_video:
            require(
                np.array_equal(pixels[0, 0], frames[0, 0, 0]),
                "Image corner padding color differs from video frame-zero padding",
            )
            require(
                np.array_equal(pixels, frames[index]),
                "PNG pixels differ from corresponding official video RGB8 frame",
            )
        images.append(image)
        receipts.append(
            {
                "path": entry["path"],
                "frame_index": index,
                "shape": list(pixels.shape),
                "dtype": str(pixels.dtype),
                "pixel_sha256": checksum,
                "matches_original_video_RGB8": matches_video,
                "video_padding_RGB": (
                    frames[0, 0, 0].tolist() if matches_video else None
                ),
            }
        )
    return images, receipts


def chunks(value, model):
    """Return complete per-image views at the documented native vision boundary."""
    shape = list(value.shape)
    if model in ("cosmos", "qwen"):
        require(
            value.dtype.__str__() == "torch.uint8"
            and len(shape) == 2
            and shape[1] == 1536
            and (shape[0] == 100 or shape[0] % 1536 == 0),
            "Native raw image patch geometry differs",
        )
        return list(value.split(1536, dim=0))
    if model == "nemo":
        if shape in ([3, 512, 512], [3, 512, 768]):
            from vllm.transformers_utils.processors.nano_nemotron_vl import (
                DynamicResolutionImageTiler,
            )

            return [DynamicResolutionImageTiler.stack([value], 16)[0]]
        if (
            len(shape) == 3
            and shape[0] == 1
            and shape[2] == 768
            and (shape[1] == 1024 or shape[1] % 1536 == 0)
        ):
            return list(value[0].split(1536, dim=0))
        require(
            len(shape) == 4 and shape[1:] in ([3, 512, 512], [3, 512, 768]),
            "Nemo native image tensor resized/cropped",
        )
        from vllm.transformers_utils.processors.nano_nemotron_vl import (
            DynamicResolutionImageTiler,
        )

        return [DynamicResolutionImageTiler.stack([image], 16)[0] for image in value]
    if shape in ([144, 768], [1260, 768], [1584, 768], [2520, 768]):
        return [value]
    require(
        len(shape) == 3
        and shape[1:] in ([144, 768], [1260, 768], [1584, 768], [2520, 768]),
        "Gemma valid patch count differs",
    )
    return list(value.unbind(0))


def payload_description(value, model):
    import torch

    if isinstance(value, torch.Tensor):
        d = base.tensor_description(value, True)
        if value.ndim >= 2:
            shape = list(value.shape)
            is_pixels = (
                (
                    model in ("cosmos", "qwen")
                    and value.dtype == torch.uint8
                    and shape[-1] == 1536
                )
                or (
                    model == "nemo"
                    and (
                        shape[-3:] in ([3, 512, 512], [3, 512, 768])
                        or (
                            len(shape) == 3
                            and shape[0] == 1
                            and shape[-1] == 768
                            and (shape[1] == 1024 or shape[1] % 1536 == 0)
                        )
                    )
                )
                or (
                    model == "gemma"
                    and shape[-2:]
                    in ([144, 768], [1260, 768], [1584, 768], [2520, 768])
                )
            )
            if is_pixels:
                d["per_image_sha256"] = [tensor_sha(x) for x in chunks(value, model)]
            if model == "gemma" and shape[-2:] in (
                [144, 2],
                [1260, 2],
                [1584, 2],
                [2520, 2],
            ):
                d["per_image_position_sha256"] = [
                    tensor_sha(x)
                    for x in (value.unbind(0) if value.ndim == 3 else [value])
                ]
        return d
    if isinstance(value, (list, tuple)):
        return [payload_description(x, model) for x in value]
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(k): payload_description(v, model) for k, v in value.items()}
    if hasattr(value, "data"):
        return payload_description(value.data, model)
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    raise ValueError("Unknown native processor payload type: " + str(type(value)))


def leaves(value):
    if isinstance(value, dict):
        if "shape" in value and "dtype" in value:
            yield value
        else:
            for x in value.values():
                yield from leaves(x)
    elif isinstance(value, list):
        for x in value:
            yield from leaves(x)


def ordered_hashes(description, field="per_image_sha256"):
    return [
        checksum for value in leaves(description) for checksum in value.get(field, [])
    ]


def placeholders(value):
    result = {}
    for modality, rows in value.items():
        result[modality] = []
        for row in rows:
            entry = {"offset": int(row.offset), "length": int(row.length)}
            if getattr(row, "is_embed", None) is not None:
                mask = row.is_embed
                entry["is_embed"] = (
                    mask.tolist() if hasattr(mask, "tolist") else list(mask)
                )
            result[modality].append(entry)
    return result


def template(processor, item):
    content = [{"type": "image"} for _ in image_entries(item)] + [
        {"type": "text", "text": item["row"]["prompt"]}
    ]
    text = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    require(not text.rstrip().endswith("<think>"), "Thinking unexpectedly enabled")
    return text


class NemoNoResize:
    """Skip only the official image tiler's identity-sized interpolation."""

    def __init__(self, verify_equivalence=False):
        from vllm.transformers_utils.processors import nano_nemotron_vl as module

        self.module = module
        self.original = module.DynamicResolutionImageTiler.apply_params
        self.verify_equivalence = verify_equivalence
        self.receipts = []
        owner = self

        def apply(tiler, params, dtype=None):
            import torch

            dtype = torch.float32 if dtype is None else dtype
            target = (
                params.patch_size[1] * tiler._patch_size,
                params.patch_size[0] * tiler._patch_size,
            )
            require(
                params.media.mode == "RGB"
                and params.media.size == (768, 512)
                and target == (512, 768)
                and params.patch_size == (48, 32)
                and tiler._patch_size == 16
                and params.num_tiles == 1
                and params.num_embeddings == 384,
                "Nemo official dynamic image plan would resize/crop/tile; HOLD",
            )
            source = module._pil_to_nhwc_tensor(params.media)
            require(
                source.dtype == torch.uint8 and list(source.shape) == [1, 512, 768, 3],
                "Nemo official raw NHWC conversion differs",
            )
            actual = module._bicubic_resize_and_normalize(
                source,
                size=None,
                norm_mean=tiler.norm_mean,
                norm_std=tiler.norm_std,
                dtype=dtype,
            )
            equal = None
            if owner.verify_equivalence:
                # CPU-only independent check of the unmodified same-size path.
                # Run outside NoSpatialTransform, then require exact bytes.
                expected = owner.original(tiler, params, dtype=dtype)
                equal = len(expected) == 1 and torch.equal(expected[0], actual[0])
                require(
                    equal,
                    "Nemo skipped identity resize changes official normalized bytes; HOLD",
                )
            owner.receipts.append(
                {
                    "source_RGB8_sha256": tensor_sha(source),
                    "target_hw": list(target),
                    "patch_grid_hw": [32, 48],
                    "patches": 1536,
                    "image_tokens": 384,
                    "dtype": str(actual.dtype),
                    "normalized_sha256": tensor_sha(actual[0]),
                    "official_same_size_interpolation_bytes_equal": equal,
                    "actual_interpolation_performed": False,
                }
            )
            return list(actual)

        module.DynamicResolutionImageTiler.apply_params = apply

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.module.DynamicResolutionImageTiler.apply_params = self.original

    def check_reference(self, native, images):
        """Compare official identity interpolation before activating hard guard."""
        processor = native.info.get_hf_processor(**native_kwargs({"model": "nemo"}))
        tiler = processor.dynamic_tiler
        require(
            tiler is not None,
            "Nemo image path must use its native dynamic-resolution tiler",
        )
        params = tiler.compute_params(images, tiler.max_num_tokens_available(1024))
        require(len(params) == len(images), "Nemo reference plan lost an image")
        self.verify_equivalence = True
        try:
            for param in params:
                tiler.apply_params(param, dtype=processor.dtype)
        finally:
            self.verify_equivalence = False
        return list(self.receipts)


def pad(values, unit=48):
    """Extend bottom/right only, with the source first frame's background RGB."""
    import numpy as np

    video = values.ndim == 4
    h, w = values.shape[-3:-1]
    bottom, right = (-h) % unit, (-w) % unit
    background = values[0, 0, 0] if video else values[0, 0]
    shape = (
        (values.shape[0], h + bottom, w + right, 3)
        if video
        else (h + bottom, w + right, 3)
    )
    output = np.empty(shape, dtype=values.dtype)
    output[...] = background
    if video:
        output[:, :h, :w] = values
    else:
        output[:h, :w] = values
    return output, {
        "source_wh": [w, h],
        "submitted_wh": [w + right, h + bottom],
        "bottom_rows": bottom,
        "right_columns": right,
        "padding_RGB": background.tolist(),
        "effective_pixels_resized_or_cropped": False,
    }


class GemmaAdapter:
    """Keep native HF processing with padding and masked-tail compaction."""

    def __init__(self, processor_checkpoint, modality):
        import vllm.model_executor.models.gemma4_mm as gemma
        from transformers import AutoProcessor

        self.module = gemma
        self.modality = modality
        self.processor = AutoProcessor.from_pretrained(
            processor_checkpoint, local_files_only=True, trust_remote_code=False
        )
        self.original_budget = gemma._VIDEO_MAX_SOFT_TOKENS
        self.original_count = gemma.Gemma4ProcessingInfo._compute_num_soft_tokens
        self.processor_class = gemma.Gemma4MultiModalProcessor
        self.original_apply = self.processor_class._apply_hf_processor_main
        self.compaction_calls = 0
        owner = self

        def count(info, image_width, image_height, max_soft_tokens=None):
            if info.ctx.get_merged_mm_kwargs({}).get("do_resize") is False:
                require(
                    image_width % 48 == image_height % 48 == 0,
                    "Gemma no-resize canvas must align to its patch/pool grid",
                )
                return image_width * image_height // (48 * 48)
            return owner.original_count(
                info, image_width, image_height, max_soft_tokens
            )

        def compact(native, mm_items, hf_kwargs):
            import torch

            result = owner.original_apply(native, mm_items, hf_kwargs)
            pixel_key = (
                "pixel_values_videos" if owner.modality == "video" else "pixel_values"
            )
            position_key = (
                "pixel_position_ids_videos"
                if owner.modality == "video"
                else "pixel_position_ids"
            )
            if pixel_key not in result:
                return result
            pixels = result[pixel_key]
            positions = result[position_key]
            tensor_layout = isinstance(pixels, torch.Tensor)
            pixel_groups = [pixels] if tensor_layout else pixels
            position_groups = [positions] if tensor_layout else positions
            compact_pixels, compact_positions = [], []
            for values, coordinates in zip(pixel_groups, position_groups, strict=True):
                require(
                    list(values.shape[1:]) == [2520, 768]
                    and list(coordinates.shape) == [values.shape[0], 2520, 2],
                    "Gemma native patch budget or geometry changed",
                )
                valid = (coordinates != -1).any(-1)
                require(
                    (valid.sum(-1) == 1584).all()
                    and valid[:, :1584].all()
                    and not valid[:, 1584:].any(),
                    "Gemma compaction would remove effective image content",
                )
                compact_pixels.append(values[:, :1584].contiguous())
                compact_positions.append(coordinates[:, :1584].contiguous())
            result[pixel_key] = compact_pixels[0] if tensor_layout else compact_pixels
            result[position_key] = (
                compact_positions[0] if tensor_layout else compact_positions
            )
            owner.compaction_calls += 1
            return result

        gemma._VIDEO_MAX_SOFT_TOKENS = 280
        gemma.Gemma4ProcessingInfo._compute_num_soft_tokens = count
        self.processor_class._apply_hf_processor_main = compact

    def prepare(self, case):
        import numpy as np
        from PIL import Image
        from vllm.multimodal.media.image import ImageMediaIO
        from vllm.multimodal.media.video import VideoMediaIO

        padding, content, media = [], [], {}
        if case.get("images"):
            images, _ = load_images(case_item(case))
            padded_images = []
            for image in images:
                pixels, geometry = pad(np.asarray(image))
                padded_images.append(Image.fromarray(pixels))
                padding.append(geometry)
            media["image"] = padded_images
            content.extend({"type": "image"} for _ in images)
        if self.modality == "video":
            reader = VideoMediaIO(
                ImageMediaIO(), video_backend="opencv", num_frames=180, fps=-1
            )
            frames, metadata = reader.load_file(Path(case["video_path"])).media
            require(
                frames.dtype == np.uint8
                and frames.shape == (180, 512, 768, 3)
                and metadata["frames_indices"] == list(range(180)),
                "Gemma source video frame order or geometry changed",
            )
            padded, geometry = pad(frames)
            padding.append(geometry)
            media["video"] = [(padded, metadata)]
            content.append({"type": "video"})
        content.append({"type": "text", "text": case["prompt"]})
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        require(not text.rstrip().endswith("<think>"), "Thinking unexpectedly enabled")
        return {
            "prompt": text,
            "multi_modal_data": media,
            "mm_processor_kwargs": {
                "do_resize": False,
                "do_sample_frames": False,
                "max_soft_tokens": (
                    140
                    if self.modality == "video" and "reference_version" not in case
                    else 280
                ),
            },
        }, padding

    def close(self):
        self.processor_class._apply_hf_processor_main = self.original_apply
        self.module.Gemma4ProcessingInfo._compute_num_soft_tokens = self.original_count
        self.module._VIDEO_MAX_SOFT_TOKENS = self.original_budget


def case_item(case):
    """Only prompt and ordered media enter the native request."""
    content = [{"type": "image", **image} for image in case.get("images", [])]
    if case["modality"] == "video":
        content.append(
            {
                "type": "video",
                "path": case["video_path"],
                "sha256": case["video_sha256"],
            }
        )
    content.append({"type": "text", "text": case["prompt"]})
    return {
        "sample_id": case["sample_id"],
        "task": case["task"],
        "row": {"prompt": case["prompt"]},
        "content": content,
    }


def chat_messages(case):
    content = []
    for entry in case_item(case)["content"]:
        if entry["type"] == "text":
            content.append({"type": "text", "text": entry["text"]})
        else:
            kind = entry["type"] + "_url"
            content.append({"type": kind, kind: {"url": Path(entry["path"]).as_uri()}})
    return [{"role": "user", "content": content}]


def cpu_reference(profile, case, prepared=None):
    """Expand one representative through the native CPU multimodal processor."""
    from vllm.config import ModelConfig
    from vllm.tokenizers import cached_tokenizer_from_config
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.processing.inputs import ProcessorInputs
    from vllm.multimodal.processing.context import TimingContext

    engine = profile["engine_kwargs"]
    allowed = (
        "dtype",
        "trust_remote_code",
        "hf_overrides",
        "max_model_len",
        "limit_mm_per_prompt",
        "mm_processor_cache_gb",
        "mm_processor_kwargs",
    )
    model_config = ModelConfig(
        model=engine["model"],
        tokenizer=engine.get("tokenizer", engine["model"]),
        allowed_local_media_path=engine["allowed_local_media_path"],
        **{name: engine[name] for name in allowed if name in engine},
    )
    tokenizer = cached_tokenizer_from_config(model_config)
    native = MULTIMODAL_REGISTRY.create_processor(model_config, tokenizer=tokenizer)
    request_kwargs = profile["request_mm_processor_kwargs"]
    if prepared is not None:
        text = prepared["prompt"]
        media = prepared["multi_modal_data"]
        request_kwargs = prepared["mm_processor_kwargs"]
    else:
        from vllm.renderers.chat_utils import parse_chat_messages
        from vllm.renderers.hf import (
            resolve_chat_template_content_format,
            safe_apply_chat_template,
        )

        content_format = resolve_chat_template_content_format(
            chat_template=None,
            tools=None,
            given_format="auto",
            tokenizer=tokenizer,
            model_config=model_config,
        )
        conversation, media, _ = parse_chat_messages(
            chat_messages(case),
            model_config,
            content_format=content_format,
            media_io_kwargs=profile["media_io_kwargs"],
            mm_processor_kwargs=request_kwargs,
        )
        text = safe_apply_chat_template(
            model_config,
            tokenizer,
            conversation,
            chat_template=None,
            tokenize=False,
            add_generation_prompt=True,
            continue_final_message=False,
            tools=None,
            enable_thinking=False,
        )
    require(
        isinstance(text, str) and not text.rstrip().endswith("<think>"),
        "Native text template or thinking differs",
    )
    if case["modality"] == "static" and prepared is None:
        media = {"image": load_images(case_item(case))[0]}
    if case["modality"] == "static" and profile["model"] == "nemo":
        profile["_nemo_adapter"].check_reference(native, media["image"])
    guard = NoSpatialTransform() if case["modality"] == "static" else nullcontext()
    with guard:
        output = native.apply(
            ProcessorInputs(
                prompt=tokenizer.encode(text, add_special_tokens=False),
                mm_data_items=native.info.parse_mm_data(media),
                hf_processor_mm_kwargs=request_kwargs,
            ),
            TimingContext(enabled=False),
        )
    prompt_ids = list(output["prompt_token_ids"])
    require(
        len(prompt_ids) + 16384 <= engine["max_model_len"],
        "Expanded input plus generation budget exceeds the configured context",
    )
    payload = payload_description(output["mm_kwargs"], profile["model"])
    return {
        "prompt_token_ids": prompt_ids,
        "prompt_sha256": sha(case["prompt"].encode("utf-8")),
        "template_sha256": sha(text.encode("utf-8")),
        "mm_placeholders": placeholders(output["mm_placeholders"]),
        "native_payload": payload,
        "processor_class": type(native).__module__ + "." + type(native).__name__,
        "processor_sources": processor_sources(native, profile["model"]),
    }, type(native)
