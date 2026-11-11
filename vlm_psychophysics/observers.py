"""Read-only native processor and GPU-boundary observations."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import inspect
import os
from pathlib import Path
import time

from .data import image_size


def require(condition, message):
    if not condition:
        raise ValueError(message)


def descriptor(path):
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def tensor_description(value, deep=False):
    import torch

    if isinstance(value, torch.Tensor):
        d = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
        if deep:
            d["sha256"] = hashlib.sha256(
                value.detach()
                .cpu()
                .contiguous()
                .reshape(-1)
                .view(torch.uint8)
                .numpy()
                .tobytes()
            ).hexdigest()
        if not value.is_floating_point() and value.numel() <= 600:
            d["values"] = value.tolist()
        return d
    if isinstance(value, (list, tuple)):
        return [tensor_description(x, deep) for x in value[:8]]
    return None


def first_tensor(args, kwargs):
    import torch

    def search(x):
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, (list, tuple)):
            for v in x:
                r = search(v)
                if r is not None:
                    return r
        if isinstance(x, dict):
            for v in x.values():
                r = search(v)
                if r is not None:
                    return r
        return None

    return search(args) if args else search(kwargs)


def qualified_name(value):
    cls = value if inspect.isclass(value) else type(value)
    return cls.__module__ + "." + cls.__name__


def resolve_module(model, paths):
    for path in paths:
        value = model
        try:
            for segment in path.split("."):
                value = (
                    value[int(segment)]
                    if segment.isdigit()
                    else getattr(value, segment)
                )
        except (AttributeError, IndexError, KeyError, TypeError):
            continue
        if hasattr(value, "register_forward_pre_hook"):
            return path, value
    raise ValueError("No actual vision module matches " + repr(paths))


def worker_identity(worker, model):
    import torch, vllm

    counts = Counter()
    backends = Counter()
    examples = defaultdict(list)
    for name, module in worker.get_model().named_modules():
        method = getattr(module, "quant_method", None)
        if method is None:
            continue
        counts["method:" + qualified_name(method)] += 1
        for k, v in vars(method).items():
            if k.endswith("backend"):
                backends[k + ":" + str(v)] += 1
            if "kernel" in k or "linear" in k:
                label = qualified_name(v)
                if label.startswith(("vllm.", "flashinfer.")):
                    counts[k + ":" + label] += 1
                    if len(examples[label]) < 2:
                        examples[label].append(name)
    runner = worker.model_runner
    config = worker.vllm_config
    from vllm.v1.core.kv_cache_utils import get_kv_cache_capacity

    capacity, concurrency = get_kv_cache_capacity(config, runner.kv_cache_config)
    result = {
        "pid": os.getpid(),
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "model_class": qualified_name(worker.get_model()),
        "runner_class": qualified_name(runner),
        "kernel_config": repr(config.kernel_config),
        "method_kernel_counts": dict(counts),
        "actual_backends": dict(backends),
        "representative_modules": dict(examples),
        "actual_kv_capacity": int(capacity),
        "kv_concurrency": concurrency,
        "mm_device_do_normalize": config.model_config.get_multimodal_config().mm_device_do_normalize,
    }
    if model == "qwen":
        from vlm_psychophysics.runtime.qsa import evidence

        result["custom_qsa_selector"] = evidence()
    return result


def install_probe(worker, spec, model):
    import torch

    network = worker.get_model()
    state = {"outer": [], "normalized": [], "handles": [], "active_outer": None}
    worker._official_x_probe = state
    outer_path, outer = resolve_module(network, spec["outer_paths"])
    normalized_path, normalized = resolve_module(network, spec["normalized_paths"])
    state.update(outer_path=outer_path, normalized_path=normalized_path)

    def outer_hook(module, args, kwargs):
        grid = kwargs.get("grid_thw")
        if isinstance(grid, torch.Tensor):
            grid = grid.tolist()
        state["active_outer"] = len(state["outer"])
        state["outer"].append(
            {
                "tensor": tensor_description(first_tensor(args, kwargs), True),
                "grid_thw": grid,
                "num_frames_argument": kwargs.get("num_frames"),
            }
        )

    def normalized_hook(module, args, kwargs):
        state["normalized"].append(
            {
                "tensor": tensor_description(first_tensor(args, kwargs), True),
                "outer_index": state["active_outer"],
            }
        )

    state["handles"].append(
        outer.register_forward_pre_hook(outer_hook, with_kwargs=True)
    )
    state["handles"].append(
        normalized.register_forward_pre_hook(normalized_hook, with_kwargs=True)
    )
    if model == "qwen":
        state["profiler"] = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        )
        state["profiler"].__enter__()
    norm = getattr(outer, "input_norm", None)
    state["normalizer"] = {
        "class": qualified_name(norm),
        "input_dtype": str(getattr(norm, "input_dtype", None)),
        "source": (
            descriptor(inspect.getsourcefile(type(norm)))
            if norm is not None and inspect.getsourcefile(type(norm))
            else None
        ),
        "weight": (
            getattr(norm, "weight", torch.empty(0)).detach().cpu().tolist()
            if norm is not None
            else None
        ),
        "bias": (
            getattr(norm, "bias", torch.empty(0)).detach().cpu().tolist()
            if norm is not None
            else None
        ),
    }
    return {
        "outer_path": outer_path,
        "normalized_path": normalized_path,
        "normalizer": state["normalizer"],
    }


def read_probe(worker, model):
    state = worker._official_x_probe
    for handle in state.pop("handles"):
        handle.remove()
    profiler = state.pop("profiler", None)
    if profiler:
        try:
            profiler.__exit__(None, None, None)
            state["actual_topk_operator_calls"] = [
                {"key": x.key, "count": x.count}
                for x in profiler.key_averages()
                if "persistent_topk" in x.key or "cooperative_topk" in x.key
            ]
        except BaseException as error:
            state["profiler_error"] = repr(error)
    try:
        state["runtime"] = worker_identity(worker, model)
    except BaseException as error:
        state["runtime_error"] = repr(error)
    del worker._official_x_probe
    return state


class NativeObserver:
    def __init__(self, p):
        self.p = p
        self.active = False
        self.deep = False
        self.contract = None
        self.clear()

    def clear(self):
        self.reads = []
        self.processors = []
        self.read_seconds = 0.0
        self.processor_seconds = 0.0

    def describe(self, data):
        return {
            k: tensor_description(v, self.deep)
            for k, v in data.items()
            if "pixel" in k
            or "grid" in k
            or k in ("video_num_patches", "video_frame_counts")
        }

    def __enter__(self):
        from vllm.multimodal.media.video import VideoMediaIO
        from vllm.multimodal.processing.context import InputProcessingContext

        self.original_read = VideoMediaIO.load_file
        self.original_call = InputProcessingContext.call_hf_processor
        self.original_post = InputProcessingContext._postprocess_output
        obs = self

        def load(loader, path):
            t = time.monotonic()
            result = obs.original_read(loader, path)
            if obs.active:
                frames, meta = result.media
                c = obs.contract
                require(
                    str(frames.dtype) == "uint8"
                    and list(frames.shape) == [180, c["height"], c["width"], 3]
                    and meta["frames_indices"] == list(range(180)),
                    "Actual native video count/order/geometry differs",
                )
                d = {
                    "path": str(path),
                    "shape": list(frames.shape),
                    "dtype": str(frames.dtype),
                    "metadata": meta,
                }
                if obs.deep:
                    d["decoded_sha256"] = hashlib.sha256(frames.tobytes()).hexdigest()
                    d["per_frame_sha256"] = [
                        hashlib.sha256(x.tobytes()).hexdigest() for x in frames
                    ]
                obs.reads.append(d)
                obs.read_seconds += time.monotonic() - t
            return result

        def call(ctx, hf_processor, data, kwargs={}):
            t = time.monotonic()
            result = obs.original_call(ctx, hf_processor, data, kwargs)
            if obs.active:
                obs.processor_seconds += time.monotonic() - t
            return result

        def post(ctx, output):
            before = obs.describe(output) if obs.active else None
            result = obs.original_post(ctx, output)
            if obs.active:
                obs.processors.append(
                    {
                        "before_native_dtype": before,
                        "after_native_dtype": obs.describe(result),
                    }
                )
            return result

        VideoMediaIO.load_file = load
        InputProcessingContext.call_hf_processor = call
        InputProcessingContext._postprocess_output = post
        return self

    def __exit__(self, *_):
        from vllm.multimodal.media.video import VideoMediaIO
        from vllm.multimodal.processing.context import InputProcessingContext

        VideoMediaIO.load_file = self.original_read
        InputProcessingContext.call_hf_processor = self.original_call
        InputProcessingContext._postprocess_output = self.original_post


class InputObserver(NativeObserver):
    """Record each reader/application, and full tensor identities on representatives."""

    def __init__(self, model, processor_class, case, deep):
        super().__init__({})
        self.model = model
        self.processor_class = processor_class
        self.case = case
        self.deep = deep
        self.contract = {"width": 768, "height": 512}
        self.applications = []
        self.lookup = {str(Path(row["path"])): row for row in case.get("images", [])}

    def describe(self, data):
        from .inputs import payload_description

        return {
            key: (
                payload_description(value, self.model)
                if self.deep
                else describe_tree(value)
            )
            for key, value in data.items()
            if any(
                word in key
                for word in ("pixel", "grid", "position", "size", "patch", "frame")
            )
        }

    def __enter__(self):
        import numpy as np
        from contextlib import nullcontext
        from vllm.multimodal.media.image import ImageMediaIO
        from .inputs import NoSpatialTransform, placeholders, payload_description

        super().__enter__()
        self.original_image_load = ImageMediaIO.load_file
        owner = self

        def load(loader, path):
            result = owner.original_image_load(loader, path)
            if owner.active:
                key = str(Path(path))
                require(key in owner.lookup, "Unexpected image reader path")
                image = result.media
                pixels = np.asarray(image)
                width, height = image_size(owner.lookup[key])
                require(
                    image.mode == "RGB"
                    and pixels.dtype == np.uint8
                    and pixels.shape == (height, width, 3),
                    "Image reader changed RGB8 geometry",
                )
                digest = hashlib.sha256(pixels.tobytes()).hexdigest()
                require(
                    digest == owner.lookup[key]["pixel_sha256"],
                    "Image reader pixels differ",
                )
                owner.reads.append(
                    {
                        "kind": "image",
                        "path": key,
                        "shape": list(pixels.shape),
                        "dtype": str(pixels.dtype),
                        "pixel_sha256": digest,
                    }
                )
            return result

        self.had_apply = "apply" in self.processor_class.__dict__
        self.original_apply = self.processor_class.apply

        def apply(processor, *args, **kwargs):
            guard = (
                NoSpatialTransform()
                if owner.case["modality"] == "static"
                else nullcontext()
            )
            with guard:
                result = owner.original_apply(processor, *args, **kwargs)
            if owner.active:
                payload = (
                    payload_description(result["mm_kwargs"], owner.model)
                    if owner.deep
                    else describe_tree(result["mm_kwargs"])
                )
                owner.applications.append(
                    {
                        "prompt_token_ids": list(result["prompt_token_ids"]),
                        "mm_placeholders": placeholders(result["mm_placeholders"]),
                        "native_payload": payload,
                    }
                )
            return result

        ImageMediaIO.load_file = load
        self.processor_class.apply = apply
        self.active = True
        return self

    def __exit__(self, *args):
        from vllm.multimodal.media.image import ImageMediaIO

        self.active = False
        ImageMediaIO.load_file = self.original_image_load
        if self.had_apply:
            self.processor_class.apply = self.original_apply
        else:
            delattr(self.processor_class, "apply")
        return super().__exit__(*args)


def describe_tree(value):
    """Keep ordinary receipts small; hashes are recorded on the representative."""
    import torch

    if isinstance(value, torch.Tensor):
        return tensor_description(value, False)
    if isinstance(value, (list, tuple)):
        return [describe_tree(item) for item in value]
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): describe_tree(item) for key, item in value.items()}
    if hasattr(value, "data"):
        return describe_tree(value.data)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Unexpected native payload type: {type(value)}")


def install_ingress(worker, spec, model):
    from .inputs import payload_description

    result = install_probe(worker, spec, "qwen_x_qsa" if model == "qwen" else model)
    state = worker._official_x_probe
    _, outer = resolve_module(worker.get_model(), spec["outer_paths"])
    state["ingress"] = []

    def observed(module, args, kwargs):
        import torch

        tensor = first_tensor(args, kwargs)
        require(
            tensor is not None and tensor.device.type == "cuda",
            "Vision input is not on CUDA",
        )
        grid = kwargs.get("grid_thw")
        if isinstance(grid, torch.Tensor):
            grid = grid.tolist()
        receipt = {
            "tensor": payload_description(tensor, model),
            "grid_thw": grid,
            "arguments": payload_description(
                {"args_after_pixels": args[1:], "kwargs": kwargs}, model
            ),
        }
        if model == "gemma":
            require(len(args) >= 3, "Gemma effective patch mask missing")
            receipt["mask_any_true"] = bool(args[2].any().item())
            receipt["masked_patch_counts"] = args[2].sum(-1).tolist()
        state["ingress"].append(receipt)

    state["handles"].append(outer.register_forward_pre_hook(observed, with_kwargs=True))
    if model == "qwen":
        from vlm_psychophysics.runtime.qsa import evidence

        state["selector_calls_before"] = evidence()["actual_calls"]
    return result


def read_ingress(worker, model):
    result = read_probe(worker, model)
    if model == "qwen":
        from vlm_psychophysics.runtime.qsa import evidence

        result["selector_calls_during_request"] = (
            evidence()["actual_calls"] - result["selector_calls_before"]
        )
    return result


def validate_ingress(
    model,
    modality,
    reference,
    applications,
    observations,
    *,
    mixed=False,
    equal_reference=False,
):
    from .inputs import ordered_hashes, official_source

    require(applications, "Native processor application was not observed")
    require(
        all(
            row["prompt_token_ids"] == reference["prompt_token_ids"]
            and row["mm_placeholders"] == reference["mm_placeholders"]
            for row in applications
        ),
        "Actual native prompt expansion or media placement differs from CPU",
    )
    payload = reference["native_payload"]
    expected_hashes = ordered_hashes(payload)
    require(expected_hashes, "CPU input reference has no ordered patches")
    require(
        any(
            ordered_hashes(row["native_payload"]) == expected_hashes
            for row in applications
        ),
        "Actual native CPU patches differ from the independent CPU reference",
    )
    actual_hashes = []
    actual_positions = []
    mixed_hashes = {"image": [], "video": []}
    mixed_positions = {"image": [], "video": []}
    for observation in observations:
        require(
            not observation.get("runtime_error")
            and not observation.get("profiler_error"),
            "Actual GPU observer failed",
        )
        if model == "qwen":
            require(
                observation["selector_calls_during_request"] > 0,
                "The deterministic Qwen QSA selector was not used",
            )
        if model in ("cosmos", "qwen"):
            normalizer = observation["normalizer"]
            require(
                normalizer["class"].endswith(
                    (".FusedMMInputNorm", ".IdentityInputNorm")
                )
                and normalizer.get("source"),
                "Native normalization identity is missing",
            )
            require(
                official_source(normalizer["source"]["path"], "vllm")["sha256"]
                == normalizer["source"]["sha256"],
                "Native normalization source changed",
            )
        for record in observation["ingress"]:
            tensor = record["tensor"]
            shape = tensor["shape"]
            expected_dtype = (
                "torch.uint8" if model in ("cosmos", "qwen") else "torch.bfloat16"
            )
            require(
                tensor["dtype"] == expected_dtype
                and tensor["device"].startswith("cuda"),
                "Actual vision ingress dtype/device changed",
            )
            hashes = tensor.get("per_image_sha256", [])
            require(hashes, "GPU input is missing ordered patch identities")
            image = False
            if model in ("cosmos", "qwen"):
                grid = record["grid_thw"]
                if grid and isinstance(grid[0], int):
                    grid = [grid]
                expected_grid = (
                    [[90, 32, 48]]
                    if modality == "video"
                    else [[1, 32, 48]] * len(hashes)
                )
                if mixed:
                    image_grid = [[1, 32, 48]] if equal_reference else [[1, 10, 10]]
                    image = grid == image_grid
                    require(
                        image or grid == [[90, 32, 48]],
                        "GPU reference or video grid differs",
                    )
                else:
                    require(
                        grid == expected_grid, "GPU video/image grid was sampled or resized"
                    )
            if model == "gemma":
                reference_slots, reference_masks = (
                    (2520, 936) if equal_reference else (1260, 1116)
                )
                image = mixed and shape[-2] == reference_slots
                mask_counts = record["masked_patch_counts"]
                expected_masks = [reference_masks] if image else [0] * len(hashes)
                require(mask_counts == expected_masks, "Gemma effective GPU patch mask differs")
            if mixed and model == "nemo":
                reference_patches = 1536 if equal_reference else 1024
                image = shape == [1, reference_patches, 768]
                require(
                    image or shape[-3:] == [3, 512, 768],
                    "GPU reference or video geometry differs",
                )
            positions = ordered_hashes(record["arguments"], "per_image_position_sha256")
            if mixed:
                kind = "image" if image else "video"
                mixed_hashes[kind].extend(hashes)
                mixed_positions[kind].extend(positions)
            actual_hashes.extend(hashes)
            actual_positions.extend(positions)
    if mixed:
        for kind in ("image", "video"):
            require(
                mixed_hashes[kind] == ordered_hashes(payload[kind]),
                f"GPU {kind} patch bytes differ from the CPU reference",
            )
            if model == "gemma":
                require(
                    mixed_positions[kind] == ordered_hashes(payload[kind], "per_image_position_sha256"),
                    f"Gemma GPU {kind} patch positions differ from the CPU reference",
                )
    # vLLM may encode one copy of an identical static image, retaining logical placements.
    unique = list(dict.fromkeys(expected_hashes))
    require(
        mixed or actual_hashes == expected_hashes
        or (modality == "static" and actual_hashes == unique),
        "GPU patch identity/order differs from the complete CPU input",
    )
    if model == "gemma" and not mixed:
        positions = ordered_hashes(
            reference["native_payload"], "per_image_position_sha256"
        )
        indices = (
            list(range(len(expected_hashes)))
            if actual_hashes == expected_hashes
            else [expected_hashes.index(value) for value in unique]
        )
        require(
            actual_positions == [positions[index] for index in indices],
            "Gemma GPU position order differs from the CPU reference",
        )
    return {
        "status": "PASS_NATIVE_INPUT",
        "CPU_GPU_patch_bytes_equal": True,
        "ordered_patch_units": len(expected_hashes),
        "source_frame_indices": list(range(180)) if modality == "video" else None,
        "prompt_token_ids_equal_CPU": True,
    }
