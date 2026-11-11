"""Native model sessions with pinned input and generation settings."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
import time

from . import data, inputs, observers


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def checksum(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def process_identity(worker=None):
    fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    return {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "startticks": fields[19],
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    }


def process_alive(identity):
    if identity["host"] != socket.gethostname():
        raise ValueError("Recorded worker belongs to another host")
    if (
        identity["boot_id"]
        != Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    ):
        return False
    try:
        fields = (
            Path(f"/proc/{identity['pid']}/stat").read_text().rsplit(")", 1)[1].split()
        )
    except FileNotFoundError:
        return False
    return fields[19] == str(identity["startticks"]) and fields[0] != "Z"


def resolved_profile(
    config, model, task, modality, data_root, *, scope=None, cohort=None
):
    model_config = config["models"][model]
    key = data.profile_key(modality, scope) if scope else f"{modality}/{task}"
    recipe_name = model_config["profiles"][key]
    recipe = config["recipes"][recipe_name]
    profile = copy.deepcopy(config["recipes"][recipe.get("base_recipe", recipe_name)])
    engine = profile["engine_kwargs"]
    if "reference_images" in recipe:
        engine["limit_mm_per_prompt"]["image"] = recipe["reference_images"]
    if cohort == "single":
        engine["limit_mm_per_prompt"]["image"] = recipe["single_reference_images"]
    if "video_inputs" in recipe:
        engine["limit_mm_per_prompt"]["video"] = recipe["video_inputs"]
    if "gpu_memory_utilization" in recipe:
        engine["gpu_memory_utilization"] = recipe["gpu_memory_utilization"]
    if "max_soft_tokens" in recipe:
        engine["mm_processor_kwargs"]["max_soft_tokens"] = recipe["max_soft_tokens"]
        profile["request_mm_processor_kwargs"]["max_soft_tokens"] = recipe[
            "max_soft_tokens"
        ]
    engine["model"] = str(Path(model_config["model_checkpoint"]).resolve())
    if "tokenizer" in engine:
        engine["tokenizer"] = str(Path(model_config["tokenizer_checkpoint"]).resolve())
    engine["allowed_local_media_path"] = str(Path(data_root).resolve())
    engine["media_io_kwargs"] = profile["media_io_kwargs"]
    profile.update(
        model=model,
        task=task,
        modality=modality,
        recipe=data.recipe_name(recipe_name, cohort),
    )
    return profile


def worker_environment(profile, gpu_uuid):
    for key in profile.get("cleared_environment", []):
        os.environ.pop(key, None)
    os.environ.update(profile.get("environment", {}))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    paths = [str(Path(sys.executable).parent)]
    toolkit = os.environ.get("VLM_PSYCHOPHYSICS_CUDA_HOME") or os.environ.get(
        "CUDA_HOME"
    )
    if toolkit:
        os.environ["CUDA_HOME"] = toolkit
        os.environ["FLASHINFER_NVCC"] = str(Path(toolkit) / "bin/nvcc")
        paths.append(str(Path(toolkit) / "bin"))
    paths.append(os.environ.get("PATH", ""))
    os.environ["PATH"] = os.pathsep.join(paths)


class Session:
    """Load one recipe; compatible tasks/domains share that loaded model."""

    def __init__(
        self,
        model,
        task,
        modality,
        config,
        data_root,
        output_dir,
        gpu_uuid,
        *,
        scope=None,
        cohort=None,
    ):
        self.model = model
        self.modality = modality
        self.config = config
        self.profile = resolved_profile(
            config, model, task, modality, data_root, scope=scope, cohort=cohort
        )
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.profile_sha256 = checksum(self.profile)
        self.references = {}
        self.processor_classes = {}
        self.engine = None
        self.adapter = None
        self.nemo_adapter = None
        self.workers = []
        self.sampling = {}
        worker_environment(self.profile, gpu_uuid)

        for package, expected in self.profile["expected_packages"].items():
            actual = importlib.metadata.version(package)
            if actual != expected:
                raise RuntimeError(f"{package}: expected {expected}, found {actual}")
        write_json(
            self.output / "resolved-model.json",
            {
                "profile": self.profile,
                "profile_sha256": self.profile_sha256,
                "generation": config["generation"],
                "owner": process_identity(),
                "gpu_uuid": gpu_uuid,
                "code_sha256": {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (
                        Path(__file__),
                        Path(inputs.__file__),
                        Path(observers.__file__),
                    )
                },
            },
        )
        from vllm import LLM

        engine_kwargs = copy.deepcopy(self.profile["engine_kwargs"])
        if isinstance(engine_kwargs.get("cpu_offload_params"), list):
            engine_kwargs["cpu_offload_params"] = set(
                engine_kwargs["cpu_offload_params"]
            )
        self.engine = LLM(**engine_kwargs)
        try:
            self.workers = self.engine.collective_rpc(process_identity)
            runtime = self.engine.collective_rpc(
                observers.worker_identity, args=(model,)
            )
            write_json(self.output / "runtime.json", runtime)
            inputs.require(len(runtime) == 1, "The local recipe requires one model worker per GPU")
            for receipt in runtime:
                inputs.require(
                    receipt["actual_kv_capacity"]
                    >= self.profile["minimum_kv_capacity"],
                    "Actual KV capacity is below the fixed recipe",
                )
                inputs.require(
                    any(
                        ("Fp4" in key or "FP4" in key or "NVFP4" in key) and count > 0
                        for key, count in receipt["method_kernel_counts"].items()
                    ),
                    "Actual model runtime has no NVFP4 implementation",
                )
                if model == "qwen":
                    inputs.require(
                        receipt["custom_qsa_selector"]["status"] == "APPLIED_X_QSA",
                        "The deterministic Qwen QSA selector is missing",
                    )
            if model == "gemma":
                self.adapter = inputs.GemmaAdapter(
                    config["models"][model]["processor_checkpoint"], modality
                )
            if model == "nemo" and modality == "static":
                self.nemo_adapter = inputs.NemoNoResize()
                self.profile["_nemo_adapter"] = self.nemo_adapter
        except BaseException:
            self.close()
            raise

    def _sampling_params(self, choices):
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        key = tuple(choices)
        if key not in self.sampling:
            settings = dict(self.config["generation"])
            thinking = settings.pop("enable_thinking")
            inputs.require(thinking is False, "The local recipe requires thinking disabled")
            self.sampling[key] = SamplingParams(
                **settings,
                structured_outputs=StructuredOutputsParams(choice=list(choices)),
            )
        return self.sampling[key]

    def predict(self, case):
        key = data.profile_key(case["modality"], data.case_scope(case))
        recipe = data.recipe_name(
            self.config["models"][self.model]["profiles"][key], case.get("cohort")
        )
        inputs.require(
            recipe == self.profile["recipe"], "Case needs a different loaded recipe"
        )
        reference_key = checksum(
            {
                "recipe": recipe,
                "prompt_utf8": case["prompt"],
                "choices": case["choices"],
                "modality": case["modality"],
                "image_count": len(case.get("images", [])),
                "width": 768,
                "height": 512,
                "frames": 180,
            }
        )
        deep = reference_key not in self.references
        media = (
            [
                *case.get("images", []),
                {"path": case["video_path"], "sha256": case["video_sha256"]},
            ]
            if self.modality == "video"
            else case["images"]
        )
        for entry in media:
            with Path(entry["path"]).open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            inputs.require(
                actual == entry["sha256"], f"Input SHA mismatch: {entry['path']}"
            )

        if deep:
            prepared = self.adapter.prepare(case)[0] if self.adapter else None
            reference, processor_class = inputs.cpu_reference(
                self.profile, case, prepared
            )
            del prepared
            self.references[reference_key] = reference
            self.processor_classes[reference_key] = processor_class
            append_json(
                self.output / "input-references.jsonl",
                {
                    "reference_key": reference_key,
                    "sample_id": case["sample_id"],
                    "recipe": recipe,
                    "reference": reference,
                },
            )
        reference = self.references[reference_key]
        sampling = self._sampling_params(case["choices"])
        if self.nemo_adapter:
            self.nemo_adapter.receipts.clear()
        if self.adapter:
            self.adapter.compaction_calls = 0
        observed = observers.InputObserver(
            self.model, self.processor_classes[reference_key], case, deep
        )
        started = time.monotonic()
        with observed:
            if self.adapter:
                prepared, padding = self.adapter.prepare(case)
            else:
                prepared, padding = None, []
            if deep:
                self.engine.collective_rpc(
                    observers.install_ingress,
                    args=(self.profile["vision_observation"], self.model),
                )
            if prepared is None:
                answer = self.engine.chat(
                    inputs.chat_messages(case),
                    sampling_params=sampling,
                    use_tqdm=False,
                    chat_template_kwargs={"enable_thinking": False},
                    mm_processor_kwargs=self.profile["request_mm_processor_kwargs"],
                )[0]
            else:
                answer = self.engine.generate(prepared, sampling, use_tqdm=False)[0]
            result = answer.outputs[0]
            # Persist returned text before any post-generation validation can fail.
            returned = {
                "sample_id": case["sample_id"],
                "case_id": case["case_id"],
                "raw_answer": result.text,
                "prompt_token_ids": list(answer.prompt_token_ids),
                "output_token_ids": list(result.token_ids),
                "finish_reason": result.finish_reason,
                "stop_reason": result.stop_reason,
                "profile_sha256": self.profile_sha256,
                "reference_key": reference_key,
                "seconds": time.monotonic() - started,
            }
            durable = {
                key: value
                for key, value in returned.items()
                if key != "prompt_token_ids"
            }
            durable["prompt_ids_sha256"] = checksum(returned["prompt_token_ids"])
            durable["prompt_token_count"] = len(returned["prompt_token_ids"])
            if returned["prompt_token_ids"] != reference["prompt_token_ids"]:
                durable["unexpected_prompt_token_ids"] = returned["prompt_token_ids"]
            append_json(self.output / "returned.jsonl", durable)
            technical_error = None
            gpu = None
            proof = None
            try:
                gpu = (
                    self.engine.collective_rpc(
                        observers.read_ingress, args=(self.model,)
                    )
                    if deep
                    else None
                )
                if deep:
                    append_json(
                        self.output / "input-observations.jsonl",
                        {
                            "sample_id": case["sample_id"],
                            "reference_key": reference_key,
                            "reader": observed.reads,
                            "application": observed.applications,
                            "GPU": gpu,
                        },
                    )
                inputs.require(
                    result.finish_reason == "stop",
                    "Generation did not finish at the native stop",
                )
                inputs.require(
                    list(answer.prompt_token_ids) == reference["prompt_token_ids"],
                    "Actual expanded prompt IDs differ from the CPU reference",
                )
                inputs.require(
                    observed.applications
                    and all(
                        row["prompt_token_ids"] == reference["prompt_token_ids"]
                        and row["mm_placeholders"] == reference["mm_placeholders"]
                        for row in observed.applications
                    ),
                    "Actual native image/video placements differ",
                )
                if self.modality == "video":
                    videos = [row for row in observed.reads if row.get("kind") != "image"]
                    inputs.require(
                        videos
                        and all(
                            row["shape"] == [180, 512, 768, 3]
                            and row["dtype"] == "uint8"
                            and row["metadata"]["frames_indices"] == list(range(180))
                            for row in videos
                        ),
                        "Actual reader did not deliver the original 180 frames in order",
                    )
                if case.get("images"):
                    expected = [
                        (str(Path(row["path"])), row["pixel_sha256"])
                        for row in case["images"]
                    ]
                    actual = [
                        (row["path"], row["pixel_sha256"])
                        for row in observed.reads
                        if row.get("kind") == "image"
                    ]
                    inputs.require(
                        actual == expected, "Actual ordered PNG inputs differ"
                    )
                mixed_video = self.modality == "video" and case["task"] == "T5"
                if mixed_video:
                    placements = reference["mm_placeholders"]
                    image, video = placements.get("image", []), placements.get("video", [])
                    inputs.require(
                        len(image) == len(video) == 1
                        and image[0]["offset"] + image[0]["length"] <= video[0]["offset"],
                        "Native placements must be one reference image before one video",
                    )
                elif self.modality == "static":
                    placements = reference["mm_placeholders"]
                    images = placements.get("image", [])
                    inputs.require(
                        set(placements) == {"image"}
                        and len(images) == len(case["images"])
                        and all(
                            left["offset"] + left["length"] <= right["offset"]
                            for left, right in zip(images, images[1:])
                        ),
                        "Native image placements must preserve the static input order",
                    )
                if self.adapter:
                    inputs.require(
                        self.adapter.compaction_calls > 0,
                        "Gemma native masked-tail compaction did not execute",
                    )
                if deep:
                    proof = observers.validate_ingress(
                        self.model,
                        self.modality,
                        reference,
                        observed.applications,
                        gpu,
                        mixed=mixed_video,
                        equal_reference=mixed_video and "reference_version" in case,
                    )
            except Exception as error:
                technical_error = f"{type(error).__name__}: {error}"

        returned["technical_pass"] = technical_error is None
        returned["technical_error"] = technical_error
        returned["input_receipt"] = {
            "reference_key": reference_key,
            "deep_observation": deep,
            "prompt_sha256": hashlib.sha256(case["prompt"].encode("utf-8")).hexdigest(),
            "prompt_ids_sha256": checksum(returned["prompt_token_ids"]),
            "frame_indices": (
                list(range(180)) if self.modality == "video" else case["frame_indices"]
            ),
            "reader": observed.reads,
            "padding": padding,
            "admission": proof,
        }
        return returned

    def close(self):
        if self.adapter:
            self.adapter.close()
            self.adapter = None
        if self.nemo_adapter:
            self.nemo_adapter.__exit__(None, None, None)
            self.nemo_adapter = None
            self.profile.pop("_nemo_adapter", None)
        if self.engine is None:
            return
        shutdown_error = None
        try:
            self.engine.llm_engine.engine_core.shutdown()
        except Exception as error:
            shutdown_error = f"{type(error).__name__}: {error}"
        while any(process_alive(worker) for worker in self.workers):
            time.sleep(1)
        write_json(
            self.output / "exit.json",
            {
                "owned_workers": self.workers,
                "owned_workers_exited": True,
                "shutdown_error": shutdown_error,
            },
        )
        self.engine = None
        if shutdown_error:
            raise RuntimeError(shutdown_error)
