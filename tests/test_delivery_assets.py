import hashlib
import io
import json
import os
from urllib.request import Request

import pytest

from vlm_psychophysics import assets


def file_entry(data, *, weight=False):
    return {
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "weight": weight,
    }


def test_offline_reuse_rehashes_same_size_weight_after_stat_change(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(assets, "_ssd", lambda path: None)
    monkeypatch.setattr(assets, "_ssd_root", lambda *_: tmp_path / "cache")
    model = tmp_path / "model"
    model.mkdir()
    config_bytes, shard_bytes = b"{}\n", b"weight contents"
    (model / "config.json").write_bytes(config_bytes)
    shard = model / "model.safetensors"
    shard.write_bytes(shard_bytes)
    material = {
        "files": {
            "config.json": file_entry(config_bytes),
            shard.name: file_entry(shard_bytes, weight=True),
        }
    }
    lock = {
        "bindings": {
            "test": {"model_checkpoint": "model", "tokenizer_checkpoint": "model"}
        },
        "materials": {"model": material},
    }
    monkeypatch.setattr(assets, "_lock", lambda config: lock)
    config = {"models": {"test": {"model_checkpoint": str(model)}}}
    resolved = assets.ensure_models(config, ["test"], download=False)
    assert resolved["models"]["test"]["tokenizer_checkpoint"] == str(model)
    assert "tokenizer_checkpoint" not in config["models"]["test"]
    assert assets._full_weight_proof(tmp_path / "cache", model, material)
    unchanged = assets._file_sha

    def no_weight_hash(path):
        assert path != shard
        return unchanged(path)

    with monkeypatch.context() as mp:
        mp.setattr(assets, "_file_sha", no_weight_hash)
        assert assets.ensure_models(config, ["test"], download=False) == resolved
    old = shard.stat()
    shard.write_bytes(b"Weight contents")
    os.utime(shard, ns=(old.st_atime_ns, old.st_mtime_ns))
    assert not assets._full_weight_proof(tmp_path / "cache", model, material)
    with pytest.raises(assets.AssetError, match="wrong SHA-256"):
        assets.ensure_models(config, ["test"], download=False)
    (model / "config.json").write_bytes(b"[]\n")
    with pytest.raises(assets.AssetError, match="wrong SHA-256"):
        assets.ensure_models(config, ["test"], download=False)


def test_gemma_uses_processor_architecture_and_pinned_quantization(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(assets, "_ssd", lambda path: None)
    raw, processor, effective = (
        tmp_path / name for name in ("raw", "processor", "effective")
    )
    raw.mkdir()
    processor.mkdir()
    original = {"architecture": "base", "nested": {"width": 32}}
    quant = {"quant_method": "modelopt", "precision": "NVFP4"}
    (processor / "config.json").write_text(json.dumps(original))
    (raw / "config.json").write_text(
        json.dumps({"architecture": "exporter", "quantization_config": quant})
    )
    (processor / "tokenizer_config.json").write_bytes(b"base")
    (raw / "tokenizer_config.json").write_bytes(b"quantized")
    (raw / "model.safetensors").write_bytes(b"weight")
    config = b'{"architecture":"base","nested":{"width":32},"quantization_config":{"precision":"NVFP4","quant_method":"modelopt"}}\n'
    material = {
        "files": {
            "config.json": file_entry(config),
            "tokenizer_config.json": file_entry(b"base"),
            "model.safetensors": file_entry(b"weight", weight=True),
        }
    }
    assets._compose_gemma(effective, material, raw, processor)
    assert (effective / "config.json").read_bytes() == config
    assert (effective / "model.safetensors").samefile(raw / "model.safetensors")
    assert (
        effective / "tokenizer_config.json"
    ).resolve() == processor / "tokenizer_config.json"


def test_ngc_redirect_strips_credential_and_download_verifies_bytes(
    tmp_path, monkeypatch
):
    origin = "https://api.ngc.nvidia.com/asset/"
    signed_url = "https://cdn.example.com/download?signature=test"
    request = Request(
        origin, headers={"Authorization": "Bearer test", "Range": "bytes=2-"}
    )
    redirect = assets._SafeRedirect().redirect_request(
        request, None, 302, "", {}, signed_url
    )
    assert redirect.get_header("Authorization") is None
    assert redirect.get_header("Range") == "bytes=2-"

    class Response(io.BytesIO):
        status = 206
        headers = {"Content-Range": "bytes 2-6/7"}

    class MockOpener:
        def open(self, req, timeout):
            assert req.full_url == origin + "shard"
            assert req.get_header("Authorization") == "Bearer test"
            assert req.get_header("Range") == "bytes=2-"
            return Response(b"12345")

    monkeypatch.setenv("NGC_API_KEY", "test")
    monkeypatch.setattr(assets, "build_opener", lambda *args: MockOpener())
    material = {"url": origin, "files": {"shard": file_entry(b"AB12345", weight=True)}}
    (tmp_path / "shard.part").write_bytes(b"AB")
    assets._download_ngc(tmp_path, material)
    assert (tmp_path / "shard").read_bytes() == b"AB12345"
    assert not (tmp_path / "shard.part").exists()


def test_runtime_gate_checks_sdk_bytes_and_requires_nvcc_only_for_bound_models(
    tmp_path, monkeypatch
):
    import platform
    import sys

    sdk_name = "vllm/processor.py"
    sdk = tmp_path / sdk_name
    sdk.parent.mkdir()
    sdk.write_bytes(b"original SDK source")
    torch_version = tmp_path / "torch/version.py"
    torch_version.parent.mkdir()
    torch_version.write_text("__version__ = '1.0+cu130'\ncuda = '13.0'\n")

    class Distribution:
        version = "1.0"

        def locate_file(self, name):
            return tmp_path / name

    runtime = {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "packages": {"torch": "1.0", "vllm": "1.0"},
        "package_sources": {},
        "torch_build": "1.0+cu130",
        "torch_cuda": "13.0",
        "files": {sdk_name: assets._file_sha(sdk)},
        "compiler_models": ["gemma"],
        "compiler": "13.0.88",
    }
    monkeypatch.setattr(assets, "_lock", lambda _: {"runtime": runtime})
    monkeypatch.setattr(assets.metadata, "distribution", lambda _: Distribution())
    monkeypatch.setattr(
        assets, "_cuda_toolchain", lambda: {"version": None, "ninja": None}
    )
    assets.assert_runtime_environment({}, ["qwen"])
    with pytest.raises(assets.AssetError, match=r"nvcc: expected 13\.0\.88"):
        assets.assert_runtime_environment({}, ["gemma"])

    sdk.write_bytes(b"modified SDK source")
    with pytest.raises(assets.AssetError, match="Missing or changed SDK files"):
        assets.assert_runtime_environment({}, ["qwen"])


def test_cli_runtime_gate_blocks_run_before_data_and_allows_download(
    tmp_path, monkeypatch, capsys
):
    from vlm_psychophysics import cli

    def reject_runtime(config, models):
        assert models == ("qwen",)
        raise assets.AssetError("Runtime unavailable")

    monkeypatch.setattr(assets, "assert_runtime_environment", reject_runtime)
    out = tmp_path / "output"
    flags = ["--task", "t1-classic", "--model", "qwen", "--out", str(out)]
    assert cli.main(flags) == 2
    assert capsys.readouterr().err.strip() == "Runtime unavailable"
    assert not out.exists()

    monkeypatch.setattr(cli.runner, "data_directory", lambda *a, **kw: tmp_path)
    monkeypatch.setattr(cli.runner, "check_storage", lambda *a, **kw: None)
    monkeypatch.setattr(
        cli.data, "build_selection", lambda *a, **kw: ({"cases": []}, {})
    )
    monkeypatch.setattr(cli.runner, "save_plan", lambda *a, **kw: None)
    monkeypatch.setattr(cli.data, "prepare_media", lambda *a, **kw: {})
    monkeypatch.setattr(
        assets,
        "ensure_models",
        lambda *a, **kw: {"models": {"qwen": {"model_checkpoint": "cached"}}},
    )
    assert cli.main(["download", *flags]) == 0
    assert json.loads(capsys.readouterr().out)["checkpoints"] == {"qwen": "cached"}


def test_default_native_output_falls_back_to_data_ssd(tmp_path, monkeypatch):
    from vlm_psychophysics import cli

    monkeypatch.chdir(tmp_path)
    cache = tmp_path / "ssd"
    def reject_local(path, **kwargs):
        assert path.parent == tmp_path / "outputs"
        raise ValueError("No SSD backing")

    monkeypatch.setattr(cli.runner, "check_storage", reject_local)
    args = cli.parse_arguments(["plan", "--task", "t5-natural"])
    output = cli.output_directory(args, cli.selection_for(args), cache)
    assert output.parent == cache / "outputs"
    assert output.name.startswith("natural-t5-")
