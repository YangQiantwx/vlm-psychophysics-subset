"""CPU proofs for exact score selection and the narrow deferred QSA hook."""
import hashlib
import importlib
import math
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

torch = pytest.importorskip("torch")

from vlm_psychophysics.runtime_compat import classic_precision_qsa_determinism as compat


ROOT = Path(__file__).resolve().parents[2]


def _reference(logits, visible, k):
    """Independent scalar specification, including canonical output order."""
    result = []
    for row, length in zip(logits.tolist(), visible.tolist()):
        assert all(not math.isnan(value) for value in row[:length])
        ranked = sorted(range(length), key=lambda column: (-row[column], column))
        chosen = sorted(ranked[:k])
        result.append(chosen + [-1] * (k - len(chosen)))
    return result


def _run(logits, visible, k, ratio=1, output=None):
    if output is None:
        output = torch.full((logits.shape[0], k), 777, dtype=torch.int32)
    workspace = torch.full((17,), 31, dtype=torch.uint8)
    workspace_before = workspace.clone()
    result = compat.exact_qsa_topk(logits, visible, k * ratio, ratio, output, workspace)
    assert result is None
    assert torch.equal(workspace, workspace_before)
    return output


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_boundary_ties_use_lower_original_index_without_perturbing_scores(dtype):
    logits = torch.tensor([[3, 7, 7, 2, 7, 7, -4],
                           [-0.0, 0.0, -0.0, 0.0, -1, -1, -1]], dtype=dtype)
    before = logits.clone()
    visible = torch.tensor([7, 7], dtype=torch.int32)
    assert _run(logits, visible, 3, ratio=16).tolist() == [[1, 2, 4], [0, 1, 2]]
    assert torch.equal(logits, before)
    # Bit representations also preserve signed zero and the original precision.
    assert torch.equal(logits.view(torch.uint8), before.view(torch.uint8))


def test_nearly_equal_floats_and_large_fp32_scores_keep_the_actual_larger_values():
    one = torch.tensor(1.0, dtype=torch.float32)
    larger = torch.nextafter(one, torch.tensor(float("inf")))
    logits = torch.tensor([[1.0, larger.item(), 1.0, larger.item()],
                           [806733.0, 806736.0, 806734.0, 806735.0]])
    visible = torch.tensor([4, 4], dtype=torch.int64)
    assert _run(logits, visible, 2).tolist() == [[1, 3], [1, 3]]
    assert torch.equal(logits[0, [1, 3]], larger.expand(2))


@pytest.mark.parametrize("k", [0, 1, 4, 9, 13])
def test_visibility_padding_infinities_and_invisible_nan_match_scalar_reference(k):
    logits = torch.tensor([[float("nan")] * 9,
                           [-float("inf"), float("nan")] + [999.0] * 7,
                           [2, -float("inf"), float("inf"), -float("inf"),
                            2, float("nan"), 999, 999, 999],
                           [5, 8, 8, 1, 8, 4, 8, 3, 7]], dtype=torch.float64)
    before = logits.view(torch.uint8).clone()
    visible = torch.tensor([0, 1, 5, 9], dtype=torch.int32)
    actual = _run(logits, visible, k)
    assert actual.tolist() == _reference(logits, visible, k)
    assert torch.equal(logits.view(torch.uint8), before)
    assert torch.equal(visible, torch.tensor([0, 1, 5, 9], dtype=torch.int32))


def test_noncontiguous_logits_visibility_and_output_are_supported(monkeypatch):
    generator = torch.Generator().manual_seed(13)
    backing = torch.randint(-3, 4, (8, 23), generator=generator).float()
    logits = backing[::2, ::2]
    lengths = torch.tensor([0, 99, 4, 99, 11, 99, 12, 99], dtype=torch.int64)[::2]
    output_backing = torch.full((8, 18), 777, dtype=torch.int64)
    output = output_backing[::2, ::2]
    # Force several row chunks; chunk boundaries must not alter the selection.
    monkeypatch.setattr(compat, "MAX_SORT_ELEMENTS", 13)
    _run(logits, lengths, 9, output=output)
    assert output.tolist() == _reference(logits, lengths, 9)
    assert torch.all(output_backing[1::2] == 777)
    assert torch.all(output_backing[::2, 1::2] == 777)


@pytest.mark.parametrize("rows,columns,k", [(1, 4096, 512), (8, 12288, 2048),
                                           (33, 16385, 512), (33, 32768, 2048)])
def test_concentrated_logits_and_capacity_boundaries_select_exact_values(rows, columns, k):
    generator = torch.Generator().manual_seed(51782)
    # Include narrow fp32 values and repeated low precision values, which stress
    # the native kernel's coarse histogram candidate capacity in issue #51782.
    logits = torch.randn((rows, columns), generator=generator) * 0.1 + 10.0
    logits[0] = torch.linspace(1.0, 1.244, columns)
    if rows > 1:
        logits[1] = logits[1].bfloat16().float()
    visible = torch.full((rows,), columns, dtype=torch.int32)
    actual = _run(logits, visible, k)
    expected = _reference(logits, visible, k)
    assert actual.tolist() == expected
    selected_values = logits.gather(1, actual.long()).sort(dim=1, descending=True).values
    reference_values = torch.topk(logits, k, dim=1).values
    assert torch.equal(selected_values, reference_values)
    for _ in range(2):
        assert torch.equal(_run(logits, visible, k), actual)


def test_random_ties_lengths_and_k_match_independent_specification(monkeypatch):
    generator = torch.Generator().manual_seed(20261022)
    monkeypatch.setattr(compat, "MAX_SORT_ELEMENTS", 47)
    for columns in (1, 7, 31, 64):
        logits = torch.randint(-4, 5, (9, columns), generator=generator).float()
        visible = torch.randint(0, columns + 1, (9,), generator=generator, dtype=torch.int32)
        for k in (0, 1, columns // 2, columns, columns + 3):
            assert _run(logits, visible, k).tolist() == _reference(logits, visible, k)


def test_empty_rows_and_empty_width_have_defined_padding():
    assert _run(torch.empty((0, 7)), torch.empty((0,), dtype=torch.int32), 3).shape == (0, 3)
    assert _run(torch.empty((2, 0)), torch.zeros((2,), dtype=torch.int32), 4).tolist() == [
        [-1] * 4, [-1] * 4]


@pytest.mark.parametrize("length", [-1, 5])
def test_out_of_range_visibility_fails_before_output_write(length):
    output = torch.full((1, 2), 777, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="outside the logits width"):
        _run(torch.ones((1, 4)), torch.tensor([length], dtype=torch.int32), 2, output=output)
    assert output.tolist() == [[777, 777]]


@pytest.mark.parametrize("k", [0, 1])
def test_visible_nan_fails_closed(k):
    with pytest.raises(RuntimeError, match="visible logits contain NaN"):
        _run(torch.tensor([[0.0, float("nan")]]), torch.tensor([2], dtype=torch.int32), k)


@pytest.mark.parametrize("change,match", [
    ({"token_topk": 3, "compress_ratio": 2}, "divisible"),
    ({"token_topk": True}, "divisible"),
    ({"compress_ratio": 0}, "divisible"),
    ({"logits": torch.ones((4,))}, "two-dimensional"),
    ({"logits": torch.ones((1, 4), dtype=torch.int32)}, "floating"),
    ({"visible_blocks": torch.tensor([4.0])}, "integer vector"),
    ({"visible_blocks": torch.tensor([[4]])}, "integer vector"),
    ({"block_indices": torch.empty((1, 3), dtype=torch.int32)}, "block_indices"),
    ({"block_indices": torch.empty((1, 2), dtype=torch.float32)}, "block_indices"),
    ({"topk_workspace": torch.empty((1,), dtype=torch.float32)}, "uint8"),
])
def test_shape_and_type_drift_is_rejected(change, match):
    args = dict(logits=torch.ones((1, 4)), visible_blocks=torch.tensor([4], dtype=torch.int32),
                token_topk=2, compress_ratio=1,
                block_indices=torch.empty((1, 2), dtype=torch.int32),
                topk_workspace=torch.empty((1,), dtype=torch.uint8))
    args.update(change)
    with pytest.raises((ValueError, TypeError), match=match):
        compat.exact_qsa_topk(**args)


@pytest.fixture
def isolated_hook(monkeypatch):
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    monkeypatch.setattr(compat, "_finder", None)
    monkeypatch.setattr(compat, "_state", None)
    monkeypatch.setenv(compat.ENVIRONMENT_VARIABLE, "1")


@pytest.fixture
def native_fixture(tmp_path, monkeypatch, isolated_hook):
    source = '''def _topk(logits, visible_blocks, token_topk, compress_ratio,
          block_indices, topk_workspace):
    block_indices.fill_(99)

def qsa_select_paged_decode(*args):
    return _topk(*args)

def qsa_select_paged_prefill(*args):
    return _topk(*args)

UNRELATED = object()
'''
    components = compat.TARGET_MODULE.split(".")
    parent = tmp_path
    for component in components[:-1]:
        parent /= component
        parent.mkdir()
        (parent / "__init__.py").write_text("")
    path = parent / (components[-1] + ".py")
    path.write_text(source)
    monkeypatch.setattr(compat, "EXPECTED_SOURCE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.syspath_prepend(str(tmp_path))
    # Track imports for cleanup without erasing any pre-existing GPU modules.
    for stop in range(1, len(components) + 1):
        monkeypatch.delitem(sys.modules, ".".join(components[:stop]), raising=False)
    return path


def test_explicit_install_is_deferred_idempotent_and_only_replaces_target(native_fixture):
    before_ops = torch.ops._C
    receipt = compat.install()
    assert receipt["state"] == "pending_target_import"
    assert compat.TARGET_MODULE not in sys.modules
    finder = compat._finder
    assert compat.install() == receipt and compat._finder is finder
    assert sys.meta_path.count(finder) == 1
    native = importlib.import_module(compat.TARGET_MODULE)
    unrelated = native.UNRELATED
    assert compat.status()["state"] == "applied"
    assert getattr(native._topk, compat._MARKER) == compat.status()
    logits = torch.tensor([[1.0, 3.0, 2.0]])
    lengths = torch.tensor([3], dtype=torch.int32)
    for caller in (native.qsa_select_paged_decode, native.qsa_select_paged_prefill):
        out = torch.empty((1, 2), dtype=torch.int32)
        caller(logits, lengths, 2, 1, out, torch.empty((1,), dtype=torch.uint8))
        assert out.tolist() == [[1, 2]]
    installed = native._topk
    assert compat.install() == compat.status() and native._topk is installed
    assert native.UNRELATED is unrelated
    assert torch.ops._C is before_ops


def test_already_imported_native_is_checked_and_replaced(native_fixture):
    native = importlib.import_module(compat.TARGET_MODULE)
    original = native._topk
    assert compat.install()["state"] == "applied"
    assert native._topk.__wrapped__ is original


def test_native_source_drift_is_rejected_before_target_execution(native_fixture):
    compat.install()
    native_fixture.write_text("raise AssertionError('UNVERIFIED_IMPORT')\n")
    with pytest.raises(RuntimeError, match="native source SHA mismatch"):
        importlib.import_module(compat.TARGET_MODULE)
    assert compat.TARGET_MODULE not in sys.modules


def test_already_loaded_runtime_code_drift_is_rejected(native_fixture):
    native = importlib.import_module(compat.TARGET_MODULE)
    original = native._topk
    # Preserve function metadata, globals and signature, while modifying its
    # actual bytecode. Checking source text or inspect.signature alone misses it.
    code = compile(native_fixture.read_bytes().replace(b"99", b"98"), str(native_fixture), "exec")
    replacement = next(value for value in code.co_consts if getattr(value, "co_name", None) == "_topk")
    native._topk.__code__ = replacement
    with pytest.raises(RuntimeError, match="code mismatch"):
        compat.install()
    assert native._topk is original


def test_wrong_runtime_signature_fails_closed(native_fixture):
    native = importlib.import_module(compat.TARGET_MODULE)
    code = compile("def _topk(logits):\n    pass\n", str(native_fixture), "exec")
    exec(code, vars(native))
    with pytest.raises(RuntimeError, match="signature mismatch"):
        compat.install()


@pytest.mark.parametrize("value", ["", "0", "true", "01", " 1", "1 "])
def test_invalid_opt_in_is_rejected_without_hook(monkeypatch, isolated_hook, value):
    monkeypatch.setenv(compat.ENVIRONMENT_VARIABLE, value)
    before = list(sys.meta_path)
    with pytest.raises(ValueError, match="must be exactly 1"):
        compat.install()
    assert sys.meta_path == before and compat.status() is None


def test_absent_opt_in_is_inert(monkeypatch, isolated_hook):
    monkeypatch.delenv(compat.ENVIRONMENT_VARIABLE)
    before = list(sys.meta_path)
    assert compat.install() is None
    assert sys.meta_path == before and compat.status() is None


def test_helper_and_deferred_install_import_no_torch_or_vllm(tmp_path):
    environment = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"}
    environment.pop(compat.ENVIRONMENT_VARIABLE, None)
    code = '''import os, sys
from vlm_psychophysics.runtime_compat import classic_precision_qsa_determinism as helper
assert helper.install() is None
os.environ[helper.ENVIRONMENT_VARIABLE] = '1'
assert helper.install()['state'] == 'pending_target_import'
assert not any(name == 'torch' or name.startswith('torch.') or name == 'vllm'
               or name.startswith('vllm.') for name in sys.modules)
'''
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=tmp_path,
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
