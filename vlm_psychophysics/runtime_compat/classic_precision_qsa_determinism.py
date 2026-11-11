"""Opt-in, exact Qwen QSA selection for the pinned vLLM implementation.

The native persistent_topk dispatch can drop candidates (vLLM issue #51782).
This helper ranks the original scores, resolves equal scores by lower column
index, and writes the selected columns in ascending order. It never substitutes
rank keys or adds an epsilon to scientific floating-point values. Only the
private Qwen QSA helper is replaced; torch.ops and installed SDK files stay intact.

install() imports no GPU libraries. A source-checked import hook applies the
replacement when the target module is loaded. status() distinguishes that
pending hook from an applied replacement. This is a selection determinism fix,
not evidence that the complete model is reproducible across devices or versions.
"""
import functools
import hashlib
import importlib.abc
import importlib.machinery
import inspect
import marshal
import os
from pathlib import Path
import sys
from types import CodeType


ENVIRONMENT_VARIABLE = "CLASSIC_PRECISION_QSA_DETERMINISM"
TARGET_MODULE = "vllm.models.qwen4_exp.nvidia.ops.qsa_indexer"
EXPECTED_SOURCE_SHA256 = "4d09eb2f409e7cd71c7849c4f3bdd71d722bce6feaccb558f7ae67ac2d489a2b"
MODEL_ARCHITECTURE = "Qwen4ExpForConditionalGeneration"
# The native prefill path already chunks logits. Further bound sort temporaries
# by rows; one exceptionally wide row is the minimum possible unit of work.
MAX_SORT_ELEMENTS = 1 << 20
_PARAMETERS = ("logits", "visible_blocks", "token_topk", "compress_ratio",
               "block_indices", "topk_workspace")
_MARKER = "_classic_precision_qsa_determinism"
_finder = None
_state = None


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _evidence(state):
    return {
        "environment_variable": ENVIRONMENT_VARIABLE,
        "target_module": TARGET_MODULE,
        "model_architecture": MODEL_ARCHITECTURE,
        "expected_native_source_sha256": EXPECTED_SOURCE_SHA256,
        "helper_source_sha256": _sha(__file__),
        "algorithm": "stable_score_descending_index_ties_then_index_ascending",
        "padding": -1,
        "visible_nan_policy": "reject",
        "max_sort_elements": MAX_SORT_ELEMENTS,
        "state": state,
    }


def _shape_checks(torch, logits, visible_blocks, token_topk, compress_ratio,
                  block_indices, topk_workspace):
    if (type(token_topk) is not int or type(compress_ratio) is not int
            or token_topk < 0 or compress_ratio <= 0
            or token_topk % compress_ratio):
        raise ValueError("QSA token_topk must be nonnegative and divisible by positive compress_ratio")
    tensors = (logits, visible_blocks, block_indices, topk_workspace)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("QSA inputs and output must be torch tensors")
    if logits.ndim != 2 or logits.dtype not in (
            torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("QSA logits must be a two-dimensional real floating tensor")
    rows, columns = logits.shape
    block_topk = token_topk // compress_ratio
    if visible_blocks.shape != (rows,) or visible_blocks.dtype not in (torch.int32, torch.int64):
        raise ValueError("QSA visible_blocks must be an integer vector with one length per row")
    if (block_indices.shape != (rows, block_topk)
            or block_indices.dtype not in (torch.int32, torch.int64)):
        raise ValueError("QSA block_indices must be an integer [rows, block_topk] output")
    if (topk_workspace.ndim != 1 or topk_workspace.dtype != torch.uint8):
        raise ValueError("QSA topk_workspace must be a one-dimensional uint8 tensor")
    if any(value.device != logits.device for value in tensors[1:]):
        raise ValueError("QSA tensors must be on the same device")
    if columns > torch.iinfo(block_indices.dtype).max:
        raise ValueError("QSA column indices do not fit the output dtype")
    if MAX_SORT_ELEMENTS <= 0:
        raise RuntimeError("QSA sort element budget must be positive")
    # Unlike bool(tensor) or .item(), this assertion does not introduce a GPU
    # to CPU synchronization, and remains usable during CUDA graph capture.
    torch._assert_async(((visible_blocks >= 0) & (visible_blocks <= columns)).all(),
                        "QSA visible_blocks lies outside the logits width")
    return rows, columns, block_topk


def exact_qsa_topk(logits, visible_blocks, token_topk, compress_ratio,
                   block_indices, topk_workspace):
    """Write exact, canonical block indices in place; return None like native.

    Visible columns are the prefix [0, visible_blocks[row]). Invisible logits
    may be uninitialized or NaN. Visible +/-inf scores are ordered normally;
    visible NaNs fail closed. Original floating values and input tensors are
    never modified. The selection at a tie boundary prefers smaller indices,
    then selected indices are sorted ascending for a fixed attention traversal.
    Unfilled output slots are -1, including when block_topk exceeds the width.
    """
    import torch

    rows, columns, block_topk = _shape_checks(
        torch, logits, visible_blocks, token_topk, compress_ratio,
        block_indices, topk_workspace)
    if rows == 0:
        return None
    if columns == 0:
        block_indices.fill_(-1)
        return None
    rows_per_chunk = max(1, MAX_SORT_ELEMENTS // columns)
    column = torch.arange(columns, device=logits.device)
    count = min(block_topk, columns)
    for start in range(0, rows, rows_per_chunk):
        stop = min(rows, start + rows_per_chunk)
        scores = logits[start:stop]
        visible = column.unsqueeze(0) < visible_blocks[start:stop].unsqueeze(1)
        torch._assert_async((~torch.isnan(scores) | ~visible).all(),
                            "QSA visible logits contain NaN")
        if count == 0:
            continue
        # Stable sort starts with ascending column indices. Masked columns are
        # a suffix, so even a visible -inf precedes all invisible -inf entries.
        masked = scores.masked_fill(~visible, float("-inf"))
        selected = torch.argsort(masked, dim=1, descending=True, stable=True)[:, :count]
        selected_live = selected < visible_blocks[start:stop].unsqueeze(1)
        # A width sentinel puts padding after all valid columns during the
        # canonical index sort; writing -1 before sorting would put padding first.
        canonical = selected.masked_fill(~selected_live, columns).sort(dim=1).values
        canonical.masked_fill_(canonical == columns, -1)
        block_indices[start:stop, :count].copy_(canonical)
        block_indices[start:stop, count:].fill_(-1)
    return None


def _verified_source(path):
    source = Path(path).read_bytes()
    if hashlib.sha256(source).hexdigest() != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("QSA native source SHA mismatch; refusing unverified replacement")
    return source


def _check_native_function(module, source):
    original = getattr(module, "_topk", None)
    if getattr(original, _MARKER, None) is not None:
        return original
    if (not inspect.isfunction(original) or original.__module__ != TARGET_MODULE
            or original.__globals__ is not vars(module)
            or original.__code__.co_filename != module.__file__):
        raise RuntimeError("QSA _topk was already replaced or has an unexpected identity")
    parameters = tuple(inspect.signature(original).parameters.values())
    if (tuple(parameter.name for parameter in parameters) != _PARAMETERS
            or any(parameter.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD
                   or parameter.default is not inspect.Parameter.empty
                   for parameter in parameters)):
        raise RuntimeError("QSA _topk signature mismatch")
    compiled = compile(source, module.__file__, "exec", dont_inherit=True)
    expected = [value for value in compiled.co_consts
                if isinstance(value, CodeType) and value.co_name == "_topk"]
    if len(expected) != 1 or marshal.dumps(original.__code__) != marshal.dumps(expected[0]):
        raise RuntimeError("QSA _topk code mismatch; refusing a modified runtime helper")
    return original


def _apply(module):
    global _state
    source = _verified_source(module.__file__)
    original = _check_native_function(module, source)
    previous = getattr(original, _MARKER, None)
    if previous is not None:
        if previous != _evidence("applied"):
            raise RuntimeError("QSA replacement identity mismatch")
        _state = dict(previous)
        return
    evidence = _evidence("applied")

    @functools.wraps(original)
    def topk(logits, visible_blocks, token_topk, compress_ratio,
             block_indices, topk_workspace):
        return exact_qsa_topk(logits, visible_blocks, token_topk, compress_ratio,
                              block_indices, topk_workspace)

    setattr(topk, _MARKER, evidence)
    module._topk = topk
    _state = dict(evidence)


class _GuardedLoader(importlib.abc.Loader):
    def __init__(self, native_loader, origin):
        self.native_loader = native_loader
        self.origin = origin

    def create_module(self, spec):
        return self.native_loader.create_module(spec)

    def exec_module(self, module):
        # Check before any target-module imports execute, and compile verified
        # source directly rather than trusting a separately cached .pyc file.
        source = _verified_source(self.origin)
        exec(compile(source, self.origin, "exec", dont_inherit=True), vars(module))
        _apply(module)


class _QSAFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET_MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if (spec is None or not spec.origin
                or not isinstance(spec.loader, importlib.machinery.SourceFileLoader)):
            raise RuntimeError("QSA target must use an inspectable Python source loader")
        _verified_source(spec.origin)
        spec.loader = _GuardedLoader(spec.loader, spec.origin)
        return spec


def status():
    """Return this process's pending/applied receipt, or None before opt-in."""
    return None if _state is None else dict(_state)


def install():
    """Enable only for the exact opt-in value 1; do not import torch or vLLM."""
    global _finder, _state
    requested = os.environ.get(ENVIRONMENT_VARIABLE)
    if requested is None:
        return None
    if requested != "1":
        raise ValueError(ENVIRONMENT_VARIABLE + " must be exactly 1")
    loaded = sys.modules.get(TARGET_MODULE)
    if loaded is not None:
        _apply(loaded)
    if _finder is None:
        _finder = _QSAFinder()
        sys.meta_path.insert(0, _finder)
    if _state is None:
        _state = _evidence("pending_target_import")
    return status()
