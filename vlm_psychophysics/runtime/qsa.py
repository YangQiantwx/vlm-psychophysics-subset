"""One process-local QSA selector adaptation for the pinned official X runtime.

Only the old, verified exact selection mathematics is reused. Its install hook,
sitecustomize and other compatibility helpers are never activated.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib
import importlib.util
import inspect
import marshal
from pathlib import Path
import sys
from types import CodeType

TARGET = 'vllm.models.qwen4_exp.nvidia.ops.qsa_indexer'
NATIVE_SOURCE_SHA = '8ee084541582f1ab597481342e8b9652b7e6d651515a698a0f8aee6e1401ec04'
HELPER_SOURCE_SHA = 'bf3f16fc3092b4edc6a8308dbad49bc84efacc327a88e2f697ae6bb3b38b7624'
PARAMETERS = ('logits','visible_blocks','token_topk','compress_ratio','block_indices','topk_workspace')
_state = None
_native = None
_wrapper = None
_math = None
_calls = 0
_shapes = Counter()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install():
    global _state, _native, _wrapper, _math
    if _state is not None:
        return evidence()
    helper = Path(__file__).resolve().parents[1]/'runtime_compat/classic_precision_qsa_determinism.py'
    if sha(helper) != HELPER_SOURCE_SHA:
        raise RuntimeError('The frozen original QSA mathematics source changed')
    meta_before = tuple(id(x) for x in sys.meta_path)
    spec = importlib.util.spec_from_file_location('_official_x_exact_qsa_math', helper)
    math = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(math)
    if math.status() is not None or math._finder is not None or tuple(id(x) for x in sys.meta_path) != meta_before:
        raise RuntimeError('Importing the original mathematics activated an unexpected hook')
    module = importlib.import_module(TARGET)
    source = Path(module.__file__).read_bytes()
    if hashlib.sha256(source).hexdigest() != NATIVE_SOURCE_SHA:
        raise RuntimeError('Official QSA source differs from the qualified version')
    native = module._topk
    expected = [x for x in compile(source,module.__file__,'exec',dont_inherit=True).co_consts
                if isinstance(x,CodeType) and x.co_name == '_topk']
    if not (inspect.isfunction(native) and native.__module__ == TARGET
            and native.__globals__ is vars(module) and len(expected) == 1
            and marshal.dumps(native.__code__) == marshal.dumps(expected[0])
            and tuple(inspect.signature(native).parameters) == PARAMETERS
            and not hasattr(native,'__wrapped__')):
        raise RuntimeError('Native QSA function was already replaced or its signature changed')
    _native = native
    _math = math

    def canonical_topk(logits, visible_blocks, token_topk, compress_ratio, block_indices, topk_workspace):
        global _calls
        _calls += 1
        _shapes[(tuple(logits.shape),str(logits.dtype),token_topk,compress_ratio)] += 1
        return math.exact_qsa_topk(logits,visible_blocks,token_topk,compress_ratio,block_indices,topk_workspace)

    _wrapper = canonical_topk
    module._topk = canonical_topk
    _state = {'status':'APPLIED_X_QSA','target_module':TARGET,'target_function':'_topk',
        'native_source_path':module.__file__,'native_source_sha256':NATIVE_SOURCE_SHA,
        'native_function_verified_before_replacement':True,'native_signature':str(inspect.signature(native)),
        'selector_source_path':__file__,'selector_source_sha256':sha(__file__),
        'mathematics_source_path':str(helper),'mathematics_source_sha256':HELPER_SOURCE_SHA,
        'mathematics_function':'exact_qsa_topk','algorithm':'stable_score_descending_index_ties_then_index_ascending',
        'helper_install_called':False,'helper_status_after_import':math.status(),
        'helper_import_added_meta_path_hook':False,'SDK_files_modified':False,
        'forward_processor_or_weights_changed':False,'topk_workspace_ignored_by_math':True}
    return evidence()


def evidence():
    if _state is None:
        raise RuntimeError('The explicit X-QSA worker extension was not installed')
    module = sys.modules[TARGET]
    if module._topk is not _wrapper or _math.status() is not None or _math._finder is not None:
        raise RuntimeError('Applied X-QSA identity changed or an old install hook became active')
    if sha(module.__file__) != NATIVE_SOURCE_SHA or sha(_math.__file__) != HELPER_SOURCE_SHA:
        raise RuntimeError('Pinned source identity changed')
    return {**_state,'actual_function_is_canonical_wrapper':True,
        'actual_function_code_sha256':hashlib.sha256(marshal.dumps(_wrapper.__code__)).hexdigest(),
        'actual_math_code_sha256':hashlib.sha256(marshal.dumps(_math.exact_qsa_topk.__code__)).hexdigest(),
        'actual_calls':_calls,'actual_call_shapes':[{'logits_shape':list(k[0]),'dtype':k[1],
            'token_topk':k[2],'compress_ratio':k[3],'count':v} for k,v in _shapes.items()]}


class XQSAWorkerExtension:
    def init_worker_extension(self):
        # vLLM calls this documented extension hook after device initialization
        # and before loading the model. No worker/model forward is overridden.
        install()

    def x_qsa_selector_evidence(self):
        return evidence()
