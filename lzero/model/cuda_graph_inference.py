"""
CUDA-graph replay for small-batch inference (MCTS network calls).

MCTS calls the network with tiny batches (one state per environment, e.g. 8), where the GPU time is dominated by
Python and kernel-launch overhead rather than by computation. Capturing the forward pass as a CUDA graph and
replaying it removes that overhead (~4-5x faster per call on an RTX 5080 for the Connect4 networks) and produces
bit-identical outputs.

A graph is captured lazily per (function, input shapes/dtypes) and replayed afterwards. It reads the module's
parameters and buffers in place, so optimizer steps and ``load_state_dict`` (both in-place) are picked up
automatically. Graphs are only used when it is safe:
    - the module is in eval mode (BatchNorm uses running statistics) and autograd is off,
    - all inputs are CUDA tensors and the batch is small (larger batches gain little and would create many graphs).
Otherwise the eager function is called, so enabling this never changes results.
"""
import logging
import weakref
from typing import Callable, Dict, Tuple

import torch

_EAGER = object()  # cache sentinel: capture failed for this key, always run eagerly

# Per-module graph caches, kept outside the module so that copy.deepcopy / pickling of the model is unaffected.
_CACHES: "weakref.WeakKeyDictionary[torch.nn.Module, Dict[tuple, tuple]]" = weakref.WeakKeyDictionary()
# One CUDA-graph memory pool per (module, device): all graphs of a module share their intermediate memory instead of
# each reserving its own (which costs hundreds of MB per process with ~30 graphs). Safe because the graphs are only
# replayed one at a time and their outputs are copied out right after each replay.
_POOLS: "weakref.WeakKeyDictionary[torch.nn.Module, Dict[torch.device, tuple]]" = weakref.WeakKeyDictionary()

MAX_GRAPH_BATCH = 64     # batches larger than this run eagerly
MAX_GRAPHS_PER_MODULE = 128


def graphs_usable(module: torch.nn.Module, *inputs: torch.Tensor) -> bool:
    return (
        not module.training
        and not torch.is_grad_enabled()
        and not torch.is_inference_mode_enabled()
        and all(isinstance(x, torch.Tensor) and x.is_cuda for x in inputs)
        and inputs[0].shape[0] <= MAX_GRAPH_BATCH
        and not torch.cuda.is_current_stream_capturing()
    )


def run_graphed(module: torch.nn.Module, name: str, fn: Callable[..., Tuple[torch.Tensor, ...]],
                *inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """
    Overview:
        Run ``fn(*inputs)`` (which must return a tuple of tensors) via a cached CUDA graph if possible,
        else eagerly. Returned tensors are fresh copies (the graph's static outputs are overwritten on replay).
    """
    if not graphs_usable(module, *inputs):
        return fn(*inputs)
    cache = _CACHES.setdefault(module, {})
    key = (name, inputs[0].device) + tuple((tuple(x.shape), x.dtype) for x in inputs)
    entry = cache.get(key)
    if entry is _EAGER:
        return fn(*inputs)
    if entry is None:
        if len(cache) >= MAX_GRAPHS_PER_MODULE:
            return fn(*inputs)
        try:
            pools = _POOLS.setdefault(module, {})
            if inputs[0].device not in pools:
                pools[inputs[0].device] = torch.cuda.graph_pool_handle()
            entry = cache[key] = _capture(fn, inputs, pools[inputs[0].device])
        except Exception as e:  # e.g. the function syncs with the host: never retry this shape, run eagerly
            cache[key] = _EAGER
            logging.warning(f"CUDA graph capture of {type(module).__name__}.{name} failed ({e}); running eagerly.")
            return fn(*inputs)
    graph, static_inputs, static_outputs = entry
    for dst, src in zip(static_inputs, inputs):
        dst.copy_(src)
    graph.replay()
    return tuple(out.clone() for out in static_outputs)


def _capture(fn: Callable[..., Tuple[torch.Tensor, ...]], inputs: Tuple[torch.Tensor, ...], pool) -> tuple:
    static_inputs = tuple(x.detach().clone() for x in inputs)
    # Warm up on a side stream (required before capture; also lets cuDNN pick its algorithms).
    stream = torch.cuda.Stream(device=inputs[0].device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            fn(*static_inputs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=pool):
        static_outputs = fn(*static_inputs)
    if not isinstance(static_outputs, tuple) or not all(isinstance(o, torch.Tensor) for o in static_outputs):
        raise TypeError("graphed function must return a tuple of tensors")
    return graph, static_inputs, static_outputs


def clear_graphs(module: torch.nn.Module) -> None:
    """Drop all captured graphs of ``module`` (e.g. after moving it to another device)."""
    _CACHES.pop(module, None)
    _POOLS.pop(module, None)
