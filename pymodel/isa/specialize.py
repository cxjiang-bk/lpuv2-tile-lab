"""Use the pinned PyPTO signature expansion for ordinary multi-shape callers."""

import importlib


_jit = importlib.import_module("pypto.jit.decorator")


class _ShapeJIT(_jit.JITFunction):
    def _build_contexts(self, tensors, scalars, constants, dynamic_maps):
        # Upstream currently enables this path only for graphs with extern
        # kernels. Ordinary Attention also reuses helpers at different shapes.
        return _jit._build_signature_contexts(
            self, tensors, scalars, constants, dynamic_maps,
            discover=_jit._discover_dep_bindings,
            allocate_names=_jit._allocate_generated_names,
            resolve_peers=_jit._resolve_peer_func_names,
        )


def specialize(kernel, *args, **kwargs):
    """Build Tile IR while keeping a distinct helper per actual call signature."""
    entry = _ShapeJIT(kernel._func, func_type=kernel._func_type,
                      level=kernel._level, auto_scope=kernel._auto_scope)
    return entry.specialize(*args, **kwargs)
