"""Small CPU codegen fixes for the pinned PyPTO 7302cd7 frontend.

Retains upstream IR and Torch codegen; fixes nested statement dispatch,
FP32 Tile dot products, E4M3FN cast/dtype checks and scalar min/max.
Also supplies the index sequence and FP32 trig operations used by MRoPE.
No upstream files or process-global dispatch tables are modified.
"""

import importlib

from pypto import DataType, ir


_upstream = importlib.import_module("pypto.debug.torch_codegen")


class _TileTorchCodegen(_upstream.TorchCodegen):
    def visit_stmt(self, stmt):
        # The C++ visitor otherwise traverses a sole nested loop's body without
        # entering the Python override that emits its loop header.
        if isinstance(stmt, ir.ForStmt):
            self.visit_for_stmt(stmt)
        elif isinstance(stmt, ir.SeqStmts):
            self.visit_seq_stmts(stmt)
        elif isinstance(stmt, ir.IfStmt):
            self.visit_if_stmt(stmt)
        else:
            super().visit_stmt(stmt)

    def visit_call(self, op):
        name = op.op.name
        if name in ("tile.matmul", "tile.matmul_acc"):
            args = [self._visit_expr_str(arg) for arg in op.args]
            offset = 1 if name == "tile.matmul_acc" else 0
            product = f"torch.matmul({args[offset]}.float(), {args[offset + 1]}.float())"
            if offset:
                self._expr_result = (
                    f"_acc_init({args[0]}, {product}, {args[3]})" if len(args) > 3
                    else f"({args[0]} + {product})"
                )
            else:
                self._expr_result = product
        elif name in ("tile.maximums", "tile.minimums"):
            value, scalar = [self._visit_expr_str(arg) for arg in op.args]
            limit = "min" if name == "tile.maximums" else "max"
            self._expr_result = f"torch.clamp({value}, {limit}={scalar})"
        elif name == "tile.cast" and op.type.dtype == DataType.FP8E4M3FN:
            # The frontend's round mode maps to Torch's E4M3FN RNE conversion.
            value = self._visit_expr_str(op.args[0])
            self._expr_result = f"{value}.to(torch.float8_e4m3fn)"
        elif name in ("tile.sin", "tile.cos"):
            value = self._visit_expr_str(op.args[0])
            self._expr_result = f"torch.{name.split('.')[1]}({value})"
        elif name == "tile.ci":
            start = self._visit_expr_str(op.args[0])
            shape = self._visit_expr_str(op.args[1])
            sign = "-" if op.kwargs.get("descending", False) else "+"
            dtype = _upstream._torch_dtype(op.type.dtype)
            self._expr_result = f"({start} {sign} torch.arange({shape}[-1], dtype={dtype})).reshape({shape})"
        else:
            super().visit_call(op)

    def _emit_shape_dtype_check(self, name, typ, *, shape=True):
        if isinstance(typ, (ir.TensorType, ir.TileType)) and typ.dtype == DataType.FP8E4M3FN:
            self._emit(f"assert {name}.dtype == torch.float8_e4m3fn")
            if shape:
                dims = ", ".join(self._visit_expr_str(dim) for dim in typ.shape)
                self._emit(f"assert {name}.shape == ({dims},)")
        else:
            super()._emit_shape_dtype_check(name, typ, shape=shape)


def torch_codegen(program, *, check_shapes=True):
    """Emit CPU Torch code for the project's sequential Tile programs."""
    generator = _TileTorchCodegen(check_shapes=check_shapes)
    generator.visit_program(program)
    return _upstream._PREAMBLE + "\n" + generator.get_output()
