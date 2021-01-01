from ast.arrays import ArrayAccess
from ast.data_types import Type_Int, Type_Vector
from ast.expr import BinOp
from ast.layouts import Layout_AoS, Layout_SoA
from ast.lit import Lit
from ast.loops import Iter
from ast.properties import Property


class Transform:
    reuse_expressions = {}

    def apply(ast, fn):
        ast.transform(fn)
        Transform.reuse_expressions = {}

    def flatten(ast):
        if isinstance(ast, BinOp):
            if ast.is_vector_property_access():
                layout = ast.lhs.layout()

                for i in ast.vector_indexes():
                    flat_index = None

                    if layout == Layout_AoS:
                        flat_index = ast.rhs * ast.sim.dimensions + i

                    elif layout == Layout_SoA:
                        flat_index = i * ast.sim.particle_capacity + ast.rhs

                    else:
                        raise Exception("Invalid property layout!")

                    ast.map_vector_index(i, flat_index)

        if isinstance(ast, Property):
            ast.flattened = True

        return ast

    def simplify(ast):
        if isinstance(ast, BinOp):
            sim = ast.lhs.sim

            if ast.op in ['+', '-'] and ast.rhs == 0:
                return ast.lhs

            if ast.op in ['+'] and ast.lhs == 0:
                return ast.rhs

            if ast.op in ['*', '/'] and ast.rhs == 1:
                return ast.lhs

            if ast.op == '*' and ast.lhs == 1:
                return ast.rhs

            if ast.op == '*' and ast.lhs == 0:
                return Lit(sim, 0 if ast.type() == Type_Int else 0.0)

        return ast

    def reuse_index_expressions(ast):
        if isinstance(ast, BinOp):
            iter_id = None

            if isinstance(ast.lhs, Iter):
                iter_id = ast.lhs.iter_id

            if isinstance(ast.rhs, Iter):
                iter_id = ast.rhs.iter_id

            if iter_id is not None:
                if iter_id in Transform.reuse_expressions:
                    item = [e for e in Transform.reuse_expressions[iter_id]
                            if ast.match(e)]
                    if item:
                        return item[0]

                else:
                    Transform.reuse_expressions[iter_id] = []

                Transform.reuse_expressions[iter_id].append(ast)

        return ast

    def reuse_expr_expressions(ast):
        if isinstance(ast, BinOp):
            expr_id = None

            if isinstance(ast.lhs, BinOp):
                expr_id = ast.lhs.expr_id

            if isinstance(ast.rhs, BinOp):
                expr_id = ast.rhs.expr_id

            if expr_id is not None:
                if expr_id in Transform.reuse_expressions:
                    item = [e for e in Transform.reuse_expressions[expr_id]
                            if ast.match(e)]
                    if item:
                        return item[0]

                else:
                    Transform.reuse_expressions[expr_id] = []

                Transform.reuse_expressions[expr_id].append(ast)

        return ast

    def reuse_array_access_expressions(ast):
        if isinstance(ast, BinOp):
            acc_id = None

            if isinstance(ast.lhs, ArrayAccess):
                acc_id = ast.lhs.acc_id

            if isinstance(ast.rhs, ArrayAccess):
                acc_id = ast.rhs.acc_id

            if acc_id is not None:
                if acc_id in Transform.reuse_expressions:
                    item = [e for e in Transform.reuse_expressions[acc_id]
                            if ast.match(e)]
                    if item:
                        return item[0]

                else:
                    Transform.reuse_expressions[acc_id] = []

                Transform.reuse_expressions[acc_id].append(ast)

        return ast

    def move_loop_invariant_expressions(ast):
        if isinstance(ast, BinOp):
            scope = ast.scope()
            if scope.level > 0:
                scope.add_expression(ast)

        return ast
